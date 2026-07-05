"""Training + validation loop for the recognition model.

Adapted from Clova AI's deep-text-recognition-benchmark (Apache-2.0), which is
the basis of EasyOCR's trainer. Call `train(opt)` from an entry script.
"""
import os
import re
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from nltk.metrics.distance import edit_distance

from .utils import CTCLabelConverter, AttnLabelConverter, Averager
from .dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from .model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation(model, criterion, evaluation_loader, converter, opt):
    """Validation or evaluation."""
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if "CTC" in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            cost = criterion(
                preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss
            )

            if opt.decode == "greedy":
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)
            else:
                preds_str = converter.decode_beamsearch(preds, beamWidth=2)
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, : text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(
                preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1)
            )

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if "Attn" in opt.Prediction:
                gt = gt[: gt.find("[s]")]
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]

            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = "0123456789abcdefghijklmnopqrstuvwxyz"
                out_of_alphanumeric_case_insensitve = f"[^{alphanumeric_case_insensitve}]"
                pred = re.sub(out_of_alphanumeric_case_insensitve, "", pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, "", gt)

            if pred == gt:
                n_correct += 1

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except IndexError:
                confidence_score = 0
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 normalized edit distance

    return (
        valid_loss_avg.val(),
        accuracy,
        norm_ED,
        preds_str,
        confidence_score_list,
        labels,
        infer_time,
        length_of_data,
    )


def train(opt, show_number=2, amp=False):
    os.makedirs(f"./saved_models/{opt.experiment_name}", exist_ok=True)

    if not opt.data_filtering_off:
        print("Filtering the images containing characters not in `opt.character`")
        print("Filtering the images whose label is longer than `opt.batch_max_length`")

    opt.select_data = opt.select_data.split("-") if isinstance(opt.select_data, str) else opt.select_data
    opt.batch_ratio = opt.batch_ratio.split("-") if isinstance(opt.batch_ratio, str) else opt.batch_ratio

    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f"./saved_models/{opt.experiment_name}/log_dataset.txt", "a")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(opt.batch_size, len(valid_dataset)), shuffle=True,
        num_workers=int(opt.workers), collate_fn=AlignCollate_valid, pin_memory=True,
    )
    log.write(valid_dataset_log)
    print("-" * 80)
    log.write("-" * 80 + "\n")
    log.close()

    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print(
        "model input parameters",
        opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
        opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation,
        opt.FeatureExtraction, opt.SequenceModeling, opt.Prediction,
    )

    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"Skip {name} as it is already initialized")
            continue
        try:
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param)
        except Exception:
            if "weight" in name:
                param.data.fill_(1)
            continue

    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != "":
        print(f"loading pretrained model from {opt.saved_model}")
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO]

    loss_avg = Averager()

    filtered_parameters = [p for p in filter(lambda p: p.requires_grad, model.parameters())]
    params_num = sum([np.prod(p.size()) for p in filtered_parameters])
    print("Trainable params num : ", params_num)

    if opt.adam:
        optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = torch.optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    with open(f"./saved_models/{opt.experiment_name}/opt.txt", "a") as opt_file:
        opt_log = "------------ Options -------------\n"
        args = vars(opt)
        for k, v in args.items():
            opt_log += f"{str(k)}: {str(v)}\n"
        opt_log += "---------------------------------------\n"
        print(opt_log)
        opt_file.write(opt_log)

    start_iter = 0
    if opt.saved_model != "":
        try:
            start_iter = int(opt.saved_model.split("_")[-1].split(".")[0])
            print(f"continue to train, start_iter: {start_iter}")
        except Exception:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    while True:
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        optimizer.zero_grad(set_to_none=True)
        if amp:
            with torch.cuda.amp.autocast():
                if "CTC" in opt.Prediction:
                    preds = model(image, text)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.log_softmax(2).permute(1, 0, 2)
                    cost = criterion(preds, text, preds_size, length)
                else:
                    preds = model(image, text[:, :-1])
                    target = text[:, 1:]
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if "CTC" in opt.Prediction:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
            else:
                preds = model(image, text[:, :-1])
                target = text[:, 1:]
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        loss_avg.add(cost)

        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            with open(f"./saved_models/{opt.experiment_name}/log_train.txt", "a") as log:
                model.eval()
                with torch.no_grad():
                    (valid_loss, current_accuracy, current_norm_ED, preds, confidence_score,
                     labels_v, infer_time, length_of_data) = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                loss_log = (
                    f"[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, "
                    f"Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}"
                )
                loss_avg.reset()

                current_model_log = (
                    f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, '
                    f'{"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
                )

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f"./saved_models/{opt.experiment_name}/best_accuracy.pth")
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f"./saved_models/{opt.experiment_name}/best_norm_ED.pth")
                best_model_log = (
                    f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, '
                    f'{"Best_norm_ED":17s}: {best_norm_ED:0.4f}'
                )

                loss_model_log = f"{loss_log}\n{current_model_log}\n{best_model_log}"
                print(loss_model_log)
                log.write(loss_model_log + "\n")

                dashed_line = "-" * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
                start = random.randint(0, max(0, len(labels_v) - show_number)) if len(labels_v) > show_number else 0
                for gt, pred, confidence in zip(
                    labels_v[start:start + show_number],
                    preds[start:start + show_number],
                    confidence_score[start:start + show_number],
                ):
                    if "Attn" in opt.Prediction:
                        gt = gt[: gt.find("[s]")]
                        pred = pred[: pred.find("[s]")]
                    predicted_result_log += f"{gt[:25]:25s} | {pred[:25]:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                predicted_result_log += f"{dashed_line}"
                print(predicted_result_log)
                log.write(predicted_result_log + "\n")

        if (iteration + 1) % 1e5 == 0:
            torch.save(model.state_dict(), f"./saved_models/{opt.experiment_name}/iter_{iteration + 1}.pth")

        if (iteration + 1) == opt.num_iter:
            print("end the training")
            torch.save(model.state_dict(), f"./saved_models/{opt.experiment_name}/iter_{iteration + 1}.pth")
            break
        iteration += 1
