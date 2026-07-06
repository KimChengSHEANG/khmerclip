"""Dataset utilities: LMDB dataset, resizing/padding collate, balanced sampler.

Adapted from Clova AI's deep-text-recognition-benchmark (Apache-2.0).
"""
import os
import sys
import math
import six
from itertools import accumulate as _accumulate

import lmdb
import numpy as np
import torch

from khmer_ocr.config import filter_label_to_charset, label_in_charset, normalize_label
from khmer_ocr.data import lmdb_sample_count
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as transforms


class Batch_Balanced_Dataset(object):
    """Sample data from several datasets according to `batch_ratio`."""

    def __init__(self, opt):
        log = open(os.path.join(opt.experiment_dir, "log_dataset.txt"), "a")
        dashed = "-" * 80
        print(dashed)
        log.write(dashed + "\n")
        print(
            f"dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\n"
            f"opt.batch_ratio: {opt.batch_ratio}"
        )
        log.write(
            f"dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\n"
            f"opt.batch_ratio: {opt.batch_ratio}\n"
        )
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed)
            log.write(dashed + "\n")
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length : offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = (
                f"num total samples of {selected_d}: {total_number_dataset} x "
                f"{opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n"
                f"num samples of {selected_d} per batch: {opt.batch_size} x "
                f"{float(batch_ratio_d)} (batch_ratio) = {_batch_size}"
            )
            print(selected_d_log)
            log.write(selected_d_log + "\n")
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size, shuffle=True,
                num_workers=int(opt.workers), collate_fn=_AlignCollate,
                pin_memory=True, drop_last=True,
            )
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f"{dashed}\n"
        batch_size_sum = "+".join(batch_size_list)
        Total_batch_size_log += f"Total_batch_size: {batch_size_sum} = {Total_batch_size}\n"
        Total_batch_size_log += dashed
        opt.batch_size = Total_batch_size
        print(Total_batch_size_log)
        log.write(Total_batch_size_log + "\n")
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass
        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data="/"):
    """Recursively collect every LMDB dataset found under `root`."""
    dataset_list = []
    dataset_log = f"dataset_root:    {root}\t dataset: {select_data[0]}"
    print(dataset_log)
    dataset_log += "\n"
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break
            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
                print(sub_log)
                dataset_log += f"{sub_log}\n"
                dataset_list.append(dataset)
    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        self._charset = set(opt.character)
        self.env = lmdb.open(
            root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False
        )
        if not self.env:
            print(f"cannot create lmdb from {root}")
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = lmdb_sample_count(self.env, txn, root)
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb keys start at 1
                    label_key = "label-%09d".encode() % index
                    label = normalize_label(txn.get(label_key).decode("utf-8"))

                    if len(label) > self.opt.batch_max_length:
                        continue
                    if not label_in_charset(label, self._charset):
                        continue
                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = normalize_label(txn.get(label_key).decode("utf-8"))
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert("RGB")
                else:
                    img = Image.open(buf).convert("L")
            except IOError:
                print(f"Corrupted image for {index}")
                if self.opt.rgb:
                    img = Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new("L", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"

            if not self.opt.sensitive:
                label = label.lower()

            label = filter_label_to_charset(label, self._charset)

        return (img, label)


class RawDataset(Dataset):
    """Read images from a folder for inference."""

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    self.image_path_list.append(os.path.join(dirpath, name))
        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert("RGB")
            else:
                img = Image.open(self.image_path_list[index]).convert("L")
        except IOError:
            print(f"Corrupted image for {index}")
            if self.opt.rgb:
                img = Image.new("RGB", (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new("L", (self.opt.imgW, self.opt.imgH))
        return (img, self.image_path_list[index])


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust=0.0):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # keep aspect ratio, pad the rest
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == "RGB" else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target=self.contrast_adjust)
                    image = Image.fromarray(image, "L")
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)
                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / max(1, (high - low))
        img = (img - low + 25) * ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / np.maximum(10, high + low), high, low


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
