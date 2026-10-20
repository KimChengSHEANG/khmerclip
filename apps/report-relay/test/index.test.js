import { test } from 'node:test';
import assert from 'node:assert/strict';
import { handle } from '../src/index.js';

const ID = 'a'.repeat(32);
const REPO = 'KimChengSHEANG/khmerclipper-reports';

function makeEnv(over = {}) {
  return {
    GITHUB_TOKEN: 'gh-token',
    RESEND_API_KEY: 're-key',
    APP_KEY: 'kc-reports-1',
    REPORTS_REPO: REPO,
    EMAIL_TO: 'cheng900@gmail.com',
    RATE_LIMITER: { limit: async () => ({ success: true }) },
    ...over,
  };
}

// route(url, init) -> Response | undefined; unhandled GitHub writes return 201, search 200 empty.
function makeFetch(route = () => undefined) {
  const calls = [];
  const fn = async (url, init = {}) => {
    calls.push({ url, method: init.method || 'GET', init, json: init.body ? JSON.parse(init.body) : null });
    const r = route(url, init, calls.length);
    if (r) return r;
    if (url.includes('/search/issues')) return Response.json({ total_count: 0, items: [] });
    if (url.includes('resend.com')) return Response.json({ id: 'x' });
    return Response.json({}, { status: 201 });
  };
  fn.calls = calls;
  return fn;
}

function req(body, { key = 'kc-reports-1', method = 'POST', path = '/report', raw } = {}) {
  const headers = { 'content-type': 'application/json', 'CF-Connecting-IP': '1.2.3.4' };
  if (key) headers['X-KhmerClipper-Key'] = key;
  return new Request('https://relay.example' + path, {
    method,
    headers,
    body: method === 'GET' ? undefined : raw ?? JSON.stringify(body),
  });
}

const base = { id: ID, app_version: '0.2.0', os: 'macOS 15', engine: 'trocr' };
const capture = (o = {}) => ({
  ...base, type: 'capture', image_png_b64: btoa('png-bytes'), ocr_text: 'ស្តី', correction: 'សួស្តី', meta: { w: 1 }, ...o,
});
const crash = (o = {}) => ({ ...base, type: 'crash', signature: 'ValueError at ocr/trocr.py:decode', details: 'Traceback...', ...o });
const feedback = (o = {}) => ({ ...base, type: 'feedback', message: 'hello', email: '', ...o });

test('rejects wrong key', async () => {
  const f = makeFetch();
  assert.equal((await handle(req(capture(), { key: 'nope' }), makeEnv(), f)).status, 401);
  assert.equal((await handle(req(capture(), { key: null }), makeEnv(), f)).status, 401);
  assert.equal(f.calls.length, 0);
});

test('only POST /report', async () => {
  const f = makeFetch();
  assert.equal((await handle(req(null, { method: 'GET' }), makeEnv(), f)).status, 404);
  assert.equal((await handle(req(capture(), { path: '/other' }), makeEnv(), f)).status, 404);
});

test('rejects bad id and unknown type', async () => {
  const f = makeFetch();
  assert.equal((await handle(req(capture({ id: 'ABC' })), makeEnv(), f)).status, 400);
  assert.equal((await handle(req(capture({ id: 'A'.repeat(32) })), makeEnv(), f)).status, 400);
  assert.equal((await handle(req(capture({ type: 'bogus' })), makeEnv(), f)).status, 400);
  assert.equal((await handle(req(feedback({ message: undefined })), makeEnv(), f)).status, 400);
  assert.equal((await handle(req(null, { raw: 'not json' }), makeEnv(), f)).status, 400);
  assert.equal(f.calls.length, 0);
});

test('present field of wrong type is 400, not a crash', async () => {
  const f = makeFetch();
  for (const extra of [{ email: null }, { image_png_b64: 5 }, { details: 1 }, { meta: [] }, { message: {} }]) {
    assert.equal((await handle(req(crash(extra)), makeEnv(), f)).status, 400, JSON.stringify(extra));
  }
  assert.equal(f.calls.length, 0);
});

test('app_version/os/engine must be single-line, 1-100 chars', async () => {
  const f = makeFetch();
  for (const o of [{ os: 'a\nb' }, { os: 'a\rb' }, { engine: '' }, { app_version: 'x'.repeat(101) }]) {
    assert.equal((await handle(req(capture(o)), makeEnv(), f)).status, 400, JSON.stringify(o));
  }
  assert.equal(f.calls.length, 0);
  assert.equal((await handle(req(capture({ os: 'x'.repeat(100) })), makeEnv(), makeFetch())).status, 200);
});

test('capture image must be valid base64', async () => {
  const f = makeFetch();
  for (const b of ['abc', 'ab!d', 'a=bc', '====']) {
    assert.equal((await handle(req(capture({ image_png_b64: b })), makeEnv(), f)).status, 400, b);
  }
  assert.equal(f.calls.length, 0);
});

test('rejects oversized image', async () => {
  const big = btoa('a'.repeat(2_000_001));
  const f = makeFetch();
  assert.equal((await handle(req(capture({ image_png_b64: big })), makeEnv(), f)).status, 413);
  const ok = btoa('a'.repeat(2_000_000));
  assert.equal((await handle(req(capture({ image_png_b64: ok })), makeEnv(), makeFetch())).status, 200);
});

test('rejects oversized text field and email', async () => {
  const f = makeFetch();
  assert.equal((await handle(req(feedback({ message: 'a'.repeat(65_537) })), makeEnv(), f)).status, 413);
  assert.equal((await handle(req(feedback({ email: 'a'.repeat(255) })), makeEnv(), f)).status, 413);
  assert.equal(f.calls.length, 0);
});

test('rejects oversized body', async () => {
  const f = makeFetch();
  const r = await handle(req(null, { raw: ' '.repeat(3_000_001) }), makeEnv(), f);
  assert.equal(r.status, 413);
});

test('rate limited', async () => {
  const seen = [];
  const env = makeEnv({ RATE_LIMITER: { limit: async (o) => (seen.push(o), { success: false }) } });
  const f = makeFetch();
  assert.equal((await handle(req(capture()), env, f)).status, 429);
  assert.deepEqual(seen, [{ key: '1.2.3.4' }]);
  assert.equal(f.calls.length, 0);
});

test('capture uploads png + sidecar then opens labeled issue', async () => {
  const f = makeFetch();
  const res = await handle(req(capture()), makeEnv(), f);
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { ok: true, via: 'github' });
  const [a, b, c] = f.calls;
  const api = `https://api.github.com/repos/${REPO}`;
  assert.equal(a.method, 'PUT');
  assert.equal(a.url, `${api}/contents/reports/${ID}.png`);
  assert.equal(a.json.content, btoa('png-bytes'));
  assert.equal(b.method, 'PUT');
  assert.equal(b.url, `${api}/contents/reports/${ID}.json`);
  const sidecar = JSON.parse(atob(b.json.content));
  assert.deepEqual(Object.keys(sidecar).sort(), ['app_version', 'correction', 'engine', 'meta', 'ocr_text', 'os']);
  assert.equal(c.method, 'POST');
  assert.equal(c.url, `${api}/issues`);
  assert.deepEqual(c.json.labels, ['capture']);
  assert.ok(c.json.title.startsWith('Wrong text: '));
  assert.ok(c.json.body.includes(`reports/${ID}.png?raw=true`));
  assert.ok(c.json.body.includes('~~~\nApp 0.2.0 · macOS 15 · trocr\n~~~'));
  assert.ok(c.json.body.includes('Should say'));
  assert.equal(a.init.headers.Authorization, 'Bearer gh-token');
  assert.equal(a.init.headers['User-Agent'], 'khmerclipper-report-relay');
});

test('capture omits Should say when correction empty', async () => {
  const f = makeFetch();
  await handle(req(capture({ correction: '' })), makeEnv(), f);
  assert.ok(!f.calls[2].json.body.includes('Should say'));
});

test('capture tolerates 422 on re-upload', async () => {
  const f = makeFetch((url, init) => (init.method === 'PUT' ? Response.json({}, { status: 422 }) : undefined));
  const res = await handle(req(capture()), makeEnv(), f);
  assert.equal(res.status, 200);
  assert.equal((await res.json()).via, 'github');
  assert.equal(f.calls.filter((c) => c.url.endsWith('/issues')).length, 1);
});

test('crash with known fingerprint comments instead of opening', async () => {
  const f = makeFetch((url) =>
    url.includes('/search/issues') ? Response.json({ total_count: 1, items: [{ number: 7 }] }) : undefined);
  const res = await handle(req(crash()), makeEnv(), f);
  assert.equal(res.status, 200);
  const posts = f.calls.filter((c) => c.method === 'POST');
  assert.equal(posts.length, 1);
  assert.equal(posts[0].url, `https://api.github.com/repos/${REPO}/issues/7/comments`);
  assert.equal(posts[0].json.body, '~~~\n+1 · 0.2.0 · macOS 15\n~~~');
  const q = new URL(f.calls[0].url).searchParams.get('q');
  assert.match(q, new RegExp(`^repo:${REPO} is:issue is:open label:crash "fp:[0-9a-f]{16}"$`));
});

test('new crash opens issue with fp marker', async () => {
  const f = makeFetch();
  const res = await handle(req(crash()), makeEnv(), f);
  assert.equal(res.status, 200);
  const issue = f.calls.find((c) => c.url.endsWith('/issues'));
  assert.match(issue.json.body, /<!-- fp:[0-9a-f]{16} -->/);
  assert.deepEqual(issue.json.labels, ['crash']);
  assert.equal(issue.json.title, 'Crash: ValueError at ocr/trocr.py:decode');
  assert.ok(issue.json.body.includes('~~~\nTraceback...\n~~~'));
});

test('title is single-line and truncated to 60', async () => {
  const f = makeFetch();
  await handle(req(feedback({ message: 'line1\nline2 ' + 'x'.repeat(100) })), makeEnv(), f);
  const t = f.calls[0].json.title;
  assert.ok(t.startsWith('Feedback: line1 line2 '));
  assert.equal(t.length, 'Feedback: '.length + 60);
});

test('user text is fenced and cannot break out', async () => {
  const f = makeFetch();
  await handle(req(feedback({ message: 'hi ~~~\n@octocat **x**' })), makeEnv(), f);
  assert.ok(f.calls[0].json.body.includes('~~~~\nhi ~~~\n@octocat **x**\n~~~~'));
});

test('feedback shows Reply to only with email', async () => {
  const f = makeFetch();
  await handle(req(feedback({ email: 'a@b.co' })), makeEnv(), f);
  assert.ok(f.calls[0].json.body.includes('Reply to:'));
  assert.ok(f.calls[0].json.body.includes('~~~\na@b.co\n~~~'));
  const g = makeFetch();
  await handle(req(feedback()), makeEnv(), g);
  assert.ok(!g.calls[0].json.body.includes('Reply to:'));
});

test('github failure falls back to email with attachment', async () => {
  const f = makeFetch((url, init) =>
    url.endsWith('/issues') && init.method === 'POST' ? Response.json({}, { status: 500 }) : undefined);
  const res = await handle(req(capture()), makeEnv(), f);
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { ok: true, via: 'email' });
  const mail = f.calls.at(-1);
  assert.equal(mail.url, 'https://api.resend.com/emails');
  assert.equal(mail.init.headers.Authorization, 'Bearer re-key');
  assert.deepEqual(mail.json.to, ['cheng900@gmail.com']);
  assert.equal(mail.json.from, 'KhmerClipper <onboarding@resend.dev>');
  assert.equal(mail.json.attachments[0].filename, `${ID}.png`);
  assert.equal(mail.json.attachments[0].content, btoa('png-bytes'));
  assert.ok(mail.json.subject.startsWith('Wrong text: '));
});

test('github throw falls back to email without attachment for non-capture', async () => {
  const f = makeFetch((url) => {
    if (url.includes('api.github.com')) throw new Error('network');
  });
  const res = await handle(req(crash()), makeEnv(), f);
  assert.equal((await res.json()).via, 'email');
  assert.deepEqual(f.calls.at(-1).json.attachments, []);
});

test('both fail → 502', async () => {
  const f = makeFetch(() => Response.json({}, { status: 500 }));
  assert.equal((await handle(req(capture()), makeEnv(), f)).status, 502);
});
