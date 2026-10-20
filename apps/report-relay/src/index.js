// KhmerClipper report relay: files app reports as GitHub issues in the private
// reports repo, falling back to email (Resend) if GitHub is unavailable.

const MAX_BODY = 3_000_000;
const MAX_IMAGE = 2_000_000;
const MAX_TEXT = 65_536;
const MAX_EMAIL = 254;
const ID_RE = /^[0-9a-f]{32}$/;
const META_RE = /^[^\r\n]{1,100}$/;
const B64_RE = /^[A-Za-z0-9+/]*={0,2}$/;
const STRING_FIELDS = ['signature', 'details', 'image_png_b64', 'ocr_text', 'correction', 'message', 'email'];
const GH = 'https://api.github.com';

const REQUIRED = {
  crash: { signature: 'string', details: 'string' },
  capture: { image_png_b64: 'string', ocr_text: 'string', correction: 'string', meta: 'object' },
  feedback: { message: 'string', email: 'string' },
};

const json = (status, obj) => Response.json(obj, { status });

class TooLarge extends Error {}

export async function handle(request, env, fetchFn) {
  const url = new URL(request.url);
  if (request.method !== 'POST' || url.pathname !== '/report') return json(404, { error: 'not found' });
  if (request.headers.get('X-KhmerClipper-Key') !== env.APP_KEY) return json(401, { error: 'unauthorized' });

  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const { success } = await env.RATE_LIMITER.limit({ key: ip });
  if (!success) return json(429, { error: 'rate limited' });

  const declared = Number(request.headers.get('Content-Length'));
  if (declared > MAX_BODY) return json(413, { error: 'too large' });
  const text = await request.text();
  if (new TextEncoder().encode(text).length > MAX_BODY) return json(413, { error: 'too large' });

  let r;
  try {
    r = JSON.parse(text);
  } catch {
    return json(400, { error: 'bad json' });
  }
  try {
    if (!validate(r)) return json(400, { error: 'invalid report' });
  } catch (e) {
    if (e instanceof TooLarge) return json(413, { error: 'too large' });
    throw e;
  }

  const { title, body } = render(r, env.REPORTS_REPO);
  try {
    await fileOnGithub(r, title, body, env, fetchFn);
    return json(200, { ok: true, via: 'github' });
  } catch {
    // fall through to email
  }
  try {
    const res = await fetchFn('https://api.resend.com/emails', {
      method: 'POST',
      headers: { Authorization: `Bearer ${env.RESEND_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        from: 'KhmerClipper <onboarding@resend.dev>',
        to: [env.EMAIL_TO],
        subject: title,
        text: body,
        attachments: r.type === 'capture' ? [{ filename: `${r.id}.png`, content: r.image_png_b64 }] : [],
      }),
    });
    if (res.ok) return json(200, { ok: true, via: 'email' });
  } catch {
    // fall through
  }
  return json(502, { error: 'delivery failed' });
}

export default { fetch: (req, env) => handle(req, env, fetch) };

// Returns true if valid; throws TooLarge if a limit is exceeded; false otherwise.
function validate(r) {
  if (!r || typeof r !== 'object' || Array.isArray(r)) return false;
  const spec = REQUIRED[r.type];
  if (!spec || typeof r.id !== 'string' || !ID_RE.test(r.id)) return false;
  for (const k of ['app_version', 'os', 'engine']) {
    if (typeof r[k] !== 'string' || !META_RE.test(r[k])) return false;
  }
  // Any present field must have the right type, even if this report type ignores it.
  for (const k of STRING_FIELDS) if (r[k] !== undefined && typeof r[k] !== 'string') return false;
  if (r.meta !== undefined && (!r.meta || typeof r.meta !== 'object' || Array.isArray(r.meta))) return false;
  for (const [k, t] of Object.entries(spec)) {
    const v = r[k];
    if (t === 'object' ? !v || typeof v !== 'object' || Array.isArray(v) : typeof v !== t) return false;
  }
  for (const k of ['app_version', 'os', 'engine', 'signature', 'details', 'ocr_text', 'correction', 'message']) {
    if (typeof r[k] === 'string' && r[k].length > MAX_TEXT) throw new TooLarge();
  }
  if (r.meta && JSON.stringify(r.meta).length > MAX_TEXT) throw new TooLarge();
  if (r.email !== undefined && r.email.length > MAX_EMAIL) throw new TooLarge();
  if (r.type === 'capture' && (r.image_png_b64.length % 4 !== 0 || !B64_RE.test(r.image_png_b64))) return false;
  if (r.image_png_b64 !== undefined && decodedSize(r.image_png_b64) > MAX_IMAGE) throw new TooLarge();
  return true;
}

function decodedSize(b64) {
  const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
  return Math.floor((b64.length * 3) / 4) - pad;
}

// Wrap user text in a ~ fence longer than any ~ run inside it.
function fence(s) {
  const longest = Math.max(0, ...(s.match(/~+/g) || []).map((m) => m.length));
  const f = '~'.repeat(Math.max(3, longest + 1));
  return `${f}\n${s}\n${f}`;
}

function oneLine(s) {
  return s.replace(/\s+/g, ' ').trim().slice(0, 60);
}

function render(r, repo) {
  const head = fence(`App ${r.app_version} · ${r.os} · ${r.engine}`);
  if (r.type === 'capture') {
    const parts = [
      head,
      `![capture](https://github.com/${repo}/blob/main/reports/${r.id}.png?raw=true)`,
      'KhmerClipper read:',
      fence(r.ocr_text),
    ];
    if (r.correction !== '') parts.push('Should say:', fence(r.correction));
    return { title: `Wrong text: ${oneLine(r.ocr_text)}`, body: parts.join('\n\n') };
  }
  if (r.type === 'crash') {
    return { title: `Crash: ${oneLine(r.signature)}`, body: [head, fence(r.details)].join('\n\n') };
  }
  const parts = [head, fence(r.message)];
  if (r.email !== '') parts.push('Reply to:', fence(r.email));
  return { title: `Feedback: ${oneLine(r.message)}`, body: parts.join('\n\n') };
}

async function sha256Hex(s) {
  const d = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function b64Utf8(s) {
  let bin = '';
  for (const b of new TextEncoder().encode(s)) bin += String.fromCharCode(b);
  return btoa(bin);
}

async function fileOnGithub(r, title, body, env, fetchFn) {
  const repo = env.REPORTS_REPO;
  const gh = async (path, method, payload, tolerate) => {
    const res = await fetchFn(GH + path, {
      method,
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        'User-Agent': 'khmerclipper-report-relay',
        Accept: 'application/vnd.github+json',
        'Content-Type': 'application/json',
      },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
    if (!res.ok && res.status !== tolerate) throw new Error(`github ${method} ${path}: ${res.status}`);
    return res;
  };

  if (r.type === 'capture') {
    const sidecar = {
      ocr_text: r.ocr_text, correction: r.correction, meta: r.meta,
      app_version: r.app_version, os: r.os, engine: r.engine,
    };
    // 422 = file already exists (client retry): carry on.
    await gh(`/repos/${repo}/contents/reports/${r.id}.png`, 'PUT',
      { message: `report ${r.id} image`, content: r.image_png_b64 }, 422);
    await gh(`/repos/${repo}/contents/reports/${r.id}.json`, 'PUT',
      { message: `report ${r.id} data`, content: b64Utf8(JSON.stringify(sidecar)) }, 422);
  } else if (r.type === 'crash') {
    const fp = (await sha256Hex(`${r.app_version}\n${r.signature}`)).slice(0, 16);
    const q = encodeURIComponent(`repo:${repo} is:issue is:open label:crash "fp:${fp}"`);
    const found = await (await gh(`/search/issues?q=${q}`, 'GET')).json();
    if (found.total_count > 0) {
      await gh(`/repos/${repo}/issues/${found.items[0].number}/comments`, 'POST',
        { body: fence(`+1 · ${r.app_version} · ${r.os}`) });
      return;
    }
    body = `<!-- fp:${fp} -->\n${body}`;
  }
  await gh(`/repos/${repo}/issues`, 'POST', { title, body, labels: [r.type] });
}
