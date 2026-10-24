// Server route helper. Never import this module into a client component.
const routes = new Map([
  ['verify', ['POST']], ['api/v1/verify', ['POST']], ['health', ['GET']],
  ['auth/login', ['POST']], ['auth/register', ['POST']], ['auth/google', ['POST']],
  ['auth/me', ['GET']], ['auth/logout', ['POST']],
]);
const json = (payload, status) => Response.json(payload, { status, headers: { 'Cache-Control': 'no-store' } });

function untilAborted(promise, signal) {
  return new Promise((resolve, reject) => {
    const aborted = () => reject(new Error('Request aborted'));
    if (signal.aborted) aborted();
    else signal.addEventListener('abort', aborted, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', aborted));
  });
}

async function boundedText(stream, limit, signal) {
  if (!stream) return '';
  const reader = stream.getReader(), decoder = new TextDecoder();
  let bytes = 0, result = '', complete = false;
  try {
    while (true) {
      const { done, value } = await untilAborted(reader.read(), signal);
      if (done) { complete = true; return result + decoder.decode(); }
      bytes += value.byteLength;
      if (bytes > limit) throw new RangeError('Payload too large');
      result += decoder.decode(value, { stream: true });
    }
  } finally {
    if (!complete) reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function proxyBackend(request, segments, options = {}) {
  if (!Array.isArray(segments) || !segments.length || segments.some(part => typeof part !== 'string' || !/^[a-z0-9-]+$/.test(part))) {
    return json({ detail: 'Unknown backend route' }, 404);
  }
  const path = segments.join('/'), methods = routes.get(path);
  if (!methods) return json({ detail: 'Unknown backend route' }, 404);
  if (!methods.includes(request.method)) return json({ detail: 'Method not allowed' }, 405);
  const query = new URL(request.url).search;
  if (query && !(path === 'health' && query === '?deep=true')) return json({ detail: 'Unsupported backend query' }, 400);
  let base;
  try {
    base = new URL(options.origin ?? process.env.BACKEND_URL);
    const localHttp = process.env.NODE_ENV !== 'production' && base.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname);
    if (base.username || base.password || base.search || base.hash || base.pathname !== '/' || (base.protocol !== 'https:' && !localHttp)) throw new Error();
  } catch { return json({ detail: 'Backend origin is not configured correctly' }, 503); }
  const headers = new Headers({ Accept: 'application/json' });
  for (const name of ['authorization', 'content-type', 'x-request-id']) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const controller = new AbortController(), cancel = () => controller.abort();
  const timer = setTimeout(cancel, options.timeoutMs ?? 280_000);
  request.signal.addEventListener('abort', cancel, { once: true });
  if (request.signal.aborted) cancel();
  try {
    if (controller.signal.aborted) throw new Error('Request aborted');
    let body;
    if (request.method !== 'GET') {
      if (request.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') return json({ detail: 'Expected a JSON request' }, 415);
      try { body = await boundedText(request.body, options.maxRequestBytes ?? 1_000_000, controller.signal); }
      catch (error) { if (error instanceof RangeError) return json({ detail: 'Request is too large' }, 413); throw error; }
      try {
        const payload = JSON.parse(body);
        if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error();
      } catch { return json({ detail: 'Invalid JSON request' }, 400); }
    }
    const url = new URL(path, base);
    if (query) url.search = query;
    if (controller.signal.aborted) throw new Error('Request aborted');
    const response = await untilAborted((options.fetchImpl ?? fetch)(url, {
      method: request.method, headers, body, signal: controller.signal, cache: 'no-store', redirect: 'error',
    }), controller.signal);
    if (response.status === 204) return new Response(null, { status: 204, headers: { 'Cache-Control': 'no-store' } });
    if (!response.headers.get('content-type')?.toLowerCase().includes('application/json')) {
      response.body?.cancel().catch(() => {});
      return json({ detail: 'Backend returned an invalid response' }, 502);
    }
    let payload;
    try { payload = JSON.parse(await boundedText(response.body, options.maxResponseBytes ?? 8_000_000, controller.signal)); }
    catch (error) { if (controller.signal.aborted) throw error; return json({ detail: 'Backend returned an invalid response' }, 502); }
    return json(payload, response.status);
  } catch {
    return json({ detail: controller.signal.aborted ? 'Verification cancelled or timed out; no verified answer was received' : 'Verification backend is unavailable' }, controller.signal.aborted ? 504 : 502);
  } finally {
    clearTimeout(timer);
    request.signal.removeEventListener('abort', cancel);
  }
}
