import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { proxyBackend } from './src/lib/backend-proxy.mjs';
import { apiBase } from './src/lib/api-base.mjs';

const origin = 'https://backend.example.test';
function request(path = 'api/v1/verify', method = 'POST', options = {}) {
  return new Request(`https://frontend.example.test/api/backend/${path}`, {
    method, headers: { 'Content-Type': 'application/json', Authorization: 'Bearer caller-token', Cookie: 'session=private', 'X-Api-Key': 'must-not-forward', 'X-Request-ID': 'request-123', ...options.headers },
    ...(method === 'GET' ? {} : { body: '{"user_query":"test"}' }), ...options,
  });
}

test('browser defaults to same-origin and ignores the unknown legacy public setting', () => {
  const previousLegacy = process.env.NEXT_PUBLIC_API_BASE_URL;
  const previousOverride = process.env.NEXT_PUBLIC_BACKEND_URL;
  try {
    delete process.env.NEXT_PUBLIC_BACKEND_URL;
    process.env.NEXT_PUBLIC_API_BASE_URL = 'https://unreviewed.example.test';
    assert.equal(apiBase(), '/api/backend');
    assert.equal(apiBase(''), '/api/backend');
    assert.equal(apiBase(' https://backend.example.test/ '), origin);
    assert.equal(apiBase('http://127.0.0.1:8020'), 'http://127.0.0.1:8020');
  } finally {
    if (previousLegacy === undefined) delete process.env.NEXT_PUBLIC_API_BASE_URL;
    else process.env.NEXT_PUBLIC_API_BASE_URL = previousLegacy;
    if (previousOverride === undefined) delete process.env.NEXT_PUBLIC_BACKEND_URL;
    else process.env.NEXT_PUBLIC_BACKEND_URL = previousOverride;
  }
});

test('public overrides cannot contain credentials, query, fragment or backend paths', () => {
  for (const value of ['https://user:password@example.test', 'https://example.test/?token=x', 'https://example.test/#secret', 'https://example.test/api', 'javascript:alert(1)', 'not-a-url']) {
    assert.throws(() => apiBase(value), /Invalid public backend origin/);
  }
});

test('server origin configuration fails closed without sending any network request', async () => {
  for (const value of ['', 'http://unsafe.example.test', 'https://user:pass@example.test', 'https://example.test/base', 'https://example.test/?key=x', 'https://example.test/#fragment']) {
    const response = await proxyBackend(request(), ['api', 'v1', 'verify'], { origin: value, fetchImpl: () => assert.fail('Invalid origin reached fetch') });
    assert.equal(response.status, 503);
    assert.doesNotMatch(await response.text(), /pass|key=x/);
  }
});

test('only configured server BACKEND_URL selects the upstream', async () => {
  const previous = process.env.BACKEND_URL;
  try {
    process.env.BACKEND_URL = origin;
    const response = await proxyBackend(request(), ['api', 'v1', 'verify'], { fetchImpl: async url => {
      assert.equal(String(url), `${origin}/api/v1/verify`);
      return Response.json({ terminal_status: 'human_review' });
    } });
    assert.equal(response.status, 200);
  } finally {
    if (previous === undefined) delete process.env.BACKEND_URL;
    else process.env.BACKEND_URL = previous;
  }
});

test('unknown paths, traversal, embedded separators and unsupported methods never reach fetch', async () => {
  for (const [segments, method, expected] of [[['admin'], 'POST', 404], [['..', 'verify'], 'POST', 404], [['api/v1/verify'], 'POST', 404], [['%2e%2e', 'verify'], 'POST', 404], [['auth', 'me'], 'POST', 405], [[], 'GET', 404]]) {
    const response = await proxyBackend(request('verify', method), segments, { origin, fetchImpl: () => assert.fail('Unknown route reached fetch') });
    assert.equal(response.status, expected);
  }
});

test('caller authorization and JSON are forwarded; cookies, secret headers and upstream cookies are excluded', async () => {
  const response = await proxyBackend(request(), ['api', 'v1', 'verify'], { origin, fetchImpl: async (url, options) => {
    assert.equal(String(url), `${origin}/api/v1/verify`);
    assert.equal(options.headers.get('authorization'), 'Bearer caller-token');
    assert.equal(options.headers.get('x-request-id'), 'request-123');
    assert.equal(options.headers.has('cookie'), false);
    assert.equal(options.headers.has('x-api-key'), false);
    assert.equal(options.redirect, 'error');
    assert.equal(options.cache, 'no-store');
    assert.equal(JSON.parse(options.body).user_query, 'test');
    return Response.json({ detail: 'Authentication required' }, { status: 401, headers: { 'Set-Cookie': 'must-not-return=1', 'X-Api-Key': 'private' } });
  } });
  assert.equal(response.status, 401);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.equal(response.headers.has('set-cookie'), false);
  assert.equal(response.headers.has('x-api-key'), false);
});

test('request queries are rejected except the explicit deep health probe', async () => {
  for (const path of ['api/v1/verify?redirect=https://evil.test', 'api/v1/verify?token=private']) {
    assert.equal((await proxyBackend(request(path), ['api', 'v1', 'verify'], { origin })).status, 400);
  }
  const response = await proxyBackend(request('health?deep=true', 'GET'), ['health'], { origin, fetchImpl: async url => {
    assert.equal(String(url), `${origin}/health?deep=true`);
    return Response.json({ status: 'healthy' });
  } });
  assert.equal(response.status, 200);
});

test('malformed, non-JSON and oversized requests are rejected before contacting the backend', async () => {
  for (const [options, expected] of [[{ body: '{bad-json' }, 400], [{ body: '[]' }, 400], [{ headers: { 'Content-Type': 'text/plain' }, body: '{}' }, 415], [{ body: '{"long":"' + 'x'.repeat(300) + '"}' }, 413]]) {
    const response = await proxyBackend(request('verify', 'POST', options), ['verify'], { origin, maxRequestBytes: 128, fetchImpl: () => assert.fail('Invalid body reached fetch') });
    assert.equal(response.status, expected);
  }
});

test('invalid and oversized upstream responses fail without inventing an answer', async () => {
  for (const makeResponse of [() => new Response('<html>upstream failure</html>'), () => new Response('{broken', { headers: { 'Content-Type': 'application/json' } }), () => Response.json({ long: 'x'.repeat(300) })]) {
    const response = await proxyBackend(request(), ['api', 'v1', 'verify'], { origin, maxResponseBytes: 128, fetchImpl: async () => makeResponse() });
    assert.equal(response.status, 502);
    assert.doesNotMatch(await response.text(), /"verdict"|"answer"|"final_response"/);
  }
});

test('backend errors do not leak provider details', async () => {
  const response = await proxyBackend(request(), ['api', 'v1', 'verify'], { origin, fetchImpl: async () => { throw new Error('private-api-key=secret'); } });
  assert.equal(response.status, 502);
  assert.doesNotMatch(await response.text(), /private-api-key|secret/);
});

test('timeout bounds fetch even if an upstream client does not implement cancellation', async () => {
  const response = await proxyBackend(request(), ['api', 'v1', 'verify'], { origin, timeoutMs: 5, fetchImpl: () => new Promise(() => {}) });
  assert.equal(response.status, 504);
});

test('request cancellation reaches the upstream signal and never returns a verdict', async () => {
  const controller = new AbortController();
  const pending = proxyBackend(request('verify', 'POST', { signal: controller.signal }), ['verify'], { origin, fetchImpl: async (_, { signal }) => {
    controller.abort();
    assert.equal(signal.aborted, true);
    throw new Error('cancelled');
  } });
  const response = await pending;
  assert.equal(response.status, 504);
  assert.doesNotMatch(await response.text(), /"verdict"/);
});

test('an already cancelled request does not contact the backend', async () => {
  const controller = new AbortController();
  controller.abort();
  const response = await proxyBackend(request('health', 'GET', { signal: controller.signal }), ['health'], { origin, fetchImpl: () => assert.fail('Cancelled request reached fetch') });
  assert.equal(response.status, 504);
});

test('slow upstream response body is bounded by the same timeout', async () => {
  const response = await proxyBackend(request(), ['api', 'v1', 'verify'], { origin, timeoutMs: 5, fetchImpl: async () => new Response(new ReadableStream({ start() {} }), { headers: { 'Content-Type': 'application/json' } }) });
  assert.equal(response.status, 504);
});

test('Next route adapter binds dynamic Node handling to the actual proxy helper', async () => {
  const moduleSource = fs.readFileSync(new URL('./src/app/api/backend/[...path]/route.js', import.meta.url), 'utf8');
  assert.match(moduleSource, /export const dynamic = 'force-dynamic'/);
  const route = await import('./src/app/api/backend/[...path]/route.js');
  assert.equal(route.runtime, 'nodejs');
  const response = await route.POST(request(), { params: Promise.resolve({ path: ['unknown'] }) });
  assert.equal(response.status, 404);
});
