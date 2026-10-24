import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { apiBase } from './src/lib/api-base.mjs';

// Execute the actual provider, using existing TypeScript and explicit hook,
// fetch, and storage seams. This tests async auth contracts, not DOM rendering.
const source = fs.readFileSync(new URL('./src/lib/auth/AuthContext.tsx', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  fileName: 'AuthContext.tsx',
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2020 },
}).outputText;
const TOKEN = 'hg.auth.jwt.v1', USER = 'hg.auth.user.v1';
const alice = { id: 'alice', email: 'alice@example.test', name: 'Alice' };
const bob = { id: 'bob', email: 'bob@example.test', name: 'Bob' };

function mount(initial = {}) {
  const storage = new Map(Object.entries(initial));
  const slots = [], effects = [], requests = [];
  let cursor = 0, contextValue;
  const same = (left, right) => left && right && left.length === right.length && left.every((item, index) => Object.is(item, right[index]));
  const react = {
    createContext: () => ({ Provider: Symbol('Context.Provider') }),
    createElement: (type, props) => ({ type, props }),
    useState(initialValue) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initialValue === 'function' ? initialValue() : initialValue;
      return [slots[index], value => { slots[index] = typeof value === 'function' ? value(slots[index]) : value; }];
    },
    useRef(value) { const index = cursor++; return slots[index] ||= { current: value }; },
    useCallback(callback, deps) {
      const index = cursor++;
      if (!same(slots[index]?.deps, deps)) slots[index] = { callback, deps };
      return slots[index].callback;
    },
    useEffect(effect, deps) {
      const index = cursor++;
      if (!same(slots[index]?.deps, deps)) {
        const previous = slots[index];
        slots[index] = { deps };
        effects.push(() => { previous?.cleanup?.(); slots[index].cleanup = effect(); });
      }
    },
  };
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports, process: { env: {} },
    require(name) {
      if (name === '../api-base.mjs') return { apiBase };
      assert.equal(name, 'react');
      return { __esModule: true, default: react, ...react };
    },
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, String(value)),
      removeItem: key => storage.delete(key),
    },
    fetch(url, options) {
      let resolve;
      const promise = new Promise(done => { resolve = done; });
      requests.push({ url, options, resolve });
      return promise;
    },
  }, { filename: 'AuthContext.transpiled.cjs' });
  function view() {
    cursor = 0;
    contextValue = module.exports.AuthProvider({ children: null }).props.value;
    while (effects.length) effects.shift()();
    return contextValue;
  }
  view();
  return {
    storage, requests, view,
    async respond(index, data, ok = true) {
      assert.ok(requests[index], `Request ${index} exists`);
      requests[index].resolve({ ok, json: async () => data });
      await new Promise(resolve => setImmediate(resolve));
      return view();
    },
  };
}

test('session restoration trusts /auth/me user and starts in loading state', async () => {
  const auth = mount({ [TOKEN]: 'stored-token', [USER]: JSON.stringify(alice) });
  assert.equal(auth.view().status, 'loading');
  assert.equal(auth.view().user, null);
  assert.equal(auth.requests[0].url, apiBase() + '/auth/me');
  assert.equal(auth.requests[0].options.headers.Authorization, 'Bearer stored-token');
  const state = await auth.respond(0, { user: bob });
  assert.equal(state.status, 'authenticated');
  assert.equal(state.user.id, 'bob');
  assert.equal(JSON.parse(auth.storage.get(USER)).id, 'bob');
});

test('failed restoration clears both token and cached identity', async () => {
  const auth = mount({ [TOKEN]: 'expired', [USER]: JSON.stringify(alice) });
  const state = await auth.respond(0, { detail: 'Expired token' }, false);
  assert.equal(state.status, 'anonymous');
  assert.equal(state.user, null);
  assert.equal(auth.storage.has(TOKEN), false);
  assert.equal(auth.storage.has(USER), false);
});

test('a restoration response without a user ID cannot authenticate', async () => {
  const auth = mount({ [TOKEN]: 'stored-token' });
  const state = await auth.respond(0, { user: { email: alice.email } });
  assert.equal(state.status, 'anonymous');
  assert.equal(auth.storage.has(TOKEN), false);
});

test('logout prevents a late restoration from resurrecting a session', async () => {
  const auth = mount({ [TOKEN]: 'stored-token', [USER]: JSON.stringify(alice) });
  await auth.view().signOut();
  const state = await auth.respond(0, { user: alice });
  assert.equal(state.status, 'anonymous');
  assert.equal(auth.storage.has(TOKEN), false);
  assert.equal(auth.storage.has(USER), false);
});

for (const restoreOk of [true, false]) {
  test(`late restoration ${restoreOk ? 'success' : 'failure'} cannot replace a newer login`, async () => {
    const auth = mount({ [TOKEN]: 'old-token', [USER]: JSON.stringify(alice) });
    const login = auth.view().signIn(bob.email, 'test-password');
    assert.ok(auth.requests[1].url.endsWith('/auth/login'));
    assert.equal(JSON.parse(auth.requests[1].options.body).email, bob.email);
    await auth.respond(1, { access_token: 'bob-token', user: bob });
    assert.equal((await login).ok, true);
    const state = await auth.respond(0, { user: alice, detail: 'Expired token' }, restoreOk);
    assert.equal(state.user.id, 'bob');
    assert.equal(state.token, 'bob-token');
    assert.equal(auth.storage.get(TOKEN), 'bob-token');
    assert.equal(JSON.parse(auth.storage.get(USER)).id, 'bob');
  });
}

test('failed sign-in during restoration resolves loading and invalidates the old restore', async () => {
  const auth = mount({ [TOKEN]: 'old-token', [USER]: JSON.stringify(alice) });
  const login = auth.view().signIn(bob.email, 'wrong-password');
  let state = await auth.respond(1, { detail: 'Invalid credentials' }, false);
  assert.equal((await login).ok, false);
  assert.equal(state.status, 'anonymous');
  state = await auth.respond(0, { user: alice });
  assert.equal(state.status, 'anonymous');
  assert.equal(auth.storage.has(TOKEN), false);
});

test('logout invalidates an in-flight sign-in before any credentials are written', async () => {
  const auth = mount();
  const login = auth.view().signIn(alice.email, 'test-password');
  await auth.view().signOut();
  const state = await auth.respond(0, { access_token: 'alice-token', user: alice });
  assert.equal((await login).ok, false);
  assert.equal(state.status, 'anonymous');
  assert.equal(auth.storage.has(TOKEN), false);
});

test('latest authentication attempt wins even when the earlier response arrives last', async () => {
  const auth = mount();
  const first = auth.view().signIn(alice.email, 'test-password');
  const second = auth.view().signIn(bob.email, 'test-password');
  await auth.respond(1, { access_token: 'bob-token', user: bob });
  assert.equal((await second).ok, true);
  const state = await auth.respond(0, { access_token: 'alice-token', user: alice });
  assert.equal((await first).ok, false);
  assert.equal(state.user.id, 'bob');
  assert.equal(auth.storage.get(TOKEN), 'bob-token');
});

test('a malformed successful sign-in response cannot store credentials', async () => {
  const auth = mount();
  const login = auth.view().signIn(alice.email, 'test-password');
  const state = await auth.respond(0, { access_token: 'orphan-token', user: { email: alice.email } });
  assert.equal((await login).ok, false);
  assert.equal(state.status, 'anonymous');
  assert.equal(auth.storage.has(TOKEN), false);
});
