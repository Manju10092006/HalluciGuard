import test from 'node:test';
import assert from 'node:assert/strict';
import axios from 'axios';

const storage = new Map();
globalThis.window = {};
globalThis.localStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: key => storage.delete(key),
};
const response = {
  terminal_status:'accepted', verification_status:'verified_and_accepted', final_response:'Paris is the capital of France.',
  verifier:{claim_evidence:[{claim_id:'c1',claim_text:'Paris is the capital of France.',verdict:'verified',
    evidence:[{source:'fixture',snippet:'Paris is the capital of France.',entailment_label:'entailment'}]}]},
};
let sequence = 0;
async function pendingRequest() {
  storage.clear();
  storage.set('hg.auth.user.v1', JSON.stringify({id:'alice'}));
  storage.set('hg.auth.jwt.v1', 'alice-test-token');
  let finish, began;
  const started = new Promise(resolve => { began=resolve; });
  axios.defaults.adapter = config => new Promise(resolve => {
    finish = () => resolve({data:response,status:200,statusText:'OK',headers:{},config});
    began();
  });
  const {api} = await import(`./src/lib/api.js?session-test=${++sequence}`);
  const conversation=await api.createConversation();
  const request=api.sendMessage(conversation.id,'Capital of France?');
  await started;
  return {api,conversation,request,finish};
}

test('late HTTP response cannot cross a user switch', async () => {
  const pending=await pendingRequest();
  storage.set('hg.auth.user.v1', JSON.stringify({id:'bob'}));
  storage.set('hg.auth.jwt.v1','bob-test-token');
  assert.deepEqual(await pending.api.listConversations(),[]);
  pending.finish();
  await assert.rejects(pending.request,/session changed/);
  assert.deepEqual(await pending.api.listConversations(),[]);
  const alice=JSON.parse(storage.get('hg_local_conversations_v2alice'));
  assert.deepEqual(alice[0].messages,[]);
});

test('late HTTP response cannot survive logout or token replacement', async () => {
  const pending=await pendingRequest();
  storage.delete('hg.auth.jwt.v1');
  pending.finish();
  await assert.rejects(pending.request,/session changed/);
  assert.deepEqual((await pending.api.getConversation(pending.conversation.id)).messages,[]);
});

test('deleted conversation cannot be resurrected by a late response', async () => {
  const pending=await pendingRequest();
  await pending.api.deleteConversation(pending.conversation.id);
  pending.finish();
  await assert.rejects(pending.request,/Conversation not found/);
  assert.deepEqual(await pending.api.listConversations(),[]);
});

test('unchanged session stores the actual response', async () => {
  const pending=await pendingRequest();
  pending.finish();
  const result=await pending.request;
  assert.equal(result.assistant_message.verification.verdict,'supported');
  assert.equal((await pending.api.getConversation(pending.conversation.id)).messages.length,2);
});
