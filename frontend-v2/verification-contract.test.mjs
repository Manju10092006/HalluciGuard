import test from 'node:test';
import assert from 'node:assert/strict';
import { verificationView } from './src/lib/verification-contract.mjs';
const supported = { claim_id:'c1',claim_text:'Paris is the capital of France.',verdict:'verified',evidence:[{source:'fixture',url:'https://example.org/fact',snippet:'Paris is the capital of France.',entailment_label:'entailment'}] };
const response = {terminal_status:'accepted',verification_status:'verified_and_accepted',final_response:'Paris is the capital of France.',verifier:{claim_evidence:[supported]},trace:[{node:'verifier',status:'completed'}]};
test('accepted supported response preserves real evidence',()=>{const v=verificationView(response);assert.equal(v.verdict,'supported');assert.equal(v.evidence[0].excerpt,supported.evidence[0].snippet);assert.deepEqual(v.pipeline,[{agent:'verifier',detail:'completed'}])});
for (const [name,changes] of Object.entries({withheld:{terminal_status:'human_review',verification_status:'human_review_required'},failed:{terminal_status:'failed',verification_status:'agent_failed'},mixed:{verifier:{claim_evidence:[supported,{...supported,claim_id:'c2',verdict:'unverified'}]}},missingEvidence:{verifier:{claim_evidence:[{...supported,evidence:[]}]}},noClaims:{verifier:{claim_evidence:[]}}})) {
 test(name+' cannot display supported answer',()=>{const v=verificationView({...response,...changes});assert.notEqual(v.verdict,'supported');assert.notEqual(v.answer,response.final_response);assert.equal(v.correction,null)})
}
test('unverified correction never displayed',()=>{const v=verificationView({...response,terminal_status:'human_review',correction_result:{corrected_text:'Unsafe candidate'},reverification_result:{passed:false}});assert.equal(v.correction,null)});
test('malformed response fails visibly',()=>assert.throws(()=>verificationView({}),/Malformed/));

for (const label of ['neutral', 'contradiction', '']) {
 test(`bare verified verdict with ${label || 'missing'} direction is withheld`, () => {
  const v = verificationView({...response, verifier:{claim_evidence:[{...supported, evidence:[{...supported.evidence[0], entailment_label:label}]}]}});
  assert.equal(v.verdict, 'withheld');
  assert.equal(v.claims[0].status, 'insufficient');
 });
}
test('attribution and a final answer are required', () => {
 for (const change of [{final_response:''}, {verifier:{claim_evidence:[{...supported,evidence:[{snippet:'Unattributed',entailment_label:'entailment'}]}]}}]) {
  assert.equal(verificationView({...response,...change}).verdict,'withheld');
 }
});
test('failed correction cannot inherit an accepted status', () => {
 const v=verificationView({...response,correction_result:{corrected_text:'New answer'},reverification_result:{passed:false,status:'completed'}});
 assert.equal(v.verdict,'withheld'); assert.equal(v.correction,null);
});
test('completed independent correction retains its before and after values', () => {
 const v=verificationView({...response,draft_response:'False original',correction_result:{corrected_text:response.final_response},reverification_result:{passed:true,status:'completed'}});
 assert.equal(v.verdict,'supported');
 assert.equal(v.original_claim,'False original');
 assert.equal(v.correction,response.final_response);
});
test('evidence direction is preserved independently of the claim verdict', () => {
 const v=verificationView({...response,verifier:{claim_evidence:[{...supported,evidence:[...supported.evidence,{snippet:'Related context',entailment_label:'neutral'},{snippet:'Refutation',entailment_label:'contradiction'}]}]}});
 assert.deepEqual(v.evidence.map(e=>e.relationship),['supports','context','contradicts']);
});
