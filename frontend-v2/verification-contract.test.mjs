import test from 'node:test';
import assert from 'node:assert/strict';
import { verificationView } from './src/lib/verification-contract.mjs';
const supported = { claim_id:'c1',claim_text:'Paris is the capital of France.',verdict:'verified',evidence:[{source:'fixture',url:'https://example.org/fact',snippet:'Paris is the capital of France.'}] };
const response = {terminal_status:'accepted',verification_status:'verified_and_accepted',final_response:'Paris is the capital of France.',verifier:{claim_evidence:[supported]},trace:[{node:'verifier',status:'completed'}]};
test('accepted supported response preserves real evidence',()=>{const v=verificationView(response);assert.equal(v.verdict,'supported');assert.equal(v.evidence[0].excerpt,supported.evidence[0].snippet);assert.deepEqual(v.pipeline,[{agent:'verifier',detail:'completed'}])});
for (const [name,changes] of Object.entries({withheld:{terminal_status:'human_review',verification_status:'human_review_required'},failed:{terminal_status:'failed',verification_status:'agent_failed'},mixed:{verifier:{claim_evidence:[supported,{...supported,claim_id:'c2',verdict:'unverified'}]}},missingEvidence:{verifier:{claim_evidence:[{...supported,evidence:[]}]}},noClaims:{verifier:{claim_evidence:[]}}})) {
 test(name+' cannot display supported answer',()=>{const v=verificationView({...response,...changes});assert.notEqual(v.verdict,'supported');assert.notEqual(v.answer,response.final_response);assert.equal(v.correction,null)})
}
test('unverified correction never displayed',()=>{const v=verificationView({...response,terminal_status:'human_review',correction_result:{corrected_text:'Unsafe candidate'},reverification_result:{passed:false}});assert.equal(v.correction,null)});
test('malformed response fails visibly',()=>assert.throws(()=>verificationView({}),/Malformed/));
