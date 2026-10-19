"""Bounded real HTTP acceptance cases. No graph/model/provider mocks."""
import json,os,time,requests
from pathlib import Path
root=Path(os.getenv('LOCAL_HTTP_OUTPUT','runtime-logs/http-final'));root.mkdir(parents=True,exist_ok=True)
cases=[('generated-supported',{'user_query':'What is the capital of France? Answer with one short factual sentence.'}),('unsupported',{'user_query':'What exact revenue did the fictional company QuuxZorblax987 report in 2025?', 'llm_response':'QuuxZorblax987 reported revenue of exactly 483 billion dollars in 2025.'}),('correction',{'user_query':'Who founded Microsoft?','llm_response':'Snehith is the founder of Microsoft.'})]
failures=[]
for name,body in cases:
 start=time.monotonic()
 try:
  response=requests.post(os.getenv('LOCAL_API_URL','http://127.0.0.1:8020')+'/api/v1/verify',json=body,timeout=300)
  data=response.json();(root/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
  assert response.status_code==200,f'HTTP {response.status_code}'
  memory=data.get('memory_result') or {}
  nodes=[e.get('node') for e in data.get('trace',[])]
  if name=='unsupported':
   assert data['terminal_status'] in {'human_review','rejected'}
   assert data['verification_status']!='verified_and_accepted'
   assert memory.get('stored_count')==0
   assert '483 billion' not in data['final_response']
  else:
   assert data['terminal_status']=='accepted' and data['verification_status']=='verified_and_accepted'
   assert data['detector']['model_loaded'] and data['detector']['inference_executed']
   claims=data['verifier']['claim_evidence']
   assert claims and all(c['verdict']=='verified' and c['evidence'] for c in claims)
   assert memory.get('status') in {'stored','duplicate'}
   if name=='generated-supported':assert 'Paris' in data['final_response']
   else:
    assert all(x in data['final_response'] for x in ['Bill Gates','Paul Allen']) and 'Snehith' not in data['final_response']
    assert data['reverification_result']['passed'] is True
    assert data['correction_attempt_count']>=1
    assert {'detector','claim_analyzer','verifier','detector_grounded','judge','corrector','reverifier','memory'}<=set(nodes)
    bus=data['inter_agent_bus'];assert any(m.get('source_agent')=='reverifier' for m in bus)
  print('PASS',name,round(time.monotonic()-start,1),data['terminal_status'],memory.get('status'),flush=True)
 except Exception as exc:
  failures.append({'case':name,'type':type(exc).__name__,'reason':str(exc)[:200]})
  print('FAIL',name,type(exc).__name__,str(exc)[:200],flush=True)
(root/'summary.json').write_text(json.dumps({'passed':len(cases)-len(failures),'failed':failures},indent=2))
raise SystemExit(bool(failures))
