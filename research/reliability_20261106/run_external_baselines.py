"""Bounded live model comparisons on twenty already archived questions.

No web search; the frozen passage pool is shared with the archived comparison.
This run is not contemporaneous with the historical graph and is not a causal
estimate of the new graph's factuality. Failure stops after three provider errors.
"""
from common import *
configure()
import asyncio
import dataclasses
import logging
import time
from baselines import Backend, Budget, Case, run


async def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    os.environ.update({'HALLUCIGUARD_LLM_PROVIDER_ORDER':'groq',
                       'OPENROUTER_MAX_RETRIES':'0','HALLUCIGUARD_LLM_TIMEOUT':'45'})
    logging.disable(logging.CRITICAL)
    from services.base_llm_service import BaseLLMService
    service=BaseLLMService(model_overrides={'groq':'openai/gpt-oss-120b'})
    source=PRIOR/'runtime_records/paired_baselines/results.json'
    rows=load(source)
    out=RUN/'external_baselines'
    out.mkdir(parents=True,exist_ok=True)
    import psutil
    methods=['basic_rag','cove_style_factored','rarr_style_frozen_corpus','selfcheck_style_nli']
    config=provenance(__file__,[source,Path(__file__).with_name('baselines.py')],
        kind='new_live_generation_on_archived_questions',n=len(rows),methods=methods,
        provider='groq',model='openai/gpt-oss-120b',maximum_http_requests=320,
        maximum_completion_tokens_per_request=512,maximum_prompt_chars=24000,
        provider_retry_count=0,provider_failover=False,retrieval_calls=0,
        sample_temperature=1.0,other_temperature=0.0,
        ram_available_bytes=psutil.virtual_memory().available,
        limitations=['Three samples for the SelfCheck style approximation, no tuned release threshold.',
          'RARR style uses lexical top three within a shared frozen corpus, no original trained editor.',
          'CoVe style uses at most three factored checks, generic prompts, no original few shot examples.',
          'Same named hosted model as historical generator; provider revisions and time differ.',
          'All whole answer human labels pending.'])
    save(out/'configuration.json',config)
    total=0;errors=0;engine=None
    async def generate(prompt,temperature,max_tokens):
        nonlocal total,errors
        if total>=320 or errors>=3:raise RuntimeError('GLOBAL_REQUEST_STOP')
        total+=1
        raw=await service.generate(user_query=prompt,conversation_history=[],generation_mode='normal',temperature=temperature,max_tokens=max_tokens)
        value=dataclasses.asdict(raw) if dataclasses.is_dataclass(raw) else raw.model_dump()
        result={k:value.get(k) for k in ['status','provider','model','usage','token_usage','finish_reason','attempt_count','duration_ms','error_code','provider_attempts']}
        result['text']=value.get('draft_response','')
        if value.get('status')!='success':errors+=1
        return result
    async def nli(claim,passages):
        nonlocal engine
        if engine is None:
            import torch
            torch.set_num_threads(4)
            from nli import NLIEngine
            engine=NLIEngine()
        return engine.batch_classify(claim,passages)
    outputs=[]
    for row in rows:
        if row.get('generation_failed'):continue
        case=Case(row['id'],row['query'],row['initial_draft'],row['shared_first_pass_evidence'])
        for method in methods:
            if errors>=3:break
            result=await run(method,case,Backend(generate,nli=nli,budget=Budget(max_calls=8)))
            result['source_group']=row['source_group']
            save(out/f'{case.id}_{method}.json',result)
            outputs.append(result)
            print(case.id,method,result['status'],'calls',total,flush=True)
        if errors>=3:break
    save(out/'run_summary.json',{'completed_records':len(outputs),'planned_records':len(rows)*len(methods),
        'provider_requests':total,'provider_errors':errors,'status':'stopped_provider_failures' if errors>=3 else 'completed',
        'completed_by_method':{m:sum(r['method']==m and r['status']!='failed' for r in outputs) for m in methods}})

if __name__=='__main__':asyncio.run(main())
