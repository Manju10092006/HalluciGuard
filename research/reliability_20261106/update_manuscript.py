"""Add verified reliability findings to the frozen manuscript, without restart."""
from common import *
import shutil

S=RUN/'manuscript/source'
def add(path,text):
    p=S/path
    original=p.read_text(encoding='utf8')
    if text.strip() in original:raise RuntimeError('Already revised: '+path)
    p.write_text(original.rstrip()+'\n\n'+text.strip()+'\n',encoding='utf8')

def main():
    if (RUN/'manuscript/revision_started.json').exists():raise RuntimeError('Revision already started')
    save(RUN/'manuscript/revision_started.json',{'baseline':'22 page reviewed manuscript','date':'2026-11-07','approved_replacements':[1,3,5,7]})
    gen=Path.home()/'.codex/generated_images/01a15a78-5c5a-71a3-8bef-2ef5b5c305e3'
    images={'01_architecture':'exec-1b5686af-77b7-4837-91c6-361e32490465.png',
      '06_claim_analyzer':'exec-ef30f9ab-34ae-4c00-a798-8f5841d0a925.png',
      '04_relation_arbitration':'exec-dd231e64-c6c1-4ca6-9ef5-3a8897428c61.png',
      '05_controlled_loops':'exec-4c38fda3-0b82-4d42-beee-2edacbd0b0a1.png'}
    for old,new in images.items():
        shutil.copy2(gen/new,S/'figures'/f'{old}_approved_r1.png')
        for p in (S/'sections').glob('*.tex'):
            text=p.read_text(encoding='utf8')
            if f'figures/{old}.png' in text:
                text=text.replace(f'figures/{old}.png',f'figures/{old}_approved_r1.png')
                if old!='01_architecture':
                    import re
                    text=re.sub(r'\\SideFigure(\{[lr]\}\{[0-9.]+\}\{figures/'+old+r'_approved_r1.png\})',r'\\ColumnFigure\1',text)
                p.write_text(text,encoding='utf8')
    add('sections/introduction.tex',r'''
\subsection{Research questions}

The seven questions retain the scope of the original study. RQ1 asks how the complete workflow affects factual release and coverage. RQ2 concerns claim extraction, evidence selection, and relation agreement. RQ3 concerns Detector discrimination, calibration, and operational value. RQ4 concerns correction preservation and renewed assessment. RQ5 concerns Judge release decisions on competing claims. RQ6 concerns execution reliability and computational cost. RQ7 concerns admission and persistence of verified Memory records. Each has a corresponding result below, but whole answer factuality, independent human judgments, and memory assisted generation remain unmeasured. The reliability revision adds frozen state policy comparisons, a diagnostic repair experiment, and a separate ANLI round two transfer sample. It does not replace the archived studies or their unfavorable outcomes.
''')
    p=S/'sections/agent_methods.tex';t=p.read_text(encoding='utf8')
    inserts={
    r'\subsection{Claim Analyzer}':r'''The current adapter now compares each sentence risk with the configured binary threshold when setting \code{requires_verification}. A three class argmax label is still exported separately. These decisions can disagree because the verification risk combines contradiction and insufficient information. The normal graph uses \code{ALWAYS_VERIFY=true} and disables the Detector fast path. The two experimental release policies also force verification. Thus the threshold change corrects the triage contract without claiming a demonstrated reduction in verification calls.

''',
    r'\subsection{Judge}':r'''The reliability branch adds four selectable scoring modes. The historical structured mode remains the default. The no scoped mode removes the structured screening, and the temporal variant retains date screening. The experimental soft mode keeps decisive comparisons but allows inconclusive extraction to defer to NLI. An alternative value for a multi-valued creation, leadership, or location relation does not by itself force contradiction. A narrow declarative location pattern recognizes \emph{Veylan lies in North Merith.} only when a comparable passage supports the same relation. Negation and qualified controls are tested separately. These rules do not establish general entity resolution or temporal reasoning.

''',
    r'\subsection{Corrector}':r'''The current Judge exposes \code{default}, \code{strict_all_verified}, and \code{slot_aware} policies. The default preserves the archived action behavior. The strict policy disables acceptance with unresolved peripheral reports. Slot matching normalizes selected subject aliases and compares relation and temporal scope against verified assertions. It recognizes capital, creator, lead developer, and location patterns. An unresolved alternative for the same answer slot becomes a core gap, whether it is positive, negated, or hedged; matching does not itself prove contradiction. Two verified cofounders remain admissible. Conflicted reports also prevent peripheral tolerance. The parser is deliberately bounded and can miss paraphrases, implicit dates, or aliases absent from its explicit map.

''',
    r'\subsection{ReVerifier}':r'''The reliability revision also checks designated preserved strings that occur literally in the original answer. If an edit removes one, the Corrector returns \code{PRESERVATION_VIOLATION} and does not return the altered text as a usable correction. This catches a supported clause deleted while another clause in the same sentence is repaired. It does not accept paraphrases as equivalent and does not measure semantic preservation or the truth of new text.

''',
    r'\subsection{Memory}':r'''Fresh diagnostics now retain raw pair scores, selected evidence, relation mode, and pipeline stages alongside the canonical report. These records distinguish a failed search from evidence removed during decision selection. The factual pass, report reconciliation, topicality veto, and return to the Judge are unchanged. The proposed repair addresses an upstream evidence decision rather than bypassing renewed assessment.

'''}
    for needle,insert in inserts.items():
        assert needle in t;t=t.replace(needle,insert+needle,1)
    t=t.replace('Stored records retain partial provenance. The execution export retains available evidence excerpts and URLs.','The historical stores retained partial provenance. The current write path additionally preserves supplied evidence and execution metadata.')
    p.write_text(t,encoding='utf8')
    add('sections/agent_methods.tex',r'''
The revised writer carries a versioned provenance object into the graph fact, vector metadata, and SQLite evidence summary. It includes supplied evidence, source identifiers, and request metadata. A regression check reads the same URL, excerpt, and execution identifier from all three stores. Vector persistence errors now surface to the batch result, and an orchestration exception produces a canonical failed Memory result while preserving an already accepted answer. These operations are not an atomic transaction across stores: a late failure can leave partial records. Duplicate handling also does not merge newly supplied provenance into the older fact. These remain implementation limits.
''')
    add('sections/methods.tex',r'''
\subsubsection{Reliability revision protocol}

The new experiment series freezes the reviewed manuscript, original source, data locations, and checksums before editing. The original policy and each experimental mode use separate settings. The Judge comparison reuses all 180 archived states and summarizes the last state for each of 100 requests. The relation development comparison reuses all 100 reference targets with identical stored neural pair scores and passages. These records had already been inspected, so they are development evidence. Paired percentile intervals resample 20 source groups 4,000 times with seed 20261106.

The repair diagnosis runs fresh CPU NLI on 24 candidates from the eight fictional mixed card cases: a valid repair, a residual false assertion, and a newly introduced false assertion for each card. The actual ReVerifier and Judge gates then consume those frozen neural scores under the two relation settings. Source preserving fallback extraction and fixed passages isolate the evidence decision. Reported gate time excludes retrieval, inference, and hosted correction. This is a component replay on known diagnostic examples, not a new complete graph benchmark.

After development, the soft policy is frozen before sampling 120 examples from ANLI round two test, 40 per reference label, with seed 20261106. Each example supplies its original premise at fixed relevance. The four modes share fresh CPU NLI scores. Earlier transfer work used round three. Example identifiers and the source archive hash are saved before inference. Intervals for this transfer sample resample examples and are conditional on its balanced composition; source or pretraining independence is not established.

The common baseline interface provides plain draft, self critique, basic retrieval augmentation, retrieval with claim verification, versioned graph callbacks, and three literature inspired adapters. The SelfCheck style adapter samples three additional answers and reports sentence contradiction scores without inventing a release threshold~\citep{selfcheck}. The factored Chain of Verification adapter plans at most three checks and answers each without the draft or other check answers~\citep{cove}. The RARR style adapter researches up to three questions within the shared frozen passage pool and makes small revisions after unsupported judgments~\citep{rarr}. These are explicitly simplified implementations, with generic prompts, bounded calls, and different models from the original papers. The pilot uses the same named hosted generator as the archived ordinary study, but provider time and model revisions are uncontrolled. It cannot establish a matched prospective comparison with the new graph.
''')
    rel=load(RUN/'relation/summary.json')['policies'];held=load(RUN/'heldout/summary.json')
    repairs=load(RUN/'repairs/summary.json');judge=load(RUN/'judge/summary.json')['policies']
    def table(path,caption,label,columns,head,rows):
        text='\\begin{table}[!htbp]\n\\caption{'+caption+'}\\label{'+label+'}\n\\small\n\\begin{tabularx}{\\columnwidth}{'+columns+'}\n\\toprule\n'+head+' \\\\\n\\midrule\n'
        text+='\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)
        text+='\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n'
        (S/path).write_text(text,encoding='ascii')
    names={'structured':'Structured','no_scoped':'No scoped rules','soft':'Soft','nli_temporal':'NLI with temporal checks'}
    table('tables/reliability_relation.tex','New relation comparisons. Development agreement is out of 100 previously inspected targets. ANLI R2 agreement is out of 120 balanced examples. False support counts use the 80 R2 examples whose reference is not support.','tab:reliability_relation','Yrrr','Policy & Dev. & R2 & False support',[(names[m],rel[m]['agreement'],held[m]['agreement'],held[m]['false_support']) for m in names])
    table('tables/reliability_repair.tex','Actual ReVerifier and Judge gates over frozen pair scores. Valid and invalid candidates are known fictional card controls; timing excludes inference and retrieval.','tab:reliability_repair','Yrr','Measure & Original & Soft',[
      ['Valid repairs accepted / 8',repairs['structured']['valid_accepted'],repairs['soft']['valid_accepted']],
      ['Invalid repairs withheld / 16',repairs['structured']['invalid_withheld'],repairs['soft']['invalid_withheld']],
      ['Invalid repairs accepted / 16',repairs['structured']['false_acceptance'],repairs['soft']['false_acceptance']],
      ['Complete reports / 24',repairs['structured']['complete_report_coverage'],repairs['soft']['complete_report_coverage']],
      ['Mean gate time, ms',f"{1000*repairs['structured']['mean_gate_seconds']:.2f}",f"{1000*repairs['soft']['mean_gate_seconds']:.2f}"]])
    table('tables/reliability_judge.tex','Last state replay for the same 100 requests. Passing the narrow target test does not establish whole answer correctness. Additional retries are not executed.','tab:reliability_judge','Yrrr','Policy & Released & False target & Target pass',[(name,judge[m]['accepted'],judge[m]['known_false_released'],judge[m]['target_pass_released']) for m,name in [('default','Default'),('strict_all_verified','All verified'),('slot_aware','Slot aware')]])
    add('sections/results.tex',r'''
\subsection{Reliability revision: diagnosis and transfer}

\input{tables/reliability_repair}

The fresh Asterwyn diagnosis assigns entailment 0.973643 to the repaired capital but neutral probability 0.981536 to the correct location sentence repeated in the passage. The original structured extractor lacks the \emph{lies in} pattern. The passage reaches retrieval, reranking, and inference, then fails decision selection. Report IDs and text reconcile, coverage is complete, and topicality is not the blocking step. The earlier standalone probe used a different experimental policy; its capital failures cannot be attributed to the original structured mode.

Table~\ref{tab:reliability_repair} shows that the soft setting accepts all eight valid controls while withholding all 16 invalid controls. The original gate accepts none of the eight valid candidates. Neither setting accepts an invalid control. The eight templates were used during diagnosis, and the deterministic fallback avoids hosted extraction variation. This result supports the narrow evidence handling repair on these examples. It leaves general repair accuracy and the full graph effect unresolved.

\input{tables/reliability_relation}

The soft relation policy agrees with 85 of 100 development targets, compared with 68 for the original structured mode. It helps 24 targets and harms seven. The paired difference is 0.17, with a source grouped 95 percent interval of [0.01, 0.33]. This interval describes the inspected development records and is not a confirmatory improvement claim. On the subsequently frozen ANLI R2 sample, all four policies agree with 38 of 120 labels. Soft arbitration has 17 false supports, equal to the original mode, and 23 false contradictions compared with 22. Its paired agreement difference is zero with an example resampled interval of [\ensuremath{-0.025}, 0.025]. The transfer result does not justify enabling the soft policy by default.

\input{tables/reliability_judge}

All three Judge policies are independently selectable. The default reproduces all 180 historical actions. At the last state of each request, both alternatives withhold the same five known false mixed targets and retain the 48 target passing releases (Table~\ref{tab:reliability_judge}). Coverage and known false release rate each decrease by 0.05 per attempted request, with a source grouped interval of [\ensuremath{-0.09}, \ensuremath{-0.02}]. Slot matching does not outperform the strict rule on this archive. Independent labels are still needed for correct whole answers withheld, unresolved details, and actual human review outcomes. Counterfactual retry requests are not completed searches.

The new Detector reaggregation reproduces the historical binary accuracy 0.8655, F1 0.3907, ROC AUC 0.8212, and ten bin ECE 0.1296. It flags 2,627 of 18,777 pairs at the archived threshold. All 18,777 would still require verification under the normal no bypass setting. Removing Detector inputs changes zero actions among the 180 replay states. No latency or model call saving from triage is demonstrated. The audit also reproduces the cross split repetitions and 119 conflicting normalized pairs described above. Archived logits lack row identifiers, so joining their predictions to current rows by position would be unjustified.

The revised software passes 601 selected regression cases, including the new relation policies, Judge alternatives, literal preservation, and canonical Memory failure path. Four additional adapter tests check factored prompt isolation, bounded calls, missing verification reports, and score only SelfCheck output. These are execution checks, separate from factual measurements. Provenance tests recover supplied URLs, excerpts, and an execution identifier from graph, vector, and SQLite records. They do not evaluate memory assisted answer quality.

The approved external baseline pilot makes 42 provider requests before reaching the preset stop after three provider failures. Of 80 planned method and question records, 15 are attempted: four basic RAG outputs, three factored verification outputs, two research and revision outputs, three SelfCheck score records, and three failed records. No complete matched comparison is available. The frozen corpus and generic prompts differ from the published systems; partial completion and unreviewed whole answers prevent a comparative factuality conclusion. All failures and the planned denominators remain in the archive.
''')
    add('sections/discussion.tex',r'''
\subsection{What the reliability revision establishes}

The observed gains are local. A targeted location pattern resolves a known false neutral failure, and two release policies remove known unresolved competitors in a replay. The transfer experiment gives no aggregate agreement improvement and adds one false contradiction. The soft setting therefore remains experimental. The original configuration, archived results, and adverse outcomes remain available. The baseline adapters improve experimental access, but the interrupted pilot does not close the external comparison gap. New hosted whole graph runs, broader natural error cohorts, independent claim importance labels, and factorial comparisons are needed before claiming improved system factuality.

Two blank annotation files contain 309 items spanning archived claim reports, the ANLI R2 sample, fixed card repairs, and pilot answers. Model verdicts are hidden, and no human field is populated. Annotators must inspect omitted assertions as well as the supplied claim list. The agreement tool excludes missing judgments, calculates paired agreement and Cohen's kappa where defined, and exports disagreements for independent adjudication. This package prepares human evaluation; it does not constitute completed annotation.
''')
    p=S/'sections/conclusion.tex';t=p.read_text(encoding='utf8')
    paragraphs=t.split('\n\n')
    paragraphs[-1]=r'''The archived challenge exposes a release exception that admits unresolved competing claims, while fixed card runs show valid repairs being withheld. The reliability revision retains 48 target passing releases and withholds five known false targets under either alternative Judge policy. A soft evidence policy accepts eight known valid repair controls and withholds 16 invalid ones, but it gives no aggregate agreement gain on 120 ANLI R2 examples and adds one false contradiction. It remains optional. Detector replays show no demonstrated operational benefit, and Memory checks establish provenance retention rather than improved answers. Independent whole answer annotation, a completed external baseline study, and prospective matched graph runs are still required to estimate factuality and coverage in ordinary use.'''
    p.write_text('\n\n'.join(paragraphs).rstrip()+'\n',encoding='utf8')
    (S/'sections/abstract_body.tex').write_text(r'''A fluent answer can mix supported and unsupported claims, and a correction changes what must be checked. HalluciGuard uses a compiled LangGraph workflow to extract claims, assess retrieved evidence, authorize targeted edits, reassess revised text, and decide what to release and store. The archived live challenge released 53 of 100 drafts, including five known false targets. Two alternative Judge policies withhold those five in frozen state replay and retain 48 target passing releases. An evidence handling revision accepts eight known valid repair controls while withholding 16 invalid controls; the original gate withholds all eight valid controls. Soft relation arbitration improves development agreement from 68 to 85 of 100 targets, but all four scoring modes agree with 38 of 120 labels on a separately sampled ANLI round two test. Soft arbitration adds one false contradiction and remains optional. The archived Detector's binary F1 is 0.3907 on 18,777 transformed RAGTruth pairs; removing its inputs changes no action in 180 Judge states. New Memory checks establish provenance retention, and an external baseline pilot remains incomplete after provider failures. These findings identify specific control and evidence failures. They do not establish general hallucination reduction or whole answer factuality in ordinary use.
''',encoding='utf8')
    p=S/'supplement.tex';t=p.read_text(encoding='utf8').replace(r'\input{supp_sections/faculty_traceability}',r'\input{supp_sections/faculty_traceability}'+'\n\n'+r'\input{supp_sections/reliability_revision}')
    p.write_text(t,encoding='utf8')
    details=load(RUN/'analysis/verifier_metrics.json')
    content=r'''\section{Reliability revision and additional evidence}

This revision preserves the original and technical studies above. Its evidence directory separates fresh inference, frozen score replay, historical reaggregation, software tests, and the interrupted hosted baseline pilot. The new ANLI sample uses round two; the preceding technical comparison uses round three. They must not be pooled.

'''
    for cohort in ['development','heldout_r2']:
        content+='\\subsection{'+('Development reference targets' if cohort=='development' else 'ANLI R2 balanced transfer')+'}\n'
        content+=r'\begin{table}[H]\small\caption{Per class scores for '+cohort.replace('_',' ')+r'. Precision and recall use the stated reference class. A zero reference count is not evidence of performance on that class.}\begin{tabularx}{\linewidth}{YYrrrr}\toprule Policy & Class & Precision & Recall & F1 & Reference count \\ \midrule'+'\n'
        for mode,v in details[cohort].items():
            for label,metrics in v['per_class'].items():
                if label=='conflicted':continue
                content+=' & '.join([names[mode],label,*[f'{metrics[k]:.4f}' for k in ['precision','recall','f1']],str(metrics['n'])])+r' \\'+'\n'
        content+=r'\bottomrule\end{tabularx}\end{table}'+'\n'
    content+=r'''
\subsection{Scope and reproducibility}

The case level relation inventory retains claims, reference judgments, passages, raw NLI scores, structured triples, admissibility decisions, temporal exclusions, and all policy verdicts. Automatic diagnostic tags identify candidate causes rather than human adjudications. The run configuration hashes implementation files at execution, so later reporting edits do not silently redefine the evaluated source. Reproduction commands and exact input hashes are in the local reproducibility guide and result manifest.

For the new Judge replay, both alternatives change five final acceptances to withholding and request five additional verification actions across the 180 states. They retain the same 39 correction actions and four rejections. Those additional verification actions are counterfactual requests. Neither latency for new searches nor subsequent human review is observed. Whole answer correct and incorrect counts remain unassigned.

The external pilot uses a maximum of 320 requests, a 512 token completion cap, no provider fallback or retries, and a stop after three provider errors. It stops at 42 requests. Successful responses identify Groq and the configured model. Failed response objects inherit legacy default provider metadata; the fixed single provider order allows only Groq attempts. The exported pilot omitted the detailed provider error codes, so the failure causes cannot be recovered from these records. A later export adjustment preserves the attempt trail and error code for future runs. Failed records are not treated as correct abstentions.

The two annotation CSVs intentionally contain empty human labels. Each item includes its question or inference task, answer, supplied atomic claim when available, and evidence. Pilot outputs require independent segmentation. Guidelines cover support, contradiction, insufficient evidence, nonfactual text, claim importance, correction validity, withholding, and new unsupported additions. Disagreements require an adjudicator who has access to both independent rationales and the source snapshots.
'''
    (S/'supp_sections/reliability_revision.tex').write_text(content,encoding='utf8')
    print('Updated manuscript source and four approved images; original assets retained.')

if __name__=='__main__':main()
