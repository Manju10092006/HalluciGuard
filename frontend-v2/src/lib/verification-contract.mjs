const verdictOf = value => String(value || "").toLowerCase().replace("verdictlabel.", "");
export function verificationView(result, mode = "standard") {
  if (!result || typeof result !== "object" || !result.terminal_status || !result.verification_status) throw new Error("Malformed verification response");
  const reports = result.verifier?.claim_evidence || [];
  const supported = reports.length > 0 && reports.every(c => ["verified", "supported"].includes(verdictOf(c.verdict)) && c.evidence?.some(e => (e.snippet || e.text || "").trim()));
  const accepted = result.terminal_status === "accepted" && result.verification_status === "verified_and_accepted" && supported;
  const failed = result.verification_status === "agent_failed" || result.terminal_status === "failed";
  const verdict = accepted ? "supported" : failed ? "failed" : "withheld", evidence = [];
  const claims = reports.map((c, index) => {
    const status = ["verified", "supported"].includes(verdictOf(c.verdict)) ? "supported" : verdictOf(c.verdict) === "contradicted" ? "contradicted" : "insufficient";
    const evidence_ids = (c.evidence || []).map((e, n) => {
      const id = `${c.claim_id || index}-e${n}`;
      let domain = ""; try { domain = new URL(e.url).hostname; } catch {}
      evidence.push({ id, source: e.source || e.source_id || "", title: e.title || "", url: e.url || "", domain, excerpt: e.snippet || e.text || "", relationship: status === "supported" ? "supports" : status === "contradicted" ? "contradicts" : "context" });
      return id;
    });
    return { id: c.claim_id || `c${index}`, text: c.claim_text, status, evidence_ids };
  });
  const explanation = accepted ? "The final answer passed evidence verification and the release policy." : failed ? "Verification failed. No answer has been verified." : "The answer was withheld because verification did not establish sufficient support for release.";
  return { verdict, answer: accepted ? result.final_response : explanation, explanation,
    correction: accepted && result.reverification_result?.passed === true ? result.correction_result?.corrected_text || null : null,
    original_claim: accepted && result.reverification_result?.passed === true ? result.draft_response : null,
    claims, evidence, mode,
    pipeline: (result.trace || []).map(t => ({ agent: t.node || t.agent || "Pipeline", detail: t.status || t.event || "executed" })),
    counts: { sources: evidence.length, supporting: evidence.filter(e => e.relationship === "supports").length, contradicting: evidence.filter(e => e.relationship === "contradicts").length }, execution_id: result.execution_id };
}
