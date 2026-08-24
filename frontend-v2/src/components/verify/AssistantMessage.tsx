"use client";

import * as React from "react";
import { useState } from "react";
import { ShieldCheck, ChevronDown, ChevronUp, ChevronRight, Activity, ExternalLink } from "lucide-react";
import { LogoMark } from "@/components/ui/Logo";
import type { VerifyPhase } from "@/lib/hooks/useVerification";
import type { VerificationResult, EvidenceVM } from "@/lib/api/types";
import { cn } from "@/lib/utils/cn";

function getVerdictColor(verdict?: string | null) {
  if (!verdict) return "text-text-muted bg-surface-tertiary";
  switch (verdict.toLowerCase()) {
    case "verified":
      return "text-green-500 bg-green-500/10";
    case "contradicted":
      return "text-red-500 bg-red-500/10";
    case "conflicted":
      return "text-yellow-500 bg-yellow-500/10";
    default:
      return "text-text-muted bg-surface-tertiary";
  }
}

export function AssistantMessage({
  phase,
  elapsedMs,
  result,
}: {
  phase: VerifyPhase;
  elapsedMs: number;
  result: VerificationResult | null;
}) {
  const [detailsOpen, setDetailsOpen] = useState(false);

  const isComplete = phase === "complete" && result;
  const isRunning = phase === "running";

  const answerText = result?.answer?.final ?? result?.answer?.draft;

  return (
    <div className="flex w-full gap-4">
      {/* Avatar */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-surface-tertiary">
        <LogoMark className="h-5 w-5 text-signal" />
      </div>

      {/* Content */}
      <div className="flex min-w-0 flex-1 flex-col gap-4">
        <span className="text-[15px] font-semibold text-text-primary">HalluciGuard</span>

        {isRunning && (
          <div className="flex items-center gap-2 text-[15px] text-text-muted">
            <Activity className="h-4 w-4 animate-pulse text-text-muted" />
            Verifying this claim against available evidence...
          </div>
        )}

        {isComplete && (
          <div className="flex flex-col gap-4">
            {/* The model answer */}
            <div className="prose prose-invert max-w-none text-[15px] leading-relaxed text-text-primary">
              {answerText ? (
                <p className="whitespace-pre-wrap">{answerText}</p>
              ) : (
                <p className="text-text-muted italic">No answer provided.</p>
              )}
            </div>

            {/* Verdict and Evidence details if it was verified */}
            {!result.verifierSkipped && result.claims && result.claims.length > 0 && (
              <div className="mt-2 flex flex-col gap-4 rounded-xl border border-border bg-surface-secondary p-4">
                
                {/* Main claim verdict */}
                {result.claims.map((claim) => (
                  <div key={claim.id} className="flex flex-col gap-3">
                    <div className="flex items-start justify-between gap-4">
                      <div className="text-[14px] font-medium text-text-primary">
                        "{claim.text}"
                      </div>
                      <div className={cn("shrink-0 rounded-md px-2 py-1 text-[12px] font-semibold uppercase tracking-wider", getVerdictColor(claim.verdict))}>
                        {claim.verdict || "UNVERIFIED"}
                      </div>
                    </div>
                    
                    {claim.explanation && (
                      <div className="text-[14px] text-text-secondary">
                        {claim.explanation}
                      </div>
                    )}

                    {/* Evidence List */}
                    {claim.evidence && claim.evidence.length > 0 && (
                      <div className="mt-2 flex flex-col gap-3 border-t border-border pt-4">
                        <div className="text-[13px] font-semibold uppercase tracking-wider text-text-muted">
                          Retrieved Evidence
                        </div>
                        <div className="flex flex-col gap-3">
                          {claim.evidence.map((ev: EvidenceVM) => (
                            <div key={ev.id} className="flex flex-col gap-2 rounded-lg border border-border bg-background p-3">
                              <div className="flex items-start justify-between gap-2">
                                <a 
                                  href={ev.url ?? "#"} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="font-medium text-signal hover:underline flex items-center gap-1 text-[14px]"
                                >
                                  {ev.title || ev.source || "Source"}
                                  {ev.url && <ExternalLink className="h-3 w-3" />}
                                </a>
                                {ev.relation && (
                                  <span className={cn(
                                    "rounded px-1.5 py-0.5 text-[11px] font-medium uppercase",
                                    ev.relation === "supporting" ? "bg-green-500/10 text-green-500" :
                                    ev.relation === "contradicting" ? "bg-red-500/10 text-red-500" :
                                    "bg-surface-tertiary text-text-muted"
                                  )}>
                                    {ev.relation}
                                  </span>
                                )}
                              </div>
                              {ev.snippet && (
                                <p className="text-[13px] text-text-secondary line-clamp-3">
                                  "...{ev.snippet}..."
                                </p>
                              )}
                              
                              {/* Technical Details for this evidence */}
                              <details className="group mt-1">
                                <summary className="flex cursor-pointer list-none items-center gap-1 text-[12px] font-medium text-text-muted hover:text-text-primary">
                                  <ChevronRight className="h-3 w-3 transition-transform group-open:rotate-90" />
                                  Technical Details
                                </summary>
                                <div className="mt-2 grid grid-cols-2 gap-2 rounded bg-surface-tertiary p-2 text-[12px] text-text-secondary">
                                  <div className="flex flex-col">
                                    <span className="text-text-muted">Relevance (BGE)</span>
                                    <span className="font-mono">{ev.bgeScore != null ? (ev.bgeScore * 100).toFixed(1) + "%" : "—"}</span>
                                  </div>
                                  <div className="flex flex-col">
                                    <span className="text-text-muted">Credibility</span>
                                    <span className="font-mono">{ev.credibility != null ? (ev.credibility * 100).toFixed(1) + "%" : "—"}</span>
                                  </div>
                                  <div className="col-span-2 flex flex-col border-t border-border/50 pt-2">
                                    <span className="text-text-muted mb-1">NLI Classification</span>
                                    <div className="flex justify-between font-mono text-[11px]">
                                      <span className="text-green-500">E: {ev.nli?.entailment != null ? (ev.nli.entailment * 100).toFixed(1) + "%" : "—"}</span>
                                      <span className="text-red-500">C: {ev.nli?.contradiction != null ? (ev.nli.contradiction * 100).toFixed(1) + "%" : "—"}</span>
                                      <span className="text-text-muted">N: {ev.nli?.neutral != null ? (ev.nli.neutral * 100).toFixed(1) + "%" : "—"}</span>
                                    </div>
                                  </div>
                                </div>
                              </details>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            
            {/* If fast-path accepted */}
            {result.verifierSkipped && (
               <div className="mt-2 rounded-xl border border-border bg-surface-secondary p-4 text-[14px] text-text-secondary">
                 <div className="flex items-center gap-2 mb-2 text-text-primary font-medium">
                   <ShieldCheck className="h-4 w-4 text-green-500" />
                   Fast-Path Accepted
                 </div>
                 The detector classified this answer as low-risk. The verifier was not invoked.
               </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
