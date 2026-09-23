"use client";

import { ArrowUpRight, ArrowRight, Sparkles, Crown } from "lucide-react";
import { useChat } from "@/context/ChatContext";
import { CONV_STATUS, USER, greeting, relativeTime } from "@/lib/format";
import { VerificationComposer } from "./VerificationComposer";
import { QuickActions } from "./QuickActions";
import { StatusPill } from "./StatusPill";

export function EmptyVerificationState({ onSend, busy, draft }) {
  const { conversations, settings, navigate, setModal } = useChat();
  const recent = (conversations || []).slice(0, 3);
  const first = (settings.nickname || USER.name).split(" ")[0];

  return (
    <div className="flex flex-1 flex-col items-center overflow-y-auto px-4 pb-10 pt-[7vh] md:pt-[10vh]" data-testid="empty-state">
      <div className="hg-stagger flex w-full max-w-[820px] flex-col items-center">
        {/* Upgrade pill */}
        <button
          type="button"
          onClick={() => setModal({ type: "settings", tab: "general" })}
          className="mb-8 inline-flex items-center gap-1.5 rounded-full border border-[rgba(230,170,60,0.4)] bg-[rgba(230,170,60,0.12)] px-3.5 py-1.5 text-[12px] font-semibold text-amber-500 shadow-hg transition-colors hover:bg-[rgba(230,170,60,0.2)] dark:text-amber-300"
          data-testid="upgrade-pill"
        >
          <Crown className="h-3.5 w-3.5" strokeWidth={2} /> Upgrade Plan
        </button>

        {/* Greeting */}
        <h1
          className="text-center text-[34px] font-semibold leading-[1.08] tracking-[-0.03em] text-hg-text sm:text-[46px] md:text-[56px]"
          data-testid="empty-headline"
        >
          {greeting()}, {first}
        </h1>
        <p className="mt-4 max-w-lg text-center text-[15px] leading-relaxed text-hg-text2">
          Welcome to HalluciGuard — verify AI-generated claims against evidence you can inspect.
        </p>

        {/* Composer */}
        <div className="mt-10 flex w-full justify-center">
          <VerificationComposer onSend={onSend} busy={busy} variant="hero" initialText={draft} />
        </div>

        {/* Template cards */}
        <div className="mt-6 w-full">
          <QuickActions onPick={(p) => onSend(p, { mode: "standard", fromQuick: true })} />
        </div>

        {/* Workflow templates hint */}
        <div className="mt-5 flex items-center justify-center">
          <button
            type="button"
            onClick={() => setModal({ type: "create-flow" })}
            className="group inline-flex items-center gap-2 rounded-full border border-dashed border-hg-line bg-hg-sunken/40 px-3.5 py-1.5 text-[12px] font-medium text-hg-text2 transition hover:border-[rgba(var(--accent-rgb),0.6)] hover:text-hg-accent"
          >
            <Sparkles className="h-3.5 w-3.5 text-hg-accent transition-transform group-hover:scale-110" />
            <span>Need automated verification? Explore workflow templates</span>
            <ArrowRight className="h-3 w-3 text-hg-muted transition-transform group-hover:translate-x-0.5 group-hover:text-hg-accent" />
          </button>
        </div>

        {recent.length > 0 && (
          <section className="mt-12 w-full max-w-[680px]" aria-label="Recent verifications" data-testid="recent-strip">
            <div className="mb-2 flex items-center justify-between px-1">
              <h2 className="font-mono text-[11px] uppercase tracking-[0.12em] text-hg-muted">Recent verifications</h2>
              <button type="button" onClick={() => navigate("/history")} className="inline-flex items-center gap-1 text-[12px] text-hg-text2 transition-colors hover:text-hg-accent" data-testid="recent-view-all">View all <ArrowUpRight className="h-3 w-3" /></button>
            </div>
            <ul className="divide-y divide-[var(--border)]">
              {recent.map((c) => {
                const st = CONV_STATUS[c.status] || CONV_STATUS.reviewing;
                return (
                  <li key={c.id}>
                    <button type="button" onClick={() => navigate(`/c/${c.id}`)} data-testid={`recent-item-${c.id}`}
                      className="group flex w-full items-center gap-3 rounded-lg px-2 py-2.5 text-left transition-colors hover:bg-hg-sunken">
                      <span className="min-w-0 flex-1 truncate text-[14px] text-hg-text">{c.title}</span>
                      <span className="hidden text-[11.5px] text-hg-muted sm:block">{relativeTime(c.updated_at)}</span>
                      <StatusPill verdict={st.verdict} label={c.status === "verified" ? "Supported" : c.status === "needs_correction" ? "Contradicted" : "Insufficient evidence"} />
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
