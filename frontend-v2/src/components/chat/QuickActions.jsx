"use client";

import { ShieldCheck, ScanSearch, Route, AlertTriangle, GitCompare, ArrowUpRight } from "lucide-react";

const ACTIONS = [
  {
    category: "Fact check",
    color: "text-hg-accent",
    label: "Verify a factual claim",
    desc: "Test a single statement against primary sources you can inspect.",
    icon: ShieldCheck,
    prompt: "The Great Wall of China is visible from space with the naked eye.",
  },
  {
    category: "AI answer",
    color: "text-emerald-400",
    label: "Check this AI answer",
    desc: "Break a generated response into claims and grade each one.",
    icon: ScanSearch,
    prompt: "Java was created by Snehith.",
  },
  {
    category: "Provenance",
    color: "text-sky-400",
    label: "Trace the sources",
    desc: "Follow every claim back to the evidence that supports it.",
    icon: Route,
    prompt: "Apollo 11 landed on the Moon in July 1969 and Neil Armstrong was the first to walk on it.",
  },
  {
    category: "Risk",
    color: "text-amber-400",
    label: "Find unsupported claims",
    desc: "Surface confident statements that no source actually backs.",
    icon: AlertTriangle,
    prompt: "Our model achieves 99.2% accuracy on the MMLU benchmark, outperforming all prior systems.",
  },
  {
    category: "Compare",
    color: "text-fuchsia-400",
    label: "Compare with evidence",
    desc: "See where an answer agrees or conflicts with the record.",
    icon: GitCompare,
    prompt: "Python was first released in 1991 by Guido van Rossum.",
  },
];

export function QuickActions({ onPick }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3" data-testid="quick-actions">
      {ACTIONS.map((a) => {
        const Icon = a.icon;
        return (
          <button
            key={a.label}
            type="button"
            onClick={() => onPick(a.prompt)}
            data-testid={`quick-action-${a.label.toLowerCase().replace(/\s+/g, "-")}`}
            className="group flex flex-col rounded-2xl border border-hg-line bg-hg-surface p-4 text-left shadow-hg transition-[transform,border-color,box-shadow] duration-200 hover:-translate-y-0.5 hover:border-[rgba(var(--accent-rgb),0.45)] hover:shadow-hg-lg"
          >
            <div className="flex items-center justify-between">
              <span className={`inline-flex items-center gap-1.5 font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] ${a.color}`}>
                <Icon className="h-3.5 w-3.5" strokeWidth={1.75} /> {a.category}
              </span>
              <ArrowUpRight className="h-4 w-4 text-hg-muted opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
            </div>
            <span className="mt-2.5 text-[14px] font-semibold leading-snug text-hg-text">{a.label}</span>
            <span className="mt-1 text-[12.5px] leading-relaxed text-hg-text2">{a.desc}</span>
          </button>
        );
      })}
    </div>
  );
}
