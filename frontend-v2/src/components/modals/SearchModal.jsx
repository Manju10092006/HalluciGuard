"use client";

import { useEffect, useState } from "react";
import { Plus, FileSearch, CornerDownLeft } from "lucide-react";
import { useChat } from "@/context/ChatContext";
import { CONV_STATUS, relativeTime } from "@/lib/format";
import { StatusPill } from "@/components/chat/StatusPill";
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandSeparator,
} from "@/components/ui/command";

/**
 * Cmd+K palette — shadcn Command (cmdk) pattern.
 * Grouped results (actions / recent verifications / evidence sources),
 * keyboard-first, flat Linear/Vercel-style chrome. MIT (shadcn).
 */
export function SearchModal() {
  const { modal, setModal, conversations, newVerification, navigate } = useChat();
  const open = modal?.type === "search";
  const [q, setQ] = useState("");

  useEffect(() => {
    if (open) setQ("");
  }, [open ]);

  const term = q.trim().toLowerCase();
  const results = (
    term
      ? (conversations || []).filter((c) => c.title.toLowerCase().includes(term))
      : conversations || []
  ).slice(0, 8);

  const go = (c) => {
    setModal(null);
    navigate(`/c/${c.id}`);
  };

  // Evidence sources across recent conversations (deduped by title)
  const sourceResults = term
    ? (conversations || [])
        .flatMap((c) => (c.sources || []).map((s) => ({ ...s, convId: c.id })))
        .filter((s) => (s.title || "").toLowerCase().includes(term))
        .filter(
          (s, i, arr) => arr.findIndex((x) => x.title === s.title) === i
        )
        .slice(0, 5)
    : [];

  return (
    <CommandDialog
      open={open}
      onOpenChange={(o) => !o && setModal(null)}
      data-testid="search-modal"
    >
      <CommandInput
        value={q}
        onValueChange={setQ}
        placeholder="Search verifications, actions, sources..."
        aria-label="Search verifications"
        data-testid="search-input"
      />
      <CommandList>
        <CommandEmpty data-testid="search-empty">
          {q ? `No results match “${q}”.` : "Type to search verifications."}
        </CommandEmpty>

        <CommandGroup heading="Actions">
          <CommandItem
            value="new-verification"
            onSelect={() => {
              setModal(null);
              newVerification();
            }}
            data-testid="search-new-verification"
          >
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-[var(--accent-soft)] text-hg-accent">
              <Plus className="h-3.5 w-3.5" />
            </span>
            <span>New verification</span>
            <span className="ml-auto font-mono text-[11px] text-hg-muted">↵</span>
          </CommandItem>
        </CommandGroup>

        {results.length > 0 && (
          <CommandGroup heading={q ? "Verifications" : "Recent"}>
            {results.map((c) => {
              const st = CONV_STATUS[c.status] || CONV_STATUS.reviewing;
              return (
                <CommandItem
                  key={c.id}
                  value={`verification-${c.id}-${c.title}`}
                  onSelect={() => go(c)}
                  data-testid={`search-result-${c.id}`}
                >
                  <FileSearch className="h-4 w-4 shrink-0 text-hg-muted" />
                  <span className="min-w-0 flex-1 truncate">{c.title}</span>
                  <span className="text-[11px] text-hg-muted">
                    {relativeTime(c.updated_at)}
                  </span>
                  <StatusPill verdict={st.verdict} label={st.label} dot={false} />
                  <CornerDownLeft className="h-3.5 w-3.5 text-hg-muted opacity-0 data-[selected=true]:opacity-100" />
                </CommandItem>
              );
            })}
          </CommandGroup>
        )}

        {sourceResults.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Evidence sources">
              {sourceResults.map((s, i) => (
                <CommandItem
                  key={`${s.convId}-${i}`}
                  value={`source-${s.convId}-${s.title}`}
                  onSelect={() => {
                    const conv = (conversations || []).find(
                      (c) => c.id === s.convId
                    );
                    if (conv) go(conv);
                  }}
                >
                  <span className="min-w-0 flex-1 truncate text-hg-text2">
                    {s.title}
                  </span>
                  <span className="text-[11px] text-hg-muted">{s.domain}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
