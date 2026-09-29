import React from 'react'
import { ExternalLink, Globe, BookOpenCheck } from 'lucide-react'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'

/**
 * Evidence slide-over — shadcn Sheet pattern (Radix Dialog primitive,
 * zero new deps). Flat right-side panel: header (source title), body
 * (excerpt + verdict + lineage), footer (actions). MIT (shadcn).
 */
export function SourceDrawer({ source, isOpen, onClose }) {
  return (
    <Sheet
      open={!!isOpen}
      onOpenChange={(open) => {
        if (!open) onClose?.()
      }}
    >
      <SheetContent
        side="right"
        className="flex w-[420px] flex-col gap-0 bg-hg-surface p-0 sm:max-w-[420px]"
        aria-label="Primary source record"
      >
        {source && (
          <>
            <SheetHeader className="flex-row items-center justify-between space-y-0 border-b border-hg-line px-5 py-4 pr-12 text-left">
              <div>
                <SheetTitle className="flex items-center gap-2 text-[14px] font-semibold text-hg-text">
                  <BookOpenCheck size={16} className="text-hg-accent" />
                  Primary Source Record
                </SheetTitle>
                <SheetDescription className="mt-0.5 text-[12px] text-hg-muted">
                  Verified passage and evidence lineage
                </SheetDescription>
              </div>
            </SheetHeader>

            <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-5 py-5">
              {/* Source domain & relationship */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Globe size={14} className="text-hg-muted" />
                  <span className="text-[12.5px] font-semibold text-hg-text2">
                    {source.domain}
                  </span>
                </div>
                <span className={`relationship-badge tone-${source.relationshipTone}`}>
                  {source.relationship}
                </span>
              </div>

              <h2 className="text-[18px] font-semibold leading-[1.35] text-hg-text">
                {source.title}
              </h2>

              <div className="h-px bg-hg-line" />

              {/* Full passage citation */}
              <div>
                <span className="mb-2 block text-[11.5px] font-bold uppercase tracking-[0.04em] text-hg-muted">
                  Verified Passage
                </span>
                <blockquote className="rounded-md border-l-[3px] border-hg-accent bg-hg-sunken px-3.5 py-3 text-[13.5px] italic leading-[1.55] text-hg-text">
                  “{source.excerpt}”
                </blockquote>
              </div>

              {/* Provenance details */}
              <div>
                <span className="mb-2 block text-[11.5px] font-bold uppercase tracking-[0.04em] text-hg-muted">
                  Evidence Lineage
                </span>
                <dl className="flex flex-col gap-2.5 rounded-[10px] border border-hg-line bg-hg-sunken px-3.5 py-3">
                  <div className="flex justify-between text-[12.5px]">
                    <dt className="text-hg-text2">Entailment Status</dt>
                    <dd className="font-semibold text-hg-text">
                      {source.relationshipTone === 'contradicted'
                        ? 'Direct Contradiction'
                        : 'Factual Entailment'}
                    </dd>
                  </div>
                  <div className="flex justify-between text-[12.5px]">
                    <dt className="text-hg-text2">Extraction Method</dt>
                    <dd className="font-semibold text-hg-text">
                      Passage-level BGE Retrieval
                    </dd>
                  </div>
                  <div className="flex justify-between text-[12.5px]">
                    <dt className="text-hg-text2">Corpus Index</dt>
                    <dd className="font-semibold text-hg-text">
                      Authoritative Primary Record
                    </dd>
                  </div>
                </dl>
              </div>
            </div>

            <div className="flex justify-end border-t border-hg-line px-5 py-4">
              {source.url ? (
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-full bg-hg-accent px-4 py-2 text-[13px] font-semibold text-white transition-opacity hover:opacity-90"
                >
                  <span>Visit External Source</span>
                  <ExternalLink size={14} />
                </a>
              ) : (
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-full border border-hg-line px-4 py-2 text-[13px] font-semibold text-hg-text transition-colors hover:bg-hg-sunken"
                >
                  Close
                </button>
              )}
            </div>
          </>
        )}
      </SheetContent>

      <style>{`
        .relationship-badge {
          display: inline-flex;
          align-items: center;
          padding: 3px 8px;
          border-radius: var(--radius-pill);
          font-size: 11px;
          font-weight: 600;
        }
        .tone-supported {
          background: var(--supported-bg);
          color: var(--supported);
          border: 1px solid var(--supported-border);
        }
        .tone-contradicted {
          background: var(--contradicted-bg);
          color: var(--contradicted);
          border: 1px solid var(--contradicted-border);
        }
        .tone-context {
          background: var(--uncertain-bg);
          color: var(--uncertain);
          border: 1px solid var(--uncertain-border);
        }
        @media (prefers-reduced-motion: reduce) {
          [data-state="open"], [data-state="closed"] {
            animation: none !important;
            transition: none !important;
          }
        }
      `}</style>
    </Sheet>
  )
}
