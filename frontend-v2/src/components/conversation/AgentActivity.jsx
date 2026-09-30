import React, { useState } from 'react'
import { ChevronDown, ChevronRight, Cpu } from 'lucide-react'
import { AGENT_PIPELINE } from '../../types/verification'
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from '@/components/ui/accordion'

/**
 * Verification pipeline disclosure — shadcn Accordion (Radix, WAI-ARIA,
 * zero new deps). Outer toggle opens the drawer; each agent stage is an
 * expandable row: trigger = status dot + mono index + name, content = role.
 */
export function AgentActivity() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="agent-activity-disclosure">
      <button
        type="button"
        className="disclosure-toggle"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <div className="toggle-left">
          <Cpu size={12} className="toggle-icon" />
          <span>Verification pipeline details</span>
        </div>
        {isOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>

      {isOpen && (
        <div className="pipeline-drawer">
          <Accordion type="multiple" className="w-full">
            {AGENT_PIPELINE.map((agent, index) => (
              <AccordionItem
                key={agent.id}
                value={agent.id}
                className="border-b border-hg-line last:border-b-0"
              >
                <AccordionTrigger className="py-2.5 hover:no-underline">
                  <span className="flex items-center gap-3">
                    <span className="step-index">0{index + 1}</span>
                    <span className="agent-title">{agent.name}</span>
                    <span className="stage-status-dot" aria-hidden="true" />
                  </span>
                </AccordionTrigger>
                <AccordionContent>
                  <p className="agent-role-desc">{agent.role}</p>
                  <p className="stage-timing">
                    Stage {index + 1} of {AGENT_PIPELINE.length} · agent trace
                    available after run
                  </p>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      )}

      <style>{`
        .agent-activity-disclosure {
          margin-top: 14px;
          padding-top: 8px;
        }

        .disclosure-toggle {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 4px 8px;
          border-radius: var(--radius-sm);
          font-size: 11.5px;
          color: var(--text-muted);
          transition: all 120ms ease;
        }

        .disclosure-toggle:hover {
          color: var(--text-secondary);
          background: var(--surface-sunken);
        }

        .toggle-left {
          display: inline-flex;
          align-items: center;
          gap: 5px;
        }

        .toggle-icon {
          opacity: 0.7;
        }

        .pipeline-drawer {
          margin-top: 10px;
          padding: 6px 16px 10px;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          border-radius: var(--radius-inline);
        }

        .step-index {
          font-family: var(--font-mono);
          font-size: 10px;
          font-weight: 600;
          color: var(--accent);
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 1px 3px;
        }

        .stage-status-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--border-strong);
        }

        .agent-title {
          font-size: 12.5px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .agent-role-desc {
          font-size: 12px;
          color: var(--text-secondary);
          line-height: 1.5;
          margin: 0 0 6px;
        }

        .stage-timing {
          font-family: var(--font-mono);
          font-size: 10.5px;
          color: var(--text-muted);
          margin: 0;
        }

        @media (prefers-reduced-motion: reduce) {
          .disclosure-toggle {
            transition: none;
          }
        }
      `}</style>
    </div>
  )
}
