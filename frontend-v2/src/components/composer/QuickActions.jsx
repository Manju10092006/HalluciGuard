import React from 'react'
import { ShieldCheck, Search, Link2, AlertTriangle, Scale } from 'lucide-react'

const QUICK_PROMPTS = [
  {
    label: 'Verify a factual claim',
    prompt: 'Java was created by Snehith in 1995.',
    icon: ShieldCheck,
  },
  {
    label: 'Check this AI answer',
    prompt: 'Apollo 11 landed in 1969. Buzz Aldrin was the first person to step onto the lunar surface.',
    icon: Search,
  },
  {
    label: 'Trace the sources',
    prompt: 'Python 0.9.0 was released in February 1991 by Guido van Rossum at CWI.',
    icon: Link2,
  },
  {
    label: 'Find unsupported claims',
    prompt: 'GPT-5 has been officially launched and scored 99.8% on ARC-AGI benchmark.',
    icon: AlertTriangle,
  },
]

export function QuickActions({ onSelectPrompt }) {
  return (
    <div className="quick-actions-container" role="group" aria-label="Suggested verification prompts">
      {QUICK_PROMPTS.map((item, idx) => {
        const Icon = item.icon
        return (
          <button
            key={idx}
            type="button"
            className="quick-action-pill"
            onClick={() => onSelectPrompt(item.prompt)}
          >
            <Icon size={14} className="pill-icon" />
            <span>{item.label}</span>
          </button>
        )
      })}

      <style>{`
        .quick-actions-container {
          display: flex;
          align-items: center;
          justify-content: center;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 14px;
          max-width: var(--composer-max-width);
          width: 100%;
        }

        .quick-action-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 7px 14px;
          border-radius: var(--radius-pill);
          background: var(--surface);
          border: 1px solid var(--border);
          color: var(--text-secondary);
          font-size: 13px;
          font-weight: 500;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
          transition: transform 140ms ease, border-color 140ms ease, color 140ms ease, background-color 140ms ease;
          user-select: none;
        }

        .quick-action-pill:hover {
          transform: translateY(-1px);
          border-color: var(--accent);
          color: var(--text-primary);
          background: var(--surface-sunken);
        }

        .quick-action-pill:active {
          transform: translateY(0);
        }

        .pill-icon {
          color: var(--accent);
          opacity: 0.85;
          flex-shrink: 0;
        }
      `}</style>
    </div>
  )
}
