import React, { useState } from 'react'
import { ChevronDown, ChevronRight, Cpu } from 'lucide-react'
import { AGENT_PIPELINE } from '../../types/verification'

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
          <div className="pipeline-steps">
            {AGENT_PIPELINE.map((agent, index) => (
              <div className="pipeline-step-item" key={agent.id}>
                <div className="step-num-col">
                  <span className="step-index">0{index + 1}</span>
                  {index < AGENT_PIPELINE.length - 1 && <div className="step-line" />}
                </div>
                <div className="step-body">
                  <span className="agent-title">{agent.name}</span>
                  <span className="agent-role-desc">{agent.role}</span>
                </div>
              </div>
            ))}
          </div>
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
          padding: 12px 16px;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          border-radius: var(--radius-inline);
        }

        .pipeline-steps {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .pipeline-step-item {
          display: flex;
          gap: 12px;
        }

        .step-num-col {
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 20px;
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

        .step-line {
          width: 1px;
          flex: 1;
          background: var(--border-strong);
          margin-top: 4px;
          min-height: 12px;
        }

        .step-body {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .agent-title {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .agent-role-desc {
          font-size: 11.5px;
          color: var(--text-secondary);
          line-height: 1.4;
        }
      `}</style>
    </div>
  )
}
