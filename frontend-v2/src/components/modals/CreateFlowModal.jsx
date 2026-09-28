import React, { useEffect, useState } from 'react'
import { X, ArrowRight, MessageSquare, ShieldCheck, FileCheck, Layers, GitBranch } from 'lucide-react'

// Realistic app logos as clean SVGs
function SlackIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313z" fill="#E01E5A"/>
      <path d="M8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312z" fill="#36C5F0"/>
      <path d="M18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312z" fill="#2EB67D"/>
      <path d="M15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" fill="#ECB22E"/>
    </svg>
  )
}

function LinearIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <rect width="24" height="24" rx="5" fill="#5E6AD2"/>
      <path d="M6 18L18 6M6 6L18 18" stroke="#FFFFFF" strokeWidth="2.4" strokeLinecap="round"/>
    </svg>
  )
}

function NotionIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <rect width="24" height="24" rx="5" fill="#1C1C1E"/>
      <path d="M7 6L17 18M7 18V6M17 6V18" stroke="#FFFFFF" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
}

function GmailIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <rect width="24" height="24" rx="5" fill="#EA4335"/>
      <path d="M5 8L12 13L19 8V17H5V8Z" fill="#FFFFFF"/>
    </svg>
  )
}

const TEMPLATES = [
  {
    id: 'slack-linear',
    title: 'Slack #bugs to Linear Issue',
    icons: [<SlackIcon key="slack" />, <LinearIcon key="linear" />],
    description: 'Automatically converts bug reports from Slack into trackable Linear issues',
    category: 'integration',
    prompt: 'Create a flow that listens to #bugs on Slack, summarizes the report with HalluciGuard verification, and opens a Linear ticket.',
  },
  {
    id: 'linear-slack',
    title: 'Linear Urgent Issue to Slack #oncall',
    icons: [<LinearIcon key="linear" />, <SlackIcon key="slack" />],
    description: 'Instantly alerts the on-call team when critical issues require immediate attention',
    category: 'integration',
    prompt: 'Set up an alert flow: when a Linear issue is marked Urgent, verify reproducibility and post an incident alert to Slack #oncall.',
  },
  {
    id: 'linear-notion',
    title: 'Linear Issue Closed to Notion Release Notes',
    icons: [<LinearIcon key="linear" />, <NotionIcon key="notion" />],
    description: 'Maintains up-to-date release notes by documenting completed features and fixes',
    category: 'integration',
    prompt: 'Build a release notes flow: when a Linear cycle or issue is closed, summarize changes and document them in Notion.',
  },
  {
    id: 'email-research',
    title: 'Email Invitation Research',
    icons: [<NotionIcon key="notion" />, <SlackIcon key="slack" />, <GmailIcon key="gmail" />],
    description: 'Automatically researches meeting attendees and provides context before important meetings',
    category: 'integration',
    prompt: 'Scan upcoming calendar invitations from Gmail, research participants across verified public records, and generate pre-meeting briefs.',
  },
]

export function CreateFlowModal({ isOpen, onClose, onSelectTemplate, onSelectCreateFromChat }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown)
    }
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="flow-modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="flow-modal-title">
      <div className="flow-modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="flow-modal-header">
          <h2 id="flow-modal-title" className="flow-modal-heading">Create New Flow</h2>
          <button
            type="button"
            className="flow-close-btn"
            onClick={onClose}
            aria-label="Close modal"
          >
            <X size={17} />
          </button>
        </div>

        <div className="flow-modal-body">
          {/* Top Banner: Create from chat */}
          <button
            type="button"
            className="create-from-chat-card"
            onClick={() => {
              onSelectCreateFromChat?.()
              onClose()
            }}
          >
            <div className="chat-card-icon-box">
              <MessageSquare size={17} className="chat-card-icon" />
            </div>
            <div className="chat-card-content">
              <h3 className="chat-card-title">Create from chat</h3>
              <p className="chat-card-desc">Describe what you want your flow to do in natural language</p>
            </div>
          </button>

          {/* Section: Or choose a template */}
          <div className="templates-section">
            <div className="templates-section-label">Or choose a template</div>

            <div className="templates-grid">
              {TEMPLATES.map((item) => (
                <div
                  key={item.id}
                  className="template-card"
                  onClick={() => {
                    onSelectTemplate?.(item)
                    onClose()
                  }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      onSelectTemplate?.(item)
                      onClose()
                    }
                  }}
                >
                  <div className="template-top">
                    <h4 className="template-title">{item.title}</h4>
                    <div className="template-icons-row">
                      {item.icons.map((icon, idx) => (
                        <span key={idx} className="app-icon-badge">{icon}</span>
                      ))}
                    </div>
                  </div>

                  <p className="template-desc">{item.description}</p>

                  <div className="template-arrow-wrap">
                    <ArrowRight size={14} className="template-arrow" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .flow-modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(5px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 110;
          padding: 20px;
          animation: flow-fade-in 160ms ease-out;
        }

        .flow-modal-card {
          width: 100%;
          max-width: 680px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          box-shadow: 0 20px 48px -12px rgba(0, 0, 0, 0.35);
          overflow: hidden;
          display: flex;
          flex-direction: column;
          animation: flow-scale-in 180ms cubic-bezier(0.16, 1, 0.3, 1);
        }

        .flow-modal-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 18px 24px 14px;
        }

        .flow-modal-heading {
          font-size: 16px;
          font-weight: 600;
          color: var(--text-primary);
          letter-spacing: -0.01em;
        }

        .flow-close-btn {
          color: var(--text-muted);
          padding: 4px;
          border-radius: 6px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .flow-close-btn:hover {
          color: var(--text-primary);
          background: var(--surface-sunken);
        }

        .flow-modal-body {
          padding: 0 24px 24px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        /* Create from chat card */
        .create-from-chat-card {
          width: 100%;
          display: flex;
          align-items: center;
          gap: 14px;
          padding: 16px 18px;
          border-radius: 12px;
          background: var(--surface);
          border: 1px dashed var(--border-strong);
          text-align: left;
          transition: all 140ms ease;
          cursor: pointer;
        }

        .create-from-chat-card:hover {
          border-color: var(--accent);
          background: var(--accent-light);
          transform: translateY(-1px);
        }

        .chat-card-icon-box {
          width: 36px;
          height: 36px;
          border-radius: 8px;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .chat-card-icon {
          color: var(--accent);
        }

        .chat-card-content {
          display: flex;
          flex-direction: column;
          gap: 3px;
        }

        .chat-card-title {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .chat-card-desc {
          font-size: 12.5px;
          color: var(--text-secondary);
          line-height: 1.4;
          margin: 0;
        }

        /* Templates Section */
        .templates-section {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .templates-section-label {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-secondary);
        }

        .templates-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 14px;
        }

        @media (max-width: 600px) {
          .templates-grid {
            grid-template-columns: 1fr;
          }
        }

        .template-card {
          position: relative;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 16px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          min-height: 128px;
          transition: all 140ms ease;
          cursor: pointer;
        }

        .template-card:hover {
          border-color: var(--border-strong);
          background: var(--surface-sunken);
          transform: translateY(-1.5px);
          box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        .template-top {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .template-title {
          font-size: 13.5px;
          font-weight: 600;
          color: var(--text-primary);
          line-height: 1.35;
        }

        .template-icons-row {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .app-icon-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 26px;
          height: 26px;
          border-radius: 6px;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
        }

        .template-desc {
          font-size: 12px;
          color: var(--text-secondary);
          line-height: 1.45;
          margin: 8px 0 12px;
        }

        .template-arrow-wrap {
          display: flex;
          justify-content: flex-end;
          align-items: center;
        }

        .template-arrow {
          color: var(--text-muted);
          transition: transform 140ms ease, color 140ms ease;
        }

        .template-card:hover .template-arrow {
          color: var(--accent);
          transform: translateX(3px);
        }

        @keyframes flow-fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes flow-scale-in {
          from { opacity: 0; transform: scale(0.97); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  )
}
