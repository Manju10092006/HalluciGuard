import React, { useState } from 'react'
import { ChevronDown, ChevronRight, Layers } from 'lucide-react'

export function ClaimList({ claims }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeClaimIndex, setActiveClaimIndex] = useState(null)

  if (!claims || claims.length === 0) return null

  const toggleClaim = (index) => {
    setActiveClaimIndex(activeClaimIndex === index ? null : index)
  }

  return (
    <div className="claims-inspection-card">
      <button
        type="button"
        className={`claims-trigger-row ${isExpanded ? 'open' : ''}`}
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <div className="claims-trigger-left">
          {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
          <Layers size={14} className="claims-icon" />
          <span className="claims-title">
            {claims.length} {claims.length === 1 ? 'claim' : 'claims'} analyzed
          </span>
        </div>
        <span className="claims-toggle-hint">{isExpanded ? 'Hide' : 'Inspect'}</span>
      </button>

      {isExpanded && (
        <div className="claims-body">
          {claims.map((claim, idx) => {
            const isOpen = activeClaimIndex === idx
            return (
              <div
                key={claim.number || idx}
                className={`claim-item-row ${isOpen ? 'is-open' : ''}`}
              >
                <button
                  type="button"
                  className="claim-summary-btn"
                  onClick={() => toggleClaim(idx)}
                  aria-expanded={isOpen}
                >
                  <div className="claim-number-and-text">
                    <span className="claim-number">{claim.number}</span>
                    <span className="claim-text truncate">{claim.text}</span>
                  </div>
                  <div className="claim-badge-group">
                    <span className={`claim-status-pill status-${claim.tone}`}>
                      {claim.status}
                    </span>
                    {isOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                  </div>
                </button>

                {isOpen && (
                  <div className="claim-details-panel">
                    <div className="claim-detail-item">
                      <span className="detail-label">Full claim statement</span>
                      <p className="detail-value">“{claim.text}”</p>
                    </div>

                    {claim.verdict && (
                      <div className="claim-detail-item">
                        <span className="detail-label">Verification verdict</span>
                        <p className="detail-value verdict-highlight">{claim.verdict}</p>
                      </div>
                    )}

                    {claim.evidenceExcerpt && (
                      <div className="claim-detail-item">
                        <span className="detail-label">Grounding evidence</span>
                        <p className="detail-value evidence-box">
                          {claim.evidenceExcerpt}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      <style>{`
        .claims-inspection-card {
          margin: 12px 0 16px;
          border: 1px solid var(--border);
          border-radius: var(--radius-inline);
          background: var(--surface);
          overflow: hidden;
        }

        .claims-trigger-row {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 14px;
          background: transparent;
          font-size: 13px;
          color: var(--text-primary);
          transition: background-color 140ms ease;
        }

        .claims-trigger-row:hover {
          background: var(--surface-sunken);
        }

        .claims-trigger-left {
          display: flex;
          align-items: center;
          gap: 7px;
        }

        .claims-icon {
          color: var(--accent);
        }

        .claims-title {
          font-weight: 600;
        }

        .claims-toggle-hint {
          font-size: 11px;
          color: var(--text-muted);
          font-weight: 500;
        }

        .claims-body {
          padding: 8px 12px 12px;
          border-top: 1px solid var(--border);
          background: var(--surface-sunken);
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .claim-item-row {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 8px;
          overflow: hidden;
          transition: border-color 140ms ease;
        }

        .claim-item-row.is-open {
          border-color: var(--border-strong);
        }

        .claim-summary-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 12px;
          gap: 12px;
          text-align: left;
          font-size: 13px;
        }

        .claim-summary-btn:hover {
          background: var(--surface-sunken);
        }

        .claim-number-and-text {
          display: flex;
          align-items: center;
          gap: 9px;
          min-width: 0;
        }

        .claim-number {
          font-family: var(--font-mono);
          font-size: 11px;
          font-weight: 600;
          color: var(--text-muted);
          flex-shrink: 0;
        }

        .claim-text {
          color: var(--text-primary);
          font-size: 13px;
        }

        .claim-badge-group {
          display: flex;
          align-items: center;
          gap: 6px;
          flex-shrink: 0;
          color: var(--text-muted);
        }

        .claim-status-pill {
          padding: 2px 7px;
          border-radius: var(--radius-pill);
          font-size: 10.5px;
          font-weight: 600;
          letter-spacing: 0.02em;
        }

        .claim-details-panel {
          padding: 10px 14px 12px;
          border-top: 1px solid var(--border);
          background: var(--surface-sunken);
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .claim-detail-item {
          display: flex;
          flex-direction: column;
          gap: 3px;
        }

        .detail-label {
          font-size: 11px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .detail-value {
          font-size: 12.5px;
          color: var(--text-secondary);
          line-height: 1.45;
          margin: 0;
        }

        .verdict-highlight {
          color: var(--text-primary);
          font-weight: 500;
        }

        .evidence-box {
          background: var(--surface);
          border-left: 2px solid var(--accent);
          padding: 6px 10px;
          border-radius: 4px;
          font-style: italic;
        }
      `}</style>
    </div>
  )
}
