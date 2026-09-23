import React from 'react'
import { ArrowDown, CheckCheck, RefreshCw } from 'lucide-react'

export function CorrectionView({ correction }) {
  if (!correction) return null

  return (
    <div className="correction-container">
      <div className="correction-header">
        <RefreshCw size={13} className="correction-icon" />
        <span className="correction-title">Evidence-Grounded Correction</span>
      </div>

      <div className="correction-flow">
        {/* Original claim */}
        <div className="correction-card original-card">
          <span className="card-tag">Original statement</span>
          <p className="claim-quote">“{correction.original}”</p>
        </div>

        <div className="arrow-divider">
          <ArrowDown size={14} />
        </div>

        {/* Corrected version */}
        <div className="correction-card corrected-card">
          <span className="card-tag tag-verified">Corrected formulation</span>
          <p className="claim-quote">“{correction.corrected}”</p>
          {correction.verificationNote && (
            <div className="correction-note">
              <CheckCheck size={12} className="check-icon" />
              <span>{correction.verificationNote}</span>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .correction-container {
          margin: 14px 0 16px;
          border: 1px solid var(--border);
          border-radius: var(--radius-inline);
          background: var(--surface);
          padding: 12px 16px 14px;
        }

        .correction-header {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 10px;
        }

        .correction-icon {
          color: var(--accent);
        }

        .correction-title {
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: var(--text-muted);
        }

        .correction-flow {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .correction-card {
          border-radius: 10px;
          padding: 10px 14px;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .original-card {
          background: var(--surface-sunken);
          border: 1px solid var(--border);
        }

        .corrected-card {
          background: var(--supported-bg);
          border: 1px solid var(--supported-border);
        }

        .card-tag {
          font-size: 11px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .tag-verified {
          color: var(--supported);
        }

        .claim-quote {
          font-size: 13.5px;
          line-height: 1.45;
          margin: 0;
          color: var(--text-primary);
        }

        .arrow-divider {
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
          padding: 2px 0;
        }

        .correction-note {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          margin-top: 4px;
          font-size: 11.5px;
          font-weight: 600;
          color: var(--supported);
        }

        .check-icon {
          flex-shrink: 0;
        }
      `}</style>
    </div>
  )
}
