import React, { useState } from 'react'
import { ChevronDown, ChevronRight, ExternalLink, Globe, FileText, CheckCircle, XCircle, Info } from 'lucide-react'

export function EvidenceTrace({ summary, sources, onOpenSource }) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (!sources || sources.length === 0) return null

  const getRelationshipBadge = (relationship, tone) => {
    switch (tone) {
      case 'supported':
        return (
          <span className="source-rel-badge rel-supported">
            <CheckCircle size={11} /> Supports claim
          </span>
        )
      case 'contradicted':
        return (
          <span className="source-rel-badge rel-contradicted">
            <XCircle size={11} /> Contradicts claim
          </span>
        )
      case 'context':
      default:
        return (
          <span className="source-rel-badge rel-context">
            <Info size={11} /> Context only
          </span>
        )
    }
  }

  return (
    <div className="evidence-trace-container">
      {/* Collapsed Trigger Row */}
      <button
        type="button"
        className={`evidence-trigger-btn ${isExpanded ? 'open' : ''}`}
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <div className="trigger-left">
          {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
          <span className="trigger-title">Evidence trace</span>
          <span className="trigger-bullet">·</span>
          <span className="trigger-metric">{sources.length} sources</span>
          {summary?.supportingCount > 0 && (
            <>
              <span className="trigger-bullet">·</span>
              <span className="trigger-metric text-supported">
                {summary.supportingCount} supporting
              </span>
            </>
          )}
          {summary?.contradictingCount > 0 && (
            <>
              <span className="trigger-bullet">·</span>
              <span className="trigger-metric text-contradicted">
                {summary.contradictingCount} contradicting
              </span>
            </>
          )}
        </div>
        <span className="trigger-toggle-label">{isExpanded ? 'Collapse' : 'Inspect'}</span>
      </button>

      {/* Expanded Vertical Trail */}
      {isExpanded && (
        <div className="evidence-trail-body">
          <div className="trail-stem" />
          <div className="trail-items">
            {sources.map((src, index) => (
              <div className="trail-item-wrapper" key={src.id || index}>
                <div className="trail-node-point" />
                <div className="trail-card">
                  <div className="trail-card-header">
                    <div className="source-identity">
                      <Globe size={13} className="domain-icon" />
                      <span className="source-domain">{src.domain}</span>
                    </div>
                    {getRelationshipBadge(src.relationship, src.relationshipTone)}
                  </div>

                  <h4 className="source-title">{src.title}</h4>
                  <p className="source-excerpt">“{src.excerpt}”</p>

                  <div className="source-actions">
                    <button
                      type="button"
                      className="open-source-link"
                      onClick={() => onOpenSource?.(src)}
                    >
                      <span>Open primary record</span>
                      <ExternalLink size={12} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        .evidence-trace-container {
          margin: 14px 0 16px;
          border: 1px solid var(--border);
          border-radius: var(--radius-inline);
          background: var(--surface);
          overflow: hidden;
          transition: border-color 140ms ease;
        }

        .evidence-trace-container:hover {
          border-color: var(--border-strong);
        }

        .evidence-trigger-btn {
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

        .evidence-trigger-btn:hover {
          background: var(--surface-sunken);
        }

        .trigger-left {
          display: flex;
          align-items: center;
          gap: 7px;
          flex-wrap: wrap;
        }

        .trigger-title {
          font-weight: 600;
          color: var(--text-primary);
        }

        .trigger-bullet {
          color: var(--text-muted);
        }

        .trigger-metric {
          color: var(--text-secondary);
          font-size: 12px;
        }

        .text-supported {
          color: var(--supported);
          font-weight: 600;
        }

        .text-contradicted {
          color: var(--contradicted);
          font-weight: 600;
        }

        .trigger-toggle-label {
          font-size: 11px;
          color: var(--text-muted);
          font-weight: 500;
        }

        .evidence-trail-body {
          position: relative;
          padding: 10px 16px 16px 28px;
          border-top: 1px solid var(--border);
          background: var(--surface-sunken);
        }

        .trail-stem {
          position: absolute;
          left: 17px;
          top: 18px;
          bottom: 24px;
          width: 1px;
          background: var(--border-strong);
        }

        .trail-items {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .trail-item-wrapper {
          position: relative;
          padding-left: 14px;
        }

        .trail-node-point {
          position: absolute;
          left: -15px;
          top: 16px;
          width: 9px;
          height: 9px;
          border-radius: 50%;
          background: var(--surface);
          border: 2px solid var(--accent);
        }

        .trail-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 12px 14px;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        }

        .trail-card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 6px;
          gap: 8px;
        }

        .source-identity {
          display: flex;
          align-items: center;
          gap: 5px;
        }

        .domain-icon {
          color: var(--text-muted);
        }

        .source-domain {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-secondary);
        }

        .source-rel-badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 2px 7px;
          border-radius: var(--radius-pill);
          font-size: 10.5px;
          font-weight: 600;
        }

        .rel-supported {
          background: var(--supported-bg);
          color: var(--supported);
          border: 1px solid var(--supported-border);
        }

        .rel-contradicted {
          background: var(--contradicted-bg);
          color: var(--contradicted);
          border: 1px solid var(--contradicted-border);
        }

        .rel-context {
          background: var(--uncertain-bg);
          color: var(--uncertain);
          border: 1px solid var(--uncertain-border);
        }

        .source-title {
          font-size: 13.5px;
          font-weight: 600;
          color: var(--text-primary);
          margin-bottom: 4px;
        }

        .source-excerpt {
          font-size: 12.5px;
          color: var(--text-secondary);
          line-height: 1.45;
          font-style: italic;
          margin-bottom: 8px;
        }

        .source-actions {
          display: flex;
          justify-content: flex-end;
        }

        .open-source-link {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          font-size: 11.5px;
          font-weight: 600;
          color: var(--accent);
          transition: opacity 120ms ease;
        }

        .open-source-link:hover {
          opacity: 0.8;
          text-decoration: underline;
        }
      `}</style>
    </div>
  )
}
