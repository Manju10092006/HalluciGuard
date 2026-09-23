import React, { useEffect } from 'react'
import { X, ExternalLink, Globe, BookOpenCheck, ShieldCheck, AlertOctagon } from 'lucide-react'

export function SourceDrawer({ source, isOpen, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown)
    }
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen || !source) return null

  return (
    <div className="drawer-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div className="drawer-title-group">
            <BookOpenCheck size={16} className="drawer-icon" />
            <h3>Primary Source Record</h3>
          </div>
          <button
            type="button"
            className="drawer-close-btn"
            onClick={onClose}
            aria-label="Close source drawer"
          >
            <X size={16} />
          </button>
        </div>

        <div className="drawer-body">
          {/* Source domain & relationship */}
          <div className="source-meta-row">
            <div className="source-origin">
              <Globe size={14} className="globe-icon" />
              <span className="domain-text">{source.domain}</span>
            </div>
            <span className={`relationship-badge tone-${source.relationshipTone}`}>
              {source.relationship}
            </span>
          </div>

          <h2 className="source-headline">{source.title}</h2>

          <div className="section-divider" />

          {/* Full passage citation */}
          <div className="passage-section">
            <label className="section-heading">Verified Passage</label>
            <blockquote className="passage-quote">
              “{source.excerpt}”
            </blockquote>
          </div>

          {/* Provenance details */}
          <div className="provenance-section">
            <label className="section-heading">Evidence Lineage</label>
            <div className="provenance-grid">
              <div className="provenance-item">
                <span className="prov-label">Entailment Status</span>
                <span className="prov-val">
                  {source.relationshipTone === 'contradicted' ? 'Direct Contradiction' : 'Factual Entailment'}
                </span>
              </div>
              <div className="provenance-item">
                <span className="prov-label">Extraction Method</span>
                <span className="prov-val">Passage-level BGE Retrieval</span>
              </div>
              <div className="provenance-item">
                <span className="prov-label">Corpus Index</span>
                <span className="prov-val">Authoritative Primary Record</span>
              </div>
            </div>
          </div>
        </div>

        <div className="drawer-footer">
          {source.url ? (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="external-link-btn"
            >
              <span>Visit External Source</span>
              <ExternalLink size={14} />
            </a>
          ) : (
            <button type="button" className="close-btn" onClick={onClose}>
              Close
            </button>
          )}
        </div>
      </div>

      <style>{`
        .drawer-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.35);
          backdrop-filter: blur(4px);
          display: flex;
          justify-content: flex-end;
          z-index: 120;
          animation: overlay-fade 140ms ease-out;
        }

        .drawer-panel {
          width: 100%;
          max-width: 420px;
          height: 100%;
          background: var(--surface);
          border-left: 1px solid var(--border);
          box-shadow: var(--shadow-modal);
          display: flex;
          flex-direction: column;
          animation: drawer-slide-in 200ms cubic-bezier(0.16, 1, 0.3, 1);
        }

        .drawer-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 20px;
          border-bottom: 1px solid var(--border);
        }

        .drawer-title-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .drawer-icon {
          color: var(--accent);
        }

        .drawer-header h3 {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .drawer-close-btn {
          color: var(--text-muted);
          padding: 4px;
          border-radius: 6px;
        }

        .drawer-close-btn:hover {
          color: var(--text-primary);
          background: var(--surface-sunken);
        }

        .drawer-body {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .source-meta-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .source-origin {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .globe-icon {
          color: var(--text-muted);
        }

        .domain-text {
          font-size: 12.5px;
          font-weight: 600;
          color: var(--text-secondary);
        }

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

        .source-headline {
          font-size: 18px;
          font-weight: 600;
          line-height: 1.35;
          color: var(--text-primary);
        }

        .section-divider {
          height: 1px;
          background: var(--border);
          margin: 4px 0;
        }

        .section-heading {
          font-size: 11.5px;
          font-weight: 700;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.04em;
          display: block;
          margin-bottom: 8px;
        }

        .passage-quote {
          background: var(--surface-sunken);
          border-left: 3px solid var(--accent);
          padding: 12px 14px;
          border-radius: 6px;
          font-size: 13.5px;
          line-height: 1.55;
          color: var(--text-primary);
          font-style: italic;
        }

        .provenance-grid {
          display: flex;
          flex-direction: column;
          gap: 10px;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 12px 14px;
        }

        .provenance-item {
          display: flex;
          justify-content: space-between;
          font-size: 12.5px;
        }

        .prov-label {
          color: var(--text-secondary);
        }

        .prov-val {
          font-weight: 600;
          color: var(--text-primary);
        }

        .drawer-footer {
          padding: 16px 20px;
          border-top: 1px solid var(--border);
          display: flex;
          justify-content: flex-end;
        }

        .external-link-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 8px 16px;
          border-radius: var(--radius-pill);
          background: var(--accent);
          color: #ffffff;
          font-size: 13px;
          font-weight: 600;
          transition: background-color 140ms ease;
        }

        .external-link-btn:hover {
          background: var(--accent-hover);
        }

        @keyframes overlay-fade {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes drawer-slide-in {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
    </div>
  )
}
