import React from 'react'
import { ArrowRight, History } from 'lucide-react'

export function RecentVerificationStrip({ items, onSelectSession }) {
  if (!items || items.length === 0) return null

  const displayItems = items.slice(0, 3)

  return (
    <div className="recent-strip-container">
      <div className="recent-strip-header">
        <span className="recent-strip-title">
          <History size={13} />
          <span>Recent verifications</span>
        </span>
      </div>

      <div className="recent-strip-list">
        {displayItems.map((item) => (
          <button
            key={item.id}
            type="button"
            className="recent-strip-row"
            onClick={() => onSelectSession(item.id)}
          >
            <span className="recent-row-title truncate">{item.title}</span>
            <div className="recent-row-meta">
              <span className={`status-pill status-${item.statusBadgeTone}`}>
                {item.statusBadge || item.status}
              </span>
              <ArrowRight size={13} className="recent-row-arrow" />
            </div>
          </button>
        ))}
      </div>

      <style>{`
        .recent-strip-container {
          margin-top: 36px;
          max-width: var(--composer-max-width);
          width: 100%;
        }

        .recent-strip-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 4px 8px;
        }

        .recent-strip-title {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--text-muted);
        }

        .recent-strip-list {
          border-top: 1px solid var(--border);
        }

        .recent-strip-row {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 6px;
          border-bottom: 1px solid var(--border);
          transition: background-color 140ms ease, padding-left 140ms ease;
          border-radius: 6px;
          text-align: left;
        }

        .recent-strip-row:hover {
          background-color: var(--surface-sunken);
          padding-left: 10px;
        }

        .recent-row-title {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-secondary);
        }

        .recent-strip-row:hover .recent-row-title {
          color: var(--text-primary);
        }

        .recent-row-meta {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-shrink: 0;
        }

        .status-pill {
          display: inline-flex;
          align-items: center;
          padding: 2px 9px;
          border-radius: var(--radius-pill);
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.02em;
        }

        .status-supported {
          background: var(--supported-bg);
          color: var(--supported);
          border: 1px solid var(--supported-border);
        }

        .status-contradicted {
          background: var(--contradicted-bg);
          color: var(--contradicted);
          border: 1px solid var(--contradicted-border);
        }

        .status-insufficient {
          background: var(--insufficient-bg);
          color: var(--insufficient);
          border: 1px solid var(--insufficient-border);
        }

        .status-uncertain {
          background: var(--uncertain-bg);
          color: var(--uncertain);
          border: 1px solid var(--uncertain-border);
        }

        .recent-row-arrow {
          color: var(--text-muted);
          opacity: 0;
          transition: opacity 140ms ease, transform 140ms ease;
        }

        .recent-strip-row:hover .recent-row-arrow {
          opacity: 1;
          transform: translateX(2px);
        }
      `}</style>
    </div>
  )
}
