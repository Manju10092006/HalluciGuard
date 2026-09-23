import React, { useEffect } from 'react'
import { X, Sliders, ShieldCheck, Database, FileCheck } from 'lucide-react'

export function SettingsModal({ isOpen, onClose }) {
  const [strictness, setStrictness] = React.useState('high')
  const [autoCorrect, setAutoCorrect] = React.useState(true)
  const [primarySourcesOnly, setPrimarySourcesOnly] = React.useState(true)

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
    <div className="settings-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <Sliders size={16} className="modal-icon" />
            <h3>Verification Settings</h3>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close settings modal"
          >
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          {/* Strictness Level */}
          <div className="setting-row">
            <div className="setting-info">
              <div className="setting-title-row">
                <ShieldCheck size={15} className="setting-icon" />
                <span className="setting-title">Entailment Strictness</span>
              </div>
              <p className="setting-desc">
                Sets the NLI confidence threshold required before marking a claim as fully supported.
              </p>
            </div>
            <select
              className="setting-select"
              value={strictness}
              onChange={(e) => setStrictness(e.target.value)}
            >
              <option value="standard">Standard (80%)</option>
              <option value="high">High (92%)</option>
              <option value="critical">Critical (98%)</option>
            </select>
          </div>

          {/* Primary Sources Preference */}
          <div className="setting-row">
            <div className="setting-info">
              <div className="setting-title-row">
                <Database size={15} className="setting-icon" />
                <span className="setting-title">Prioritize Primary Documents</span>
              </div>
              <p className="setting-desc">
                Weigh institutional journals, flight logs, and repositories over secondary summarizers.
              </p>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={primarySourcesOnly}
                onChange={(e) => setPrimarySourcesOnly(e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          {/* Auto Correct Unsupported Statements */}
          <div className="setting-row">
            <div className="setting-info">
              <div className="setting-title-row">
                <FileCheck size={15} className="setting-icon" />
                <span className="setting-title">Synthesize Evidence-Grounded Correction</span>
              </div>
              <p className="setting-desc">
                When a contradiction is verified, generate the corrected answer alongside the verdict.
              </p>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={autoCorrect}
                onChange={(e) => setAutoCorrect(e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="done-btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>

      <style>{`
        .settings-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.4);
          backdrop-filter: blur(6px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 100;
          padding: 20px;
        }

        .settings-modal {
          width: 100%;
          max-width: 460px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-card);
          box-shadow: var(--shadow-modal);
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .modal-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 20px;
          border-bottom: 1px solid var(--border);
        }

        .modal-title-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .modal-icon {
          color: var(--accent);
        }

        .modal-header h3 {
          font-size: 15px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .modal-close-btn {
          color: var(--text-muted);
          padding: 4px;
          border-radius: 6px;
        }

        .modal-close-btn:hover {
          color: var(--text-primary);
          background: var(--surface-sunken);
        }

        .modal-body {
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 18px;
        }

        .setting-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
        }

        .setting-info {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .setting-title-row {
          display: flex;
          align-items: center;
          gap: 7px;
        }

        .setting-icon {
          color: var(--accent);
        }

        .setting-title {
          font-size: 13.5px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .setting-desc {
          font-size: 12px;
          color: var(--text-secondary);
          line-height: 1.45;
          margin: 0;
          max-width: 320px;
        }

        .setting-select {
          padding: 6px 10px;
          border-radius: var(--radius-sm);
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          color: var(--text-primary);
          font-size: 12.5px;
          font-weight: 500;
          outline: none;
        }

        /* Toggle switch */
        .toggle-switch {
          position: relative;
          display: inline-block;
          width: 38px;
          height: 22px;
          flex-shrink: 0;
        }

        .toggle-switch input {
          opacity: 0;
          width: 0;
          height: 0;
        }

        .toggle-slider {
          position: absolute;
          cursor: pointer;
          inset: 0;
          background-color: var(--surface-sunken);
          border: 1px solid var(--border);
          transition: 160ms;
          border-radius: 22px;
        }

        .toggle-slider:before {
          position: absolute;
          content: "";
          height: 16px;
          width: 16px;
          left: 2px;
          bottom: 2px;
          background-color: var(--text-secondary);
          transition: 160ms;
          border-radius: 50%;
        }

        .toggle-switch input:checked + .toggle-slider {
          background-color: var(--accent);
          border-color: var(--accent);
        }

        .toggle-switch input:checked + .toggle-slider:before {
          transform: translateX(16px);
          background-color: #ffffff;
        }

        .modal-footer {
          display: flex;
          justify-content: flex-end;
          padding: 12px 20px 16px;
          border-top: 1px solid var(--border);
        }

        .done-btn {
          padding: 7px 18px;
          border-radius: var(--radius-pill);
          background: var(--accent);
          color: #ffffff;
          font-size: 13px;
          font-weight: 600;
        }

        .done-btn:hover {
          background: var(--accent-hover);
        }
      `}</style>
    </div>
  )
}
