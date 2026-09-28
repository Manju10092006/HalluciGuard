import React, { useEffect, useRef } from 'react'
import { X, Sun, Moon, Laptop, Sparkles, Sliders, Check } from 'lucide-react'

export function AppearanceModal({
  isOpen,
  onClose,
  theme,
  setTheme,
  atmosphere,
  setAtmosphere,
  density,
  setDensity,
}) {
  const modalRef = useRef(null)

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown)
      modalRef.current?.focus()
    }
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="appearance-modal-title">
      <div
        className="appearance-modal-card"
        ref={modalRef}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title-group">
            <Sparkles size={16} className="modal-title-icon" />
            <h3 id="appearance-modal-title">Appearance</h3>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close appearance modal"
          >
            <X size={16} />
          </button>
        </div>

        <div className="modal-content-body">
          {/* Theme Section */}
          <div className="appearance-section">
            <label className="section-label">Interface Theme</label>
            <div className="segmented-control" role="radiogroup">
              <button
                type="button"
                className={`segment-btn ${theme === 'system' ? 'selected' : ''}`}
                onClick={() => setTheme('system')}
                role="radio"
                aria-checked={theme === 'system'}
              >
                <Laptop size={14} />
                <span>System</span>
              </button>
              <button
                type="button"
                className={`segment-btn ${theme === 'light' ? 'selected' : ''}`}
                onClick={() => setTheme('light')}
                role="radio"
                aria-checked={theme === 'light'}
              >
                <Sun size={14} />
                <span>Light</span>
              </button>
              <button
                type="button"
                className={`segment-btn ${theme === 'dark' ? 'selected' : ''}`}
                onClick={() => setTheme('dark')}
                role="radio"
                aria-checked={theme === 'dark'}
              >
                <Moon size={14} />
                <span>Dark</span>
              </button>
            </div>
          </div>

          {/* Background Atmosphere Section */}
          <div className="appearance-section">
            <label className="section-label">Background Atmosphere</label>
            <div className="atmosphere-grid">
              <button
                type="button"
                className={`atmosphere-card ${atmosphere === 'neutral' ? 'active' : ''}`}
                onClick={() => setAtmosphere('neutral')}
              >
                <div className="preview-swatch neutral-swatch" />
                <div className="swatch-label-row">
                  <span>Pure Neutral</span>
                  {atmosphere === 'neutral' && <Check size={13} className="check-mark" />}
                </div>
              </button>

              <button
                type="button"
                className={`atmosphere-card ${atmosphere === 'mesh' ? 'active' : ''}`}
                onClick={() => setAtmosphere('mesh')}
              >
                <div className="preview-swatch mesh-swatch" />
                <div className="swatch-label-row">
                  <span>Atmospheric Mesh</span>
                  {atmosphere === 'mesh' && <Check size={13} className="check-mark" />}
                </div>
              </button>
            </div>
          </div>

          {/* Interface Density */}
          <div className="appearance-section">
            <label className="section-label">Interface Spacing</label>
            <div className="segmented-control">
              <button
                type="button"
                className={`segment-btn ${density === 'comfortable' ? 'selected' : ''}`}
                onClick={() => setDensity('comfortable')}
              >
                <Sliders size={13} />
                <span>Comfortable</span>
              </button>
              <button
                type="button"
                className={`segment-btn ${density === 'compact' ? 'selected' : ''}`}
                onClick={() => setDensity('compact')}
              >
                <Sliders size={13} />
                <span>Compact</span>
              </button>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="save-btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>

      <style>{`
        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.4);
          backdrop-filter: blur(6px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 100;
          padding: 20px;
          animation: modal-fade-in 160ms ease-out;
        }

        .appearance-modal-card {
          width: 100%;
          max-width: 440px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-card);
          box-shadow: var(--shadow-modal);
          overflow: hidden;
          display: flex;
          flex-direction: column;
          outline: none;
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

        .modal-title-icon {
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

        .modal-content-body {
          padding: 18px 20px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .appearance-section {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .section-label {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .segmented-control {
          display: flex;
          background: var(--surface-sunken);
          padding: 3px;
          border-radius: var(--radius-pill);
          border: 1px solid var(--border);
        }

        .segment-btn {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          padding: 6px 12px;
          border-radius: var(--radius-pill);
          font-size: 12.5px;
          font-weight: 500;
          color: var(--text-secondary);
          transition: all 140ms ease;
        }

        .segment-btn.selected {
          background: var(--surface);
          color: var(--text-primary);
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
          font-weight: 600;
        }

        .atmosphere-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
        }

        .atmosphere-card {
          display: flex;
          flex-direction: column;
          gap: 8px;
          padding: 10px;
          background: var(--surface-sunken);
          border: 1.5px solid var(--border);
          border-radius: 12px;
          text-align: left;
          transition: all 140ms ease;
        }

        .atmosphere-card.active {
          border-color: var(--accent);
          background: var(--accent-light);
        }

        .preview-swatch {
          width: 100%;
          height: 52px;
          border-radius: 8px;
          border: 1px solid var(--border);
        }

        .neutral-swatch {
          background: var(--bg);
        }

        .mesh-swatch {
          background: radial-gradient(at 15% 15%, rgba(109, 94, 245, 0.3) 0px, transparent 60%),
                      radial-gradient(at 85% 85%, rgba(63, 174, 106, 0.25) 0px, transparent 60%),
                      var(--bg);
        }

        .swatch-label-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 12px;
          font-weight: 500;
          color: var(--text-primary);
        }

        .check-mark {
          color: var(--accent);
        }

        .modal-footer {
          display: flex;
          justify-content: flex-end;
          padding: 12px 20px 16px;
          border-top: 1px solid var(--border);
        }

        .save-btn {
          padding: 7px 18px;
          border-radius: var(--radius-pill);
          background: var(--accent);
          color: #ffffff;
          font-size: 13px;
          font-weight: 600;
        }

        .save-btn:hover {
          background: var(--accent-hover);
        }

        @keyframes modal-fade-in {
          from {
            opacity: 0;
            transform: scale(0.97);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
      `}</style>
    </div>
  )
}
