import React, { useState, useRef, useEffect } from 'react'
import { Paperclip, Sparkles, Database, ArrowUp, X, FileText } from 'lucide-react'

export function VerificationComposer({
  onSendMessage,
  disabled = false,
  isSticky = false,
  initialText = '',
  placeholder = 'Paste an AI-generated answer to verify...',
}) {
  const [text, setText] = useState(initialText)
  const [deepVerify, setDeepVerify] = useState(true)
  const [attachedEvidence, setAttachedEvidence] = useState(null)
  const [evidenceMode, setEvidenceMode] = useState(false)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    if (initialText) {
      setText(initialText)
      if (textareaRef.current) {
        adjustHeight(textareaRef.current)
      }
    }
  }, [initialText])

  const adjustHeight = (el) => {
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }

  const handleInput = (e) => {
    setText(e.target.value)
    adjustHeight(e.target)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleSubmit = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSendMessage({
      text: trimmed,
      deepVerify,
      evidenceAttachment: attachedEvidence,
    })
    setText('')
    setAttachedEvidence(null)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setAttachedEvidence({
        name: file.name,
        size: `${(file.size / 1024).toFixed(1)} KB`,
      })
    }
  }

  const canSubmit = text.trim().length > 0 && !disabled

  return (
    <div className={`composer-shell ${isSticky ? 'is-sticky' : 'is-centered'}`}>
      <div className="composer-container">
        {/* Attachment preview if any */}
        {attachedEvidence && (
          <div className="composer-attachment-tag">
            <FileText size={13} className="attachment-icon" />
            <span className="attachment-name truncate">{attachedEvidence.name}</span>
            <button
              type="button"
              className="attachment-remove-btn"
              onClick={() => setAttachedEvidence(null)}
              aria-label="Remove attached evidence"
            >
              <X size={12} />
            </button>
          </div>
        )}

        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="composer-textarea"
          disabled={disabled}
          aria-label="Verification input"
        />

        <div className="composer-toolbar">
          {/* Left action pills */}
          <div className="composer-actions-left">
            <button
              type="button"
              className="action-pill"
              onClick={() => fileInputRef.current?.click()}
              title="Attach primary document or passage"
              aria-label="Attach evidence file"
            >
              <Paperclip size={13} />
              <span>Attach</span>
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              style={{ display: 'none' }}
              accept=".txt,.pdf,.md,.json,.csv"
            />

            <button
              type="button"
              className={`action-pill ${evidenceMode ? 'active' : ''}`}
              onClick={() => setEvidenceMode(!evidenceMode)}
              title="Filter by primary-source repositories"
              aria-pressed={evidenceMode}
            >
              <Database size={13} />
              <span>Evidence</span>
            </button>

            <button
              type="button"
              className={`action-pill deep-verify-pill ${deepVerify ? 'active' : ''}`}
              onClick={() => setDeepVerify(!deepVerify)}
              title="Run 6-stage cross-agent verification pipeline"
              aria-pressed={deepVerify}
            >
              <Sparkles size={13} />
              <span>Deep Verify</span>
            </button>
          </div>

          {/* Right submit button */}
          <div className="composer-actions-right">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={`send-button ${canSubmit ? 'can-send' : ''}`}
              aria-label="Submit claim for verification"
              title="Verify claim (Enter)"
            >
              <ArrowUp size={18} strokeWidth={2.4} />
            </button>
          </div>
        </div>
      </div>

      <style>{`
        .composer-shell {
          width: 100%;
          display: flex;
          justify-content: center;
          transition: all var(--transition-smooth);
        }

        .composer-shell.is-centered {
          max-width: var(--composer-max-width);
        }

        .composer-shell.is-sticky {
          position: sticky;
          bottom: 0;
          padding: 16px 20px 20px;
          background: linear-gradient(to top, var(--bg) 75%, transparent);
          z-index: 20;
        }

        .composer-container {
          width: 100%;
          max-width: var(--composer-max-width);
          min-height: 112px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-card);
          box-shadow: var(--shadow-subtle);
          padding: 14px 16px 12px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          transition: border-color 160ms ease, box-shadow 160ms ease;
        }

        .composer-container:focus-within {
          border-color: rgba(109, 94, 245, 0.45);
          box-shadow: 0 0 0 3px rgba(109, 94, 245, 0.12), var(--shadow-float);
        }

        [data-theme='dark'] .composer-container:focus-within {
          border-color: rgba(139, 124, 246, 0.45);
          box-shadow: 0 0 0 3px rgba(139, 124, 246, 0.14), var(--shadow-float);
        }

        .composer-attachment-tag {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          align-self: flex-start;
          padding: 4px 10px;
          margin-bottom: 8px;
          border-radius: var(--radius-pill);
          background: var(--accent-light);
          border: 1px solid rgba(109, 94, 245, 0.2);
          font-size: 12px;
          color: var(--accent);
          font-weight: 500;
        }

        .attachment-icon {
          color: var(--accent);
        }

        .attachment-name {
          max-width: 200px;
        }

        .attachment-remove-btn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
          padding: 2px;
          border-radius: 50%;
        }

        .attachment-remove-btn:hover {
          color: var(--contradicted);
        }

        .composer-textarea {
          width: 100%;
          min-height: 48px;
          max-height: 220px;
          font-size: 15px;
          line-height: 1.55;
          color: var(--text-primary);
          background: transparent;
          border: none;
          outline: none;
          padding: 0;
          margin-bottom: 12px;
        }

        .composer-textarea::placeholder {
          color: var(--text-muted);
          font-weight: 400;
        }

        .composer-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 6px;
          gap: 12px;
        }

        .composer-actions-left {
          display: flex;
          align-items: center;
          gap: 6px;
          flex-wrap: wrap;
        }

        .action-pill {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          padding: 5px 11px;
          border-radius: var(--radius-pill);
          background: var(--surface-sunken);
          border: 1px solid transparent;
          color: var(--text-secondary);
          font-size: 12px;
          font-weight: 500;
          transition: all 140ms ease;
        }

        .action-pill:hover {
          background: var(--surface-hover);
          color: var(--text-primary);
        }

        .action-pill.active {
          background: var(--accent-light);
          color: var(--accent);
          border-color: rgba(109, 94, 245, 0.25);
        }

        [data-theme='dark'] .action-pill.active {
          border-color: rgba(139, 124, 246, 0.3);
        }

        .composer-actions-right {
          display: flex;
          align-items: center;
        }

        .send-button {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          background: var(--surface-sunken);
          color: var(--text-muted);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 140ms cubic-bezier(0.16, 1, 0.3, 1);
        }

        .send-button.can-send {
          background: var(--accent);
          color: #ffffff;
          box-shadow: 0 2px 10px var(--accent-glow);
          cursor: pointer;
        }

        .send-button.can-send:hover {
          background: var(--accent-hover);
          transform: scale(1.05);
        }

        .send-button.can-send:active {
          transform: scale(0.96);
        }
      `}</style>
    </div>
  )
}
