import React from 'react'

export function UserMessage({ message }) {
  return (
    <div className="user-message-row">
      <div className="user-message-bubble">
        <p className="user-message-text">{message.text}</p>
        {message.timestamp && (
          <span className="user-message-time">{message.timestamp}</span>
        )}
      </div>

      <style>{`
        .user-message-row {
          display: flex;
          justify-content: flex-end;
          width: 100%;
          margin: 16px 0;
        }

        .user-message-bubble {
          max-width: 72%;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 12px 18px;
          color: var(--text-primary);
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
          word-break: break-word;
        }

        .user-message-text {
          font-size: 15px;
          line-height: 1.55;
          margin: 0;
        }

        .user-message-time {
          display: block;
          margin-top: 6px;
          font-size: 11px;
          color: var(--text-muted);
          text-align: right;
        }
      `}</style>
    </div>
  )
}
