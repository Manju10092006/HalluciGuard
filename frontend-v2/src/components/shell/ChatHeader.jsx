import React, { useState } from 'react'
import { PanelLeft, Sparkles, Share2, Settings, Check, Layers, User, LogIn } from 'lucide-react'

export function ChatHeader({
  title,
  statusBadge,
  statusTone = 'supported',
  onToggleSidebar,
  sidebarCollapsed,
  onOpenAppearance,
  onOpenSettings,
  onOpenCreateFlow,
  onOpenAuth,
  user,
  isHome = false,
}) {
  const [copied, setCopied] = useState(false)

  const handleShare = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(window.location.href)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <header className="chat-topbar" aria-label="Conversation navigation">
      <div className="topbar-left">
        {/* Toggle sidebar button */}
        <button
          type="button"
          className="topbar-icon-btn sidebar-toggle-btn"
          onClick={onToggleSidebar}
          aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <PanelLeft size={18} />
        </button>

        {/* Conversation Title & Status Badge */}
        {!isHome ? (
          <div className="title-status-cluster">
            <h2 className="topbar-title truncate">{title || 'Verification session'}</h2>
            {statusBadge && (
              <span className={`topbar-status-badge status-${statusTone}`}>
                {statusBadge}
              </span>
            )}
          </div>
        ) : (
          <div className="home-engine-badge">
            <span className="engine-dot" />
            <span className="engine-text">HalluciGuard Verification Engine</span>
          </div>
        )}
      </div>

      <div className="topbar-right">
        {/* Create Flow button */}
        <button
          type="button"
          className="topbar-action-pill-btn"
          onClick={onOpenCreateFlow}
          title="Create new workflow or verification template"
        >
          <Layers size={13} />
          <span className="btn-label">New Flow</span>
        </button>

        {/* Appearance modal trigger */}
        <button
          type="button"
          className="topbar-appearance-btn"
          onClick={onOpenAppearance}
          title="Customize appearance (Theme, atmosphere, density)"
          aria-label="Open appearance settings"
        >
          <Sparkles size={14} />
          <span className="btn-label">Appearance</span>
        </button>

        {/* Share button */}
        <button
          type="button"
          className="topbar-icon-btn"
          onClick={handleShare}
          title={copied ? 'Link copied!' : 'Share verification record'}
          aria-label="Share verification"
        >
          {copied ? <Check size={16} className="text-supported" /> : <Share2 size={16} />}
        </button>

        {/* Settings button */}
        <button
          type="button"
          className="topbar-icon-btn"
          onClick={onOpenSettings}
          title="Verification parameters"
          aria-label="Verification settings"
        >
          <Settings size={16} />
        </button>

        {/* Auth / Profile trigger */}
        <button
          type="button"
          className="topbar-auth-btn"
          onClick={onOpenAuth}
          title={user ? `${user.name} (${user.email})` : 'Sign in or sign up'}
          aria-label="User account"
        >
          {user ? (
            <div className="topbar-avatar-circle">
              <User size={14} />
            </div>
          ) : (
            <div className="topbar-signin-pill">
              <LogIn size={13} />
              <span>Sign In</span>
            </div>
          )}
        </button>
      </div>

      <style>{`
        .chat-topbar {
          height: 48px;
          min-height: 48px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 16px;
          border-bottom: 1px solid var(--border);
          background: var(--bg);
          z-index: 10;
          transition: background-color var(--transition-smooth), border-color var(--transition-smooth);
        }

        .topbar-left {
          display: flex;
          align-items: center;
          gap: 12px;
          min-width: 0;
        }

        .title-status-cluster {
          display: flex;
          align-items: center;
          gap: 10px;
          min-width: 0;
        }

        .topbar-title {
          font-size: 14.5px;
          font-weight: 600;
          color: var(--text-primary);
          letter-spacing: -0.01em;
          max-width: 320px;
        }

        .topbar-status-badge {
          display: inline-flex;
          align-items: center;
          padding: 2px 8px;
          border-radius: var(--radius-pill);
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.02em;
          flex-shrink: 0;
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

        .home-engine-badge {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          padding: 3px 10px;
          border-radius: var(--radius-pill);
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          font-size: 11.5px;
          color: var(--text-secondary);
          font-weight: 500;
        }

        .engine-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--supported);
          box-shadow: 0 0 6px var(--supported);
        }

        .topbar-right {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .topbar-icon-btn {
          width: 32px;
          height: 32px;
          border-radius: var(--radius-sm);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-secondary);
          transition: all 120ms ease;
        }

        .topbar-icon-btn:hover {
          color: var(--text-primary);
          background: var(--surface-sunken);
        }

        .topbar-action-pill-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 5px 11px;
          border-radius: var(--radius-pill);
          background: var(--accent-light);
          border: 1px solid rgba(109, 94, 245, 0.25);
          color: var(--accent);
          font-size: 12px;
          font-weight: 600;
          transition: all 140ms ease;
        }

        .topbar-action-pill-btn:hover {
          background: rgba(109, 94, 245, 0.18);
          transform: translateY(-0.5px);
        }

        .topbar-appearance-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 5px 11px;
          border-radius: var(--radius-pill);
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          color: var(--text-secondary);
          font-size: 12px;
          font-weight: 500;
          transition: all 140ms ease;
        }

        .topbar-appearance-btn:hover {
          background: var(--surface-hover);
          color: var(--text-primary);
          border-color: var(--border-strong);
        }

        .topbar-auth-btn {
          margin-left: 2px;
        }

        .topbar-avatar-circle {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--accent);
          transition: border-color 120ms ease;
        }

        .topbar-avatar-circle:hover {
          border-color: var(--accent);
        }

        .topbar-signin-pill {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          padding: 4px 10px;
          border-radius: var(--radius-pill);
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          color: var(--text-primary);
          font-size: 11.5px;
          font-weight: 600;
        }

        .topbar-signin-pill:hover {
          background: var(--accent-light);
          color: var(--accent);
        }

        @media (max-width: 640px) {
          .topbar-appearance-btn .btn-label,
          .topbar-action-pill-btn .btn-label {
            display: none;
          }
          .topbar-appearance-btn,
          .topbar-action-pill-btn {
            padding: 6px;
            border-radius: 50%;
          }
        }
      `}</style>
    </header>
  )
}
