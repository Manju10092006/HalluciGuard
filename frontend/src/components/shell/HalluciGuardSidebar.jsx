import React, { useState } from 'react'
import {
  PanelLeftClose,
  PanelLeft,
  Plus,
  Home,
  ShieldCheck,
  Database,
  History,
  ChevronsUpDown,
  User,
  Check,
  Sparkles,
  LogOut,
  Layers,
  GitBranch,
  Cpu,
  Boxes,
  LogIn,
  Gift,
  ChevronDown,
} from 'lucide-react'
import { BrandMark } from '../common/BrandMark'

export function HalluciGuardSidebar({
  collapsed,
  onToggleCollapse,
  activeNav,
  setActiveNav,
  onNewVerification,
  onOpenCreateFlow,
  onOpenAuth,
  user,
  onLogout,
  recentSessions,
  activeSessionId,
  onSelectSession,
  isMobileOpen,
  onCloseMobile,
}) {
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const [sidebarTab, setSidebarTab] = useState('chats') // 'chats' | 'flow-runs'

  const mockFlowRuns = [
    { id: 'fr-1', name: 'Slack #bugs to Linear Issue', status: 'Active', time: '10m ago', tone: 'supported' },
    { id: 'fr-2', name: 'Linear Urgent to Slack #oncall', status: 'Triggered', time: '1h ago', tone: 'contradicted' },
    { id: 'fr-3', name: 'Email Invitation Research', status: 'Completed', time: 'Yesterday', tone: 'supported' },
  ]

  // Group recent sessions by relative time
  const timeGroups = ['Today', 'Yesterday', '2 days ago', '3 days ago', 'Last week']
  const groupedSessions = timeGroups.reduce((acc, group) => {
    const items = recentSessions.filter((s) => (s.timeGroup || 'Yesterday') === group)
    if (items.length > 0) acc[group] = items
    return acc
  }, {})

  const handleNavClick = (navId) => {
    setActiveNav(navId)
    if (navId === 'home') {
      onNewVerification()
    } else if (navId === 'flows') {
      onOpenCreateFlow()
    }
    onCloseMobile()
  }

  const handleSessionClick = (id) => {
    onSelectSession(id)
    onCloseMobile()
  }

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      <aside
        className={`halluciguard-sidebar ${collapsed ? 'is-collapsed' : ''} ${
          isMobileOpen ? 'is-mobile-open' : ''
        }`}
        aria-label="Application sidebar"
      >
        {/* Top: Workspace row matching "Cofounder ⌄" in Image 1 */}
        <div className="sidebar-top-row">
          <div className="workspace-selector-btn" onClick={onNewVerification} role="button" tabIndex={0}>
            <BrandMark size={24} />
            {!collapsed && (
              <div className="workspace-name-cluster">
                <span className="workspace-name">HalluciGuard</span>
                <ChevronDown size={14} className="workspace-chevron" />
              </div>
            )}
          </div>

          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <PanelLeft size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>

        {/* Invite Friends / Earn Credits banner (from Image 1) */}
        {!collapsed && (
          <div className="sidebar-invite-strip">
            <button
              type="button"
              className="invite-credits-btn"
              onClick={() => alert('Invite referral link copied: https://halluciguard.ai/join/judha')}
            >
              <Gift size={13} className="gift-icon" />
              <span>Invite Friends, Earn Credits</span>
            </button>
          </div>
        )}

        {/* Primary action: + New Chat / New Verification */}
        <div className="sidebar-primary-action-wrap">
          <button
            type="button"
            className="new-verification-btn"
            onClick={onNewVerification}
            title="Start new chat / verification"
          >
            <Plus size={16} strokeWidth={2.4} />
            {!collapsed && <span>New Chat</span>}
          </button>
        </div>

        {/* Navigation: Flows, Memory, Integrations (as in Image 1) */}
        <nav className="sidebar-nav-list" aria-label="Main Navigation">
          <button
            type="button"
            className={`nav-item-btn ${activeNav === 'flows' ? 'is-active' : ''}`}
            onClick={() => {
              setActiveNav('flows')
              onOpenCreateFlow()
            }}
            title={collapsed ? 'Flows' : undefined}
          >
            {activeNav === 'flows' && <div className="active-indicator-bar" />}
            <Layers size={16} className="nav-item-icon" />
            {!collapsed && (
              <div className="nav-label-with-badge">
                <span>Flows</span>
                <span className="new-badge">Templates</span>
              </div>
            )}
          </button>

          <button
            type="button"
            className={`nav-item-btn ${activeNav === 'memory' ? 'is-active' : ''}`}
            onClick={() => handleNavClick('memory')}
            title={collapsed ? 'Memory' : undefined}
          >
            {activeNav === 'memory' && <div className="active-indicator-bar" />}
            <Database size={16} className="nav-item-icon" />
            {!collapsed && <span className="nav-item-label">Memory</span>}
          </button>

          <button
            type="button"
            className={`nav-item-btn ${activeNav === 'integrations' ? 'is-active' : ''}`}
            onClick={() => handleNavClick('integrations')}
            title={collapsed ? 'Integrations' : undefined}
          >
            {activeNav === 'integrations' && <div className="active-indicator-bar" />}
            <Boxes size={16} className="nav-item-icon" />
            {!collapsed && <span className="nav-item-label">Integrations</span>}
          </button>
        </nav>

        {/* Sub-tabs: Chats / Flow Runs (as in Image 1) */}
        {!collapsed && (
          <div className="sidebar-subtabs-row">
            <button
              type="button"
              className={`subtab-btn ${sidebarTab === 'chats' ? 'active' : ''}`}
              onClick={() => setSidebarTab('chats')}
            >
              Chats
            </button>
            <button
              type="button"
              className={`subtab-btn ${sidebarTab === 'flow-runs' ? 'active' : ''}`}
              onClick={() => setSidebarTab('flow-runs')}
            >
              Flow Runs
            </button>
          </div>
        )}

        {/* Sub-tab content (Chats list or Flow Runs list) */}
        {!collapsed && (
          <div className="recent-verifications-area">
            {sidebarTab === 'chats' ? (
              <div className="recent-scroll-list">
                {Object.entries(groupedSessions).map(([group, sessions]) => (
                  <div key={group} className="time-group-block">
                    <div className="time-group-label">{group}</div>
                    {sessions.map((session) => (
                      <button
                        key={session.id}
                        type="button"
                        className={`recent-session-item ${
                          activeSessionId === session.id ? 'is-selected' : ''
                        }`}
                        onClick={() => handleSessionClick(session.id)}
                      >
                        <span className="session-title-text truncate">{session.title}</span>
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <div className="recent-scroll-list">
                <div className="time-group-label">Recent Automated Runs</div>
                {mockFlowRuns.map((fr) => (
                  <div key={fr.id} className="flow-run-item">
                    <div className="flow-run-top">
                      <span className="flow-run-title truncate">{fr.name}</span>
                      <span className={`flow-run-pill status-${fr.tone}`}>{fr.status}</span>
                    </div>
                    <span className="flow-run-time">{fr.time}</span>
                  </div>
                ))}
                <button
                  type="button"
                  className="create-flow-quick-btn"
                  onClick={onOpenCreateFlow}
                >
                  <Plus size={13} />
                  <span>Create new flow</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* Bottom User Area */}
        <div className="sidebar-user-section">
          {userMenuOpen && (
            <div className="user-dropdown-popover">
              <div className="popover-user-info">
                <span className="popover-name">{user?.name || 'Judha Maygustya'}</span>
                <span className="popover-email">{user?.email || 'judha.design@halluciguard.ai'}</span>
              </div>
              <div className="popover-divider" />
              <div className="popover-plan-row">
                <div className="plan-badge-pill">
                  <Sparkles size={12} />
                  <span>{user?.plan || 'HalluciGuard Pro'}</span>
                </div>
              </div>
              <div className="popover-divider" />
              <button
                type="button"
                className="popover-menu-item"
                onClick={() => {
                  setUserMenuOpen(false)
                  onOpenAuth()
                }}
              >
                <LogIn size={13} />
                <span>Sign in / Switch account</span>
              </button>
              <button
                type="button"
                className="popover-menu-item text-danger"
                onClick={() => {
                  setUserMenuOpen(false)
                  onLogout?.()
                }}
              >
                <LogOut size={13} />
                <span>Log out</span>
              </button>
            </div>
          )}

          <button
            type="button"
            className="user-profile-btn"
            onClick={() => setUserMenuOpen(!userMenuOpen)}
            aria-expanded={userMenuOpen}
            aria-label="User account and settings"
          >
            <div className="user-avatar-circle">
              <User size={15} />
            </div>
            {!collapsed && (
              <>
                <div className="user-details-col">
                  <span className="user-name-text truncate">{user?.name || 'Judha Maygustya'}</span>
                  <span className="user-plan-label">{user?.plan || 'HalluciGuard Pro'}</span>
                </div>
                <ChevronsUpDown size={14} className="user-chevron" />
              </>
            )}
          </button>
        </div>
      </aside>

      <style>{`
        .halluciguard-sidebar {
          width: var(--sidebar-width);
          min-width: var(--sidebar-width);
          height: 100vh;
          background: rgba(250, 249, 244, 0.94);
          backdrop-filter: blur(24px) saturate(1.08);
          border-right: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          transition: width 280ms cubic-bezier(0.16, 1, 0.3, 1), min-width 280ms cubic-bezier(0.16, 1, 0.3, 1), transform 240ms ease-out, opacity 180ms ease;
          z-index: 50;
          user-select: none;
        }

        .halluciguard-sidebar.is-collapsed {
          width: var(--sidebar-collapsed-width);
          min-width: var(--sidebar-collapsed-width);
          overflow: hidden;
          opacity: 0;
          border-right-color: transparent;
          pointer-events: none;
        }

        /* Top Row */
        .sidebar-top-row {
          height: 60px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 14px;
          border-bottom: 1px solid rgba(26, 32, 26, 0.07);
        }

        .workspace-selector-btn {
          display: flex;
          align-items: center;
          gap: 9px;
          cursor: pointer;
          padding: 4px;
          border-radius: 10px;
          transition: background-color 120ms ease;
        }

        .workspace-selector-btn:hover {
          background-color: var(--surface-sunken);
        }

        .workspace-name-cluster {
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .workspace-name {
          font-size: 14.5px;
          font-weight: 500;
          letter-spacing: -0.01em;
          color: var(--text-primary);
        }

        .workspace-chevron {
          color: var(--text-muted);
        }

        .sidebar-collapse-btn {
          width: 28px;
          height: 28px;
          border-radius: 999px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
          transition: all 120ms ease;
        }

        .sidebar-collapse-btn:hover {
          color: var(--text-primary);
          background: var(--surface-sunken);
        }

        /* Invite banner */
        .sidebar-invite-strip {
          padding: 10px 12px 2px;
        }

        .invite-credits-btn {
          width: 100%;
          display: flex;
          align-items: center;
          gap: 7px;
          padding: 6px 10px;
          border-radius: 999px;
          background: var(--accent-light);
          color: var(--accent);
          font-size: 11.5px;
          font-weight: 500;
          transition: background-color 140ms ease;
        }

        .invite-credits-btn:hover {
          background: rgba(34, 124, 104, 0.12);
        }

        .gift-icon {
          flex-shrink: 0;
        }

        /* New Chat Action */
        .sidebar-primary-action-wrap {
          padding: 10px 12px 6px;
        }

        .new-verification-btn {
          width: 100%;
          height: 38px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          border-radius: var(--radius-pill);
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          color: var(--text-primary);
          font-size: 13px;
          font-weight: 500;
          transition: all 140ms ease;
        }

        .new-verification-btn:hover {
          border-color: var(--accent);
          background: var(--accent-light);
          color: var(--accent);
        }

        /* Nav List */
        .sidebar-nav-list {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: 6px 8px 10px;
          border-bottom: 1px solid var(--border);
        }

        .nav-item-btn {
          position: relative;
          width: 100%;
          height: 34px;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 0 10px;
          border-radius: 999px;
          color: var(--text-secondary);
          font-size: 13px;
          font-weight: 500;
          transition: all 120ms ease;
        }

        .nav-item-btn:hover {
          color: var(--text-primary);
          background: var(--surface-sunken);
        }

        .nav-item-btn.is-active {
          color: var(--text-primary);
          background: var(--surface-sunken);
          font-weight: 500;
        }

        .active-indicator-bar {
          position: absolute;
          left: 0;
          top: 6px;
          bottom: 6px;
          width: 2.5px;
          border-radius: 2px;
          background: var(--accent);
        }

        .nav-item-icon {
          flex-shrink: 0;
        }

        .nav-label-with-badge {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .new-badge {
          font-size: 10px;
          padding: 1px 6px;
          border-radius: var(--radius-pill);
          background: var(--accent-light);
          color: var(--accent);
          font-weight: 600;
        }

        .is-collapsed .nav-item-btn {
          justify-content: center;
          padding: 0;
        }

        /* Subtabs: Chats / Flow Runs */
        .sidebar-subtabs-row {
          display: flex;
          padding: 10px 12px 4px;
          gap: 14px;
          border-bottom: 1px solid var(--border);
        }

        .subtab-btn {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          padding-bottom: 6px;
          border-bottom: 2px solid transparent;
          transition: all 120ms ease;
        }

        .subtab-btn:hover {
          color: var(--text-primary);
        }

        .subtab-btn.active {
          color: var(--text-primary);
          border-bottom-color: var(--accent);
        }

        /* Recent Verifications / Flows */
        .recent-verifications-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          padding: 10px 10px 8px;
        }

        .recent-scroll-list {
          flex: 1;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .time-group-block {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .time-group-label {
          font-size: 10.5px;
          font-weight: 600;
          color: var(--text-muted);
          padding: 2px 8px;
          letter-spacing: 0.02em;
        }

        .recent-session-item {
          width: 100%;
          height: 30px;
          display: flex;
          align-items: center;
          padding: 0 8px;
          border-radius: 999px;
          color: var(--text-secondary);
          font-size: 13px;
          text-align: left;
          transition: all 120ms ease;
        }

        .recent-session-item:hover {
          background: var(--surface-sunken);
          color: var(--text-primary);
        }

        .recent-session-item.is-selected {
          background: var(--surface-sunken);
          color: var(--accent);
          font-weight: 500;
        }

        /* Flow Runs */
        .flow-run-item {
          padding: 8px;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          border-radius: 8px;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .flow-run-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 6px;
        }

        .flow-run-title {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .flow-run-pill {
          font-size: 9.5px;
          font-weight: 700;
          padding: 1px 6px;
          border-radius: var(--radius-pill);
        }

        .flow-run-time {
          font-size: 10.5px;
          color: var(--text-muted);
        }

        .create-flow-quick-btn {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          padding: 8px;
          border-radius: var(--radius-sm);
          border: 1px dashed var(--border-strong);
          color: var(--accent);
          font-size: 12px;
          font-weight: 500;
          margin-top: 6px;
          transition: background-color 140ms ease;
        }

        .create-flow-quick-btn:hover {
          background: var(--accent-light);
        }

        /* User Area */
        .sidebar-user-section {
          position: relative;
          padding: 10px;
          border-top: 1px solid var(--border);
        }

        .user-profile-btn {
          width: 100%;
          display: flex;
          align-items: center;
          gap: 9px;
          padding: 6px 8px;
          border-radius: 8px;
          transition: background-color 120ms ease;
        }

        .user-profile-btn:hover {
          background: var(--surface-sunken);
        }

        .user-avatar-circle {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: var(--surface-sunken);
          border: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-secondary);
          flex-shrink: 0;
        }

        .user-details-col {
          flex: 1;
          display: flex;
          flex-direction: column;
          text-align: left;
          min-width: 0;
        }

        .user-name-text {
          font-size: 12.5px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .user-plan-label {
          font-size: 11px;
          color: var(--text-muted);
        }

        .user-chevron {
          color: var(--text-muted);
        }

        /* Popover */
        .user-dropdown-popover {
          position: absolute;
          bottom: calc(100% + 6px);
          left: 10px;
          right: 10px;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 12px;
          box-shadow: var(--shadow-float);
          padding: 10px;
          z-index: 60;
          animation: popover-fade 140ms ease-out;
        }

        .popover-user-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: 4px 6px;
        }

        .popover-name {
          font-size: 12.5px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .popover-email {
          font-size: 11px;
          color: var(--text-muted);
        }

        .popover-divider {
          height: 1px;
          background: var(--border);
          margin: 6px 0;
        }

        .plan-badge-pill {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          font-size: 11px;
          font-weight: 600;
          color: var(--accent);
          background: var(--accent-light);
          padding: 3px 8px;
          border-radius: var(--radius-pill);
        }

        .popover-menu-item {
          width: 100%;
          display: flex;
          align-items: center;
          gap: 7px;
          padding: 6px 8px;
          border-radius: 6px;
          font-size: 12px;
          color: var(--text-primary);
          transition: background-color 120ms ease;
        }

        .popover-menu-item:hover {
          background: var(--surface-sunken);
        }

        .popover-menu-item.text-danger {
          color: var(--contradicted);
        }

        .popover-menu-item.text-danger:hover {
          background: var(--contradicted-bg);
        }

        /* Mobile Drawer */
        @media (max-width: 768px) {
          .halluciguard-sidebar {
            position: fixed;
            left: 0;
            top: 0;
            bottom: 0;
            transform: translateX(-100%);
            box-shadow: var(--shadow-modal);
          }

          .halluciguard-sidebar.is-mobile-open {
            transform: translateX(0);
          }

          .sidebar-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(2px);
            z-index: 45;
          }
        }

        @keyframes popover-fade {
          from {
            opacity: 0;
            transform: translateY(4px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </>
  )
}
