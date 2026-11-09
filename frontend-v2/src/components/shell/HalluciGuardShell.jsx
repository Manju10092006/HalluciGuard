import React, { useState, useEffect, useRef } from 'react'
import { HalluciGuardSidebar } from './HalluciGuardSidebar'
import { ChatHeader } from './ChatHeader'
import { EmptyVerificationState } from '../empty/EmptyVerificationState'
import { Conversation } from '../conversation/Conversation'
import { VerificationComposer } from '../composer/VerificationComposer'
import { AppearanceModal } from '../modals/AppearanceModal'
import { SettingsModal } from '../modals/SettingsModal'
import { SourceDrawer } from '../modals/SourceDrawer'
import { CreateFlowModal } from '../modals/CreateFlowModal'
import { AuthDialog } from '../auth/AuthDialog'
import { useAuth } from '../../lib/auth/AuthContext'
import { verifyMessage } from '../../lib/verification'
import { toast } from 'sonner'

import { VERIFICATION_STATUS } from '../../types/verification'

export function HalluciGuardShell() {

  // Theme & Appearance State
  const [theme, setTheme] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_theme') : null) || 'light')
  const [atmosphere, setAtmosphere] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_atmosphere') : null) || 'mesh')
  const [density, setDensity] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_density') : null) || 'comfortable')

  // User Authentication State
  const { user: account, token, status: authStatus, signOut } = useAuth()
  const user = account ? { ...account, plan: 'HalluciGuard' } : null

  // Modals & Drawers
  const [isAppearanceOpen, setIsAppearanceOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isCreateFlowOpen, setIsCreateFlowOpen] = useState(false)
  const [isAuthOpen, setIsAuthOpen] = useState(false)
  const [activeSource, setActiveSource] = useState(null)
  const [isSourceDrawerOpen, setIsSourceDrawerOpen] = useState(false)

  // Sidebar Layout State
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  const [activeNav, setActiveNav] = useState('home')

  // Verification Sessions & Active Conversation
  const [recentSessions, setRecentSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [activeSession, setActiveSession] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const [verificationError, setVerificationError] = useState(null)
  const storageKey = `hg.sessions.v2.${account?.id || 'guest'}`
  const accountScope = useRef(storageKey)
  accountScope.current = storageKey
  useEffect(() => {
    if (authStatus === 'loading') return
    try { const saved = JSON.parse(localStorage.getItem(storageKey) || '[]'); setRecentSessions(Array.isArray(saved) ? saved : []); }
    catch { setRecentSessions([]) }
    setActiveSession(null)
    setActiveSessionId(null)
    setIsLoading(false)
    setVerificationError(null)
  }, [storageKey, authStatus])

  // Apply Theme & Atmosphere
  useEffect(() => {
    localStorage.setItem('hg_theme', theme)
    const root = document.documentElement
    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      root.setAttribute('data-theme', prefersDark ? 'dark' : 'light')
    } else {
      root.setAttribute('data-theme', theme)
    }
  }, [theme])

  useEffect(() => {
    localStorage.setItem('hg_atmosphere', atmosphere)
    if (atmosphere === 'mesh') {
      document.body.classList.add('wallpaper-mesh')
    } else {
      document.body.classList.remove('wallpaper-mesh')
    }
  }, [atmosphere])

  useEffect(() => {
    localStorage.setItem('hg_density', density)
  }, [density])

  // Select an existing recent session
  const handleSelectSession = (id) => {
    const found = recentSessions.find((s) => s.id === id)
    if (found) {
      setActiveSessionId(id)
      setActiveSession(found)
      setActiveNav('verify')
    }
  }

  // Start a new verification (reset to empty state)
  const handleNewVerification = () => {
    setActiveSessionId(null)
    setActiveSession(null)
    setActiveNav('home')
  }

  // Open primary source in slide-out drawer
  const handleOpenSource = (source) => {
    setActiveSource(source)
    setIsSourceDrawerOpen(true)
  }

  // Handle template selection from CreateFlowModal (Image 1)
  const handleSelectFlowTemplate = (template) => {
    handleSendMessage({
      text: template.prompt || `Run flow: ${template.title}`,
      deepVerify: true,

    })
  }

  // Handle "Create from chat" from CreateFlowModal
  const handleSelectCreateFromChat = () => {
    handleNewVerification()
  }

  // Every verdict comes from FastAPI; outages remain visible errors.
  const handleSendMessage = async ({ text, evidenceAttachment, flowMeta }) => {
    if (isLoading || !text?.trim()) return
    if (evidenceAttachment) { toast.error('File evidence is not supported yet. Paste the claim as text.'); return }
    if (flowMeta) { toast.error('External workflow integrations are not connected.'); return }
    setVerificationError(null)
    const sessionId = activeSessionId || crypto.randomUUID()
    const previous = activeSession?.messages || []
    const userMessage = { id: crypto.randomUUID(), sender: 'user', text, timestamp: 'Just now' }
    const session = { id: sessionId, title: activeSession?.title || text.slice(0, 60), timeGroup: 'Today', messages: [...previous, userMessage] }
    setActiveSessionId(sessionId)
    setActiveSession(session)
    setIsLoading(true)
    try {
      const reply = await verifyMessage(text, { token, history: previous })
      if (accountScope.current !== storageKey) return
      const completed = { ...session, statusBadge: reply.statusTitle, statusBadgeTone: reply.status === 'SUPPORTED' ? 'supported' : 'insufficient', messages: [...session.messages, reply] }
      setRecentSessions(current => {
        const sessions = [completed, ...current.filter(s => s.id !== sessionId)].slice(0, 50)
        try { localStorage.setItem(storageKey, JSON.stringify(sessions)) }
        catch { toast.error('This browser could not save the conversation.') }
        return sessions
      })
      setActiveSession(current => current?.id === sessionId ? completed : current)
    } catch (error) {
      if (accountScope.current !== storageKey) return
      setVerificationError(error.message || 'The verification service is unavailable.')
      toast.error(error.message || 'Verification failed')
    } finally { if (accountScope.current === storageKey) setIsLoading(false) }
  }

  const isHome = !activeSession

  return (
    <div className="halluciguard-app-shell">
      {/* Persistent Left Sidebar */}
      <HalluciGuardSidebar
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        activeNav={activeNav}
        setActiveNav={setActiveNav}
        onNewVerification={handleNewVerification}
        onOpenCreateFlow={() => setIsCreateFlowOpen(true)}
        onOpenAuth={() => setIsAuthOpen(true)}
        user={user}
        onLogout={signOut}
        recentSessions={recentSessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        isMobileOpen={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Chat Workspace */}
      <div className="chat-workspace-area">
        {/* Minimal Top Bar (Active Conversation Only) */}
        {!isHome && (
          <ChatHeader
            title={activeSession?.title}
            statusBadge={activeSession?.statusBadge}
            statusTone={activeSession?.statusBadgeTone}
            onToggleSidebar={() => {
              if (typeof window !== 'undefined' && window.innerWidth <= 768) {
                setIsMobileSidebarOpen(true)
              } else {
                setSidebarCollapsed(!sidebarCollapsed)
              }
            }}
            sidebarCollapsed={sidebarCollapsed}
            onOpenAppearance={() => setIsAppearanceOpen(true)}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onOpenCreateFlow={() => setIsCreateFlowOpen(true)}
            onOpenAuth={() => setIsAuthOpen(true)}
            user={user}
            isHome={isHome}
          />
        )}

        {/* Workspace Central Stage */}
        <main className="chat-main-viewport">
          {isHome ? (
            <EmptyVerificationState
              onSendMessage={handleSendMessage}
              recentSessions={recentSessions}
              onSelectSession={handleSelectSession}
              disabled={isLoading}
              onToggleSidebar={() => {
                if (typeof window !== 'undefined' && window.innerWidth <= 768) {
                  setIsMobileSidebarOpen(!isMobileSidebarOpen)
                } else {
                  setSidebarCollapsed(!sidebarCollapsed)
                }
              }}
              onOpenSettings={() => setIsSettingsOpen(true)}
              onOpenCreateFlow={() => setIsCreateFlowOpen(true)}
              onOpenAuth={() => setIsAuthOpen(true)}
              user={user}
            />
          ) : (
            <div className="active-conversation-layout">
              <Conversation
                messages={activeSession.messages}
                isLoading={isLoading}
                onOpenSource={handleOpenSource}
              />
              {verificationError && <div role="alert" style={{ padding: '12px 24px', color: 'var(--contradicted)' }}>{verificationError} Your message is above; submit it again to retry.</div>}
              <VerificationComposer
                onSendMessage={handleSendMessage}
                disabled={isLoading}
                isSticky={true}
                placeholder="Ask HalluciGuard to verify an answer or run a flow..."
              />
            </div>
          )}
        </main>
      </div>

      {/* Create New Flow Modal (matching Image 1) */}
      <CreateFlowModal
        isOpen={isCreateFlowOpen}
        onClose={() => setIsCreateFlowOpen(false)}
        onSelectTemplate={handleSelectFlowTemplate}
        onSelectCreateFromChat={handleSelectCreateFromChat}
      />

      {/* Auth Modal with purple gradient border (matching Image 2) */}
      <AuthDialog open={isAuthOpen} onClose={() => setIsAuthOpen(false)} />

      {/* Appearance Modal */}
      <AppearanceModal
        isOpen={isAppearanceOpen}
        onClose={() => setIsAppearanceOpen(false)}
        theme={theme}
        setTheme={setTheme}
        atmosphere={atmosphere}
        setAtmosphere={setAtmosphere}
        density={density}
        setDensity={setDensity}
      />

      {/* Verification Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />

      {/* Source Inspection Drawer */}
      <SourceDrawer
        source={activeSource}
        isOpen={isSourceDrawerOpen}
        onClose={() => setIsSourceDrawerOpen(false)}
      />

      <style>{`
        .halluciguard-app-shell {
          display: flex;
          width: 100vw;
          height: 100vh;
          overflow: hidden;
          background: var(--bg);
        }

        .chat-workspace-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          height: 100%;
          min-width: 0;
          overflow: hidden;
          position: relative;
        }

        .chat-main-viewport {
          flex: 1;
          display: flex;
          flex-direction: column;
          height: 100vh;
          overflow: hidden;
          position: relative;
        }

        .active-conversation-layout {
          flex: 1;
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
      `}</style>
    </div>
  )
}
