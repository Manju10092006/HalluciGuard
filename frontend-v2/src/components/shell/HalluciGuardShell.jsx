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
import { AuthModal } from '../modals/AuthModal'
import { api } from '../../lib/api'
import { useAuth } from '../../lib/auth/AuthContext'

export function HalluciGuardShell() {

  // Theme & Appearance State
  const [theme, setTheme] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_theme') : null) || 'light')
  const [atmosphere, setAtmosphere] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_atmosphere') : null) || 'mesh')
  const [density, setDensity] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_density') : null) || 'comfortable')

  // User Authentication State
  const { user, status: authStatus, signOut } = useAuth()

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
  const [error, setError] = useState(null)
  const pendingRef = useRef(null)
  useEffect(() => {
    let active = true
    setActiveSession(null)
    setActiveSessionId(null)
    setRecentSessions([])
    setActiveSource(null)
    setIsSourceDrawerOpen(false)
    setError(null)
    setIsLoading(false)
    pendingRef.current?.abort()
    pendingRef.current = null
    if (authStatus !== 'loading') {
      api.listConversations().then(convs => { if (active) setRecentSessions(convs.map(toSession)) })
    }
    return () => { active = false; pendingRef.current?.abort() }
  }, [authStatus, user?.id])

  function toSession(conv) {
    return { ...conv, timeGroup: 'Today', timestamp: new Date(conv.updated_at).toLocaleString(), statusBadge: conv.status === 'verified' ? 'Verified' : conv.status === 'reviewing' ? 'Pending' : conv.status === 'failed' ? 'Failed' : 'Withheld', statusBadgeTone: conv.status === 'verified' ? 'supported' : 'insufficient', messages: (conv.messages || []).map(m => {
      if (m.role === 'user') return { id: m.id, sender: 'user', text: m.content }
      const v = m.verification
      return { id: m.id, sender: 'halluciguard', text: m.content, status: v.verdict.toUpperCase(), statusTitle: v.verdict.toUpperCase(), statusDescription: v.explanation, correction: v.correction && v.original_claim ? { original: v.original_claim, corrected: v.correction } : null, sources: v.evidence.map(e => ({ ...e, relationshipTone: e.relationship === 'supports' ? 'supported' : e.relationship === 'contradicts' ? 'contradicted' : 'insufficient' })), evidenceSummary: { totalSources: v.evidence.length, supportingCount: v.counts.supporting, contradictingCount: v.counts.contradicting }, claims: v.claims.map((c,i) => ({ ...c, number: String(i+1).padStart(2,'0'), status: c.status.toUpperCase(), tone: c.status, verdict: c.status })), pipeline: v.pipeline }
    }) }
  }

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
    if (found && authStatus !== 'loading') {
      pendingRef.current?.abort()
      pendingRef.current = null
      setIsLoading(false)
      setError(null)
      setActiveSource(null)
      setIsSourceDrawerOpen(false)
      setActiveSessionId(id)
      setActiveSession(found)
      setActiveNav('verify')
    }
  }

  // Start a new verification (reset to empty state)
  const handleNewVerification = () => {
    pendingRef.current?.abort()
    pendingRef.current = null
    setIsLoading(false)
    setError(null)
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
      flowMeta: template,
    })
  }

  // Handle "Create from chat" from CreateFlowModal
  const handleSelectCreateFromChat = () => {
    handleNewVerification()
  }

  // Only the real local backend can establish a verification result.
  const handleSendMessage = async ({ text, evidenceAttachment, flowMeta }) => {
    if (authStatus === 'loading' || pendingRef.current || !text?.trim()) return
    if (evidenceAttachment || flowMeta) { setError('Attachments and automated flows are not supported by the local verification API. Submit a text question.'); return }
    const controller = new AbortController()
    pendingRef.current = controller
    setError(null)
    setIsLoading(true)
    try {
      const conv = activeSessionId ? await api.getConversation(activeSessionId) : await api.createConversation({ title: text.slice(0,60) })
        if (controller.signal.aborted || pendingRef.current !== controller) return
      setActiveSessionId(conv.id)
      setActiveSession({ ...toSession(conv), messages: [...toSession(conv).messages, { id: 'pending', sender: 'user', text }], statusBadge: 'Pending' })
      const res = await api.sendMessage(conv.id, text, 'standard', controller.signal)
        if (controller.signal.aborted || pendingRef.current !== controller) return
      const completed = toSession(res.conversation)
      setActiveSession(completed)
      setRecentSessions(prev => [completed, ...prev.filter(c => c.id !== conv.id)])
    } catch (err) {
      if (!controller.signal.aborted) {
        setActiveSession(prev => prev ? { ...prev, statusBadge: 'Failed', statusBadgeTone: 'insufficient' } : prev)
        setError('Verification could not complete. No answer was verified. ' + (err.response?.data?.detail || err.message || 'Service unavailable.'))
      }
    } finally {
      if (pendingRef.current === controller) { pendingRef.current = null; setIsLoading(false) }
    }
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
          {error && <div role="alert" data-testid="verification-error" style={{padding:16}}>{error}</div>}
            {isLoading && <button type="button" onClick={() => { pendingRef.current?.abort(); pendingRef.current = null; setIsLoading(false); setActiveSession(prev => prev ? { ...prev, statusBadge: 'Withheld', statusBadgeTone: 'insufficient' } : prev); setError("Verification stopped. No result was verified.") }}>Stop verification</button>}
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
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onLoginSuccess={() => setIsAuthOpen(false)}
      />

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
