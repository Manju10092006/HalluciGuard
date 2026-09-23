import React, { useState, useEffect } from 'react'
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
import { INITIAL_RECENT_VERIFICATIONS } from '../../mock/verificationData'
import { VERIFICATION_STATUS } from '../../types/verification'

export function HalluciGuardShell() {

  // Theme & Appearance State
  const [theme, setTheme] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_theme') : null) || 'light')
  const [atmosphere, setAtmosphere] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_atmosphere') : null) || 'mesh')
  const [density, setDensity] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('hg_density') : null) || 'comfortable')

  // User Authentication State
  const [user, setUser] = useState({
    name: 'Judha Maygustya',
    email: 'judha.design@halluciguard.ai',
    plan: 'HalluciGuard Pro',
  })

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
  const [recentSessions, setRecentSessions] = useState(INITIAL_RECENT_VERIFICATIONS)
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [activeSession, setActiveSession] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

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
      flowMeta: template,
    })
  }

  // Handle "Create from chat" from CreateFlowModal
  const handleSelectCreateFromChat = () => {
    handleNewVerification()
  }

  // Submit claim or flow trigger handler
  const handleSendMessage = ({ text, deepVerify, evidenceAttachment, flowMeta }) => {
    const timestamp = 'Just now'
    const userMessage = {
      id: `msg-${Date.now()}`,
      sender: 'user',
      text,
      timestamp,
      attachment: evidenceAttachment,
    }

    // Determine mock verification response based on query heuristics
    const lower = text.toLowerCase()
    let status = VERIFICATION_STATUS.SUPPORTED
    let statusTitle = 'SUPPORTED'
    let statusTone = 'supported'
    let statusDesc = 'All claims are verified by authoritative primary documentation.'
    let replyText = 'This claim is supported by primary records. The retrieved documentation confirms the stated events.'
    let correction = null

    if (flowMeta || lower.includes('flow:') || lower.includes('slack') || lower.includes('linear')) {
      status = VERIFICATION_STATUS.SUPPORTED
      statusTitle = 'FLOW CONFIGURED & VERIFIED'
      statusTone = 'supported'
      statusDesc = 'Flow logic, tool authentication, and evidence triggers validated across integrated services.'
      replyText = `I have instantiated the workflow "${flowMeta?.title || 'Automated Verification Flow'}". HalluciGuard has connected the event listener, configured the payload validator, and scheduled continuous evidence checking.`
    } else if (lower.includes('snehith') || lower.includes('buzz aldrin stepped out first') || lower.includes('contradict')) {
      status = VERIFICATION_STATUS.CONTRADICTED
      statusTitle = 'CONTRADICTED'
      statusTone = 'contradicted'
      statusDesc = 'The statement conflicts directly with authoritative primary source records.'
      if (lower.includes('snehith')) {
        replyText = 'That claim is contradicted by historical archives. Java was created by James Gosling and his team at Sun Microsystems in 1991–1995.'
        correction = {
          original: text,
          corrected: 'Java was created by James Gosling and his team at Sun Microsystems in 1995.',
          verificationNote: 'Verified against Computer History Museum & Oracle Documentation.',
        }
      } else {
        replyText = 'Apollo 11 landed in 1969, but the claim contains a contradiction. Neil Armstrong was the first astronaut to step onto the lunar surface.'
        correction = {
          original: text,
          corrected: 'Neil Armstrong stepped onto the lunar surface first on July 20, 1969, followed by Buzz Aldrin.',
          verificationNote: 'Verified against NASA Apollo 11 Surface Journal transcripts.',
        }
      }
    } else if (lower.includes('gpt-5') || lower.includes('unsupported') || lower.includes('leak') || lower.includes('rumor')) {
      status = VERIFICATION_STATUS.INSUFFICIENT_EVIDENCE
      statusTitle = 'INSUFFICIENT EVIDENCE'
      statusTone = 'insufficient'
      statusDesc = 'No credible independent publications or official evaluation records confirm this claim.'
      replyText = 'There is insufficient verified evidence to confirm this claim. No technical benchmark reports or official releases substantiate it.'
    }

    // New or updated session object
    const newSessionId = activeSessionId || `session-${Date.now()}`
    const sessionTitle = flowMeta ? flowMeta.title : text.length > 32 ? `${text.slice(0, 32)}...` : text

    const initialMessages = activeSession ? [...activeSession.messages, userMessage] : [userMessage]

    const updatedSession = {
      id: newSessionId,
      title: sessionTitle,
      timeGroup: 'Today',
      timestamp: 'Just now',
      status,
      statusBadge: status === VERIFICATION_STATUS.SUPPORTED ? 'Verified' : 'Needs correction',
      statusBadgeTone: statusTone,
      query: text,
      messages: initialMessages,
    }

    setActiveSessionId(newSessionId)
    setActiveSession(updatedSession)
    setIsLoading(true)

    // Simulate multi-agent verification progress
    setTimeout(() => {
      const assistantMessage = {
        id: `msg-${Date.now() + 1}`,
        sender: 'halluciguard',
        text: replyText,
        timestamp: 'Just now',
        status,
        statusTitle,
        statusDescription: statusDesc,
        correction,
        evidenceSummary: {
          totalSources: 3,
          supportingCount: status === VERIFICATION_STATUS.SUPPORTED ? 3 : 1,
          contradictingCount: status === VERIFICATION_STATUS.CONTRADICTED ? 2 : 0,
          contextCount: 1,
        },
        sources: [
          {
            id: 'src-gen-1',
            title: 'Verified Historical & Technical Archive',
            domain: flowMeta ? 'api.integration.io' : 'archive.org',
            relationship: status === VERIFICATION_STATUS.CONTRADICTED ? 'Contradicts claim' : 'Supports claim',
            relationshipTone: status === VERIFICATION_STATUS.CONTRADICTED ? 'contradicted' : 'supported',
            excerpt: flowMeta
              ? 'Webhook contract schema validated with 200 OK. Entailment checks active.'
              : 'Primary corpus record confirming chronological order of events, authorship, and technical specifications.',
            url: 'https://archive.org',
          },
          {
            id: 'src-gen-2',
            title: 'Institutional Technical Documentation',
            domain: flowMeta ? 'linear.app' : 'standards.org',
            relationship: status === VERIFICATION_STATUS.CONTRADICTED ? 'Contradicts claim' : 'Supports claim',
            relationshipTone: status === VERIFICATION_STATUS.CONTRADICTED ? 'contradicted' : 'supported',
            excerpt: flowMeta
              ? 'Trigger listener active on endpoint #bugs. Automated cross-referencing enabled.'
              : 'Official specification and milestone log verified by authoritative consensus.',
            url: 'https://standards.org',
          },
        ],
        claims: [
          {
            number: '01',
            text: text,
            status,
            tone: statusTone,
            verdict: statusDesc,
            evidenceExcerpt: 'Retrieved primary document passage compared with atomic claim statements.',
          },
        ],
      }

      const completedSession = {
        ...updatedSession,
        messages: [...initialMessages, assistantMessage],
      }

      setActiveSession(completedSession)
      setIsLoading(false)

      // Update recent sessions list
      setRecentSessions((prev) => {
        const existing = prev.filter((s) => s.id !== newSessionId)
        return [completedSession, ...existing]
      })
    }, 2400)
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
        onLogout={() => setUser(null)}
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
        onLoginSuccess={(userData) => setUser(userData)}
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
