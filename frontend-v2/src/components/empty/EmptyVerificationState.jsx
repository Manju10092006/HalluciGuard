import React, { useState } from 'react'

export function EmptyVerificationState({
  onSendMessage,
  recentSessions = [],
  onSelectSession,
  disabled = false,
  onToggleSidebar,
  onOpenSettings,
  onOpenCreateFlow,
  onOpenAuth,
  user
}) {
  const [inputText, setInputText] = useState('')
  const [deepVerify, setDeepVerify] = useState(false)

  const handleSend = () => {
    if (!inputText.trim() || disabled) return
    onSendMessage({
      text: inputText.trim(),
      deepVerify,
    })
    setInputText('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSelectTemplate = (promptText) => {
    setInputText(promptText)
    onSendMessage({
      text: promptText,
      deepVerify,
    })
  }

  const userName = user?.name ? user.name.split(' ')[0] : 'Liam'

  return (
    <div className="empty-verification-viewport">
      {/* Architectural Sunlight & Window Blinds Motion Blur Atmosphere */}
      <div className="shadow-atmosphere">
        <svg className="sunlight-svg" viewBox="0 0 1440 900" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
          <defs>
            <filter id="directionalMotionBlur" x="-30%" y="-30%" width="160%" height="160%" filterUnits="objectBoundingBox">
              <feGaussianBlur stdDeviation="55 18" />
            </filter>
            <filter id="softFoliageMotionBlur" x="-40%" y="-40%" width="180%" height="180%" filterUnits="objectBoundingBox">
              <feGaussianBlur stdDeviation="35 12" />
            </filter>
            <radialGradient id="ambientSunlight" cx="50%" cy="40%" r="85%">
              <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.9" />
              <stop offset="50%" stopColor="#FFFFFF" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#F6F6F4" stopOpacity="0.2" />
            </radialGradient>
          </defs>

          {/* Full Viewport Ambient Sunlight Pool */}
          <rect width="1440" height="900" fill="url(#ambientSunlight)" filter="url(#directionalMotionBlur)" />

          {/* Architectural Window Blind Shadows (Directional Motion Blur Slats) */}
          <g opacity="0.07" filter="url(#directionalMotionBlur)">
            <rect x="420" y="-120" width="85" height="1300" transform="rotate(34 420 -120)" fill="#111111" />
            <rect x="570" y="-120" width="80" height="1300" transform="rotate(34 570 -120)" fill="#111111" />
            <rect x="720" y="-120" width="90" height="1300" transform="rotate(34 720 -120)" fill="#111111" />
            <rect x="870" y="-120" width="75" height="1300" transform="rotate(34 870 -120)" fill="#111111" />
            <rect x="1020" y="-120" width="95" height="1300" transform="rotate(34 1020 -120)" fill="#111111" />
            <rect x="1170" y="-120" width="85" height="1300" transform="rotate(34 1170 -120)" fill="#111111" />
            <rect x="1320" y="-120" width="105" height="1300" transform="rotate(34 1320 -120)" fill="#111111" />
          </g>

          {/* Organic Tree & Window Structure Motion Blurred Shadows */}
          <g opacity="0.08" filter="url(#softFoliageMotionBlur)">
            <path d="M1080,40 Q1240,110 1420,70 Q1310,260 1460,420 Q1190,320 1080,40 Z" fill="#050505" />
            <path d="M1220,-60 Q1340,140 1490,190 Q1370,360 1495,570 Q1240,420 1220,-60 Z" fill="#050505" />
            <ellipse cx="1340" cy="190" rx="110" ry="80" transform="rotate(-15 1340 190)" fill="#050505" />
            <ellipse cx="1420" cy="330" rx="130" ry="95" transform="rotate(-20 1420 330)" fill="#050505" />
          </g>
        </svg>
      </div>

      {/* Main Viewport Container */}
      <div className="app-viewport">
        {/* Top Left Micro Controls */}
        <div className="top-controls">
          <button
            className="icon-btn"
            title="Toggle Sidebar"
            aria-label="Toggle Sidebar"
            onClick={onToggleSidebar}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <rect x="3" y="3" width="18" height="18" rx="3" />
              <path d="M9 3v18" />
            </svg>
          </button>

          <button
            className="icon-btn"
            title="Settings"
            aria-label="Settings"
            onClick={onOpenSettings}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>

          <button
            className="icon-btn"
            title="Open External / Create Flow"
            aria-label="Open External / Create Flow"
            onClick={onOpenCreateFlow}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
          </button>
        </div>

        {/* Center Workspace */}
        <main className="workspace-center">
          <div className="upgrade-pill" onClick={onOpenAuth}>
            <svg className="flower-icon" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="4" fill="#E9B51D" />
              <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" stroke="#E9B51D" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <span>Upgrade Plan</span>
          </div>

          <h1 className="greeting-heading">Good morning, {userName}</h1>
          <p className="greeting-subtitle">Welcome to HalluciGuard, by The General Intelligence Company</p>

          <div className="composer-card">
            <textarea
              className="composer-input"
              rows={2}
              placeholder="my name is Connie what can you do?"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
            />

            <div className="composer-bottom">
              <div className="composer-tools">
                <button className="tool-icon-btn" title="Attach file" aria-label="Attach file">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                  </svg>
                </button>

                <div
                  className={`mini-toggle ${deepVerify ? 'active' : ''}`}
                  title="Toggle Deep Verification Mode"
                  onClick={() => setDeepVerify(!deepVerify)}
                />

                <button
                  className="tool-icon-btn"
                  title="Enhance prompt"
                  aria-label="Enhance prompt"
                  onClick={() => {
                    if (!inputText) setInputText('Verify claim: Java was created by James Gosling at Sun Microsystems in 1995.')
                  }}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                  </svg>
                </button>
              </div>

              <button className="send-btn" title="Send query" aria-label="Send query" onClick={handleSend} disabled={disabled}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <line x1="12" y1="19" x2="12" y2="5" />
                  <polyline points="5 12 12 5 19 12" />
                </svg>
              </button>
            </div>
          </div>
        </main>

        {/* Bottom Section & Template Cards Rail */}
        <footer className="bottom-section">
          <div className="template-rail-wrapper">
            <div className="template-rail">
              <div
                className="template-card"
                onClick={() => handleSelectTemplate('Comprehensive analysis, benchmarks, and market strategy...')}
              >
                <div className="card-header">
                  <span className="card-category category-cofounder">(HalluciGuard)</span>
                  <span className="card-watch">Watch ↗</span>
                </div>
                <div className="card-title">Deep dive: build an analysis</div>
                <div className="card-desc">Comprehensive analysis, benchmarks, and market strategy...</div>
              </div>

              <div
                className="template-card"
                onClick={() => handleSelectTemplate("What's going on in engineering team status, priorities, and challenges.")}
              >
                <div className="card-header">
                  <span className="card-category category-engineering">Sync with engineering</span>
                  <span className="card-watch">Watch ↗</span>
                </div>
                <div className="card-title">What's going on in engineering</div>
                <div className="card-desc">Get a snapshot of your engineering team's status, priorities, and challenges.</div>
              </div>

              <div
                className="template-card"
                onClick={() => handleSelectTemplate('Make me a resume based on known data and online research...')}
              >
                <div className="card-header">
                  <span className="card-category category-build">Build a resume</span>
                  <span className="card-watch">Watch ↗</span>
                </div>
                <div className="card-title">Make me a resume based on what you...</div>
                <div className="card-desc">Create a public-ready PDF resume from known data and online research, omitting all personal...</div>
              </div>

              <div
                className="template-card"
                onClick={() => handleSelectTemplate('Make this into retro pixel art with adjustable pixel size.')}
              >
                <div className="card-header">
                  <span className="card-category category-design">Design</span>
                  <span className="card-watch">Watch ↗</span>
                </div>
                <div className="card-title">Make this into pixel art</div>
                <div className="card-desc">Convert any image into retro pixel art with adjustable pixel size.</div>
              </div>

              <div
                className="template-card"
                onClick={() => handleSelectTemplate('Extract key insights from meeting transcripts and organize them in structured docs.')}
              >
                <div className="card-header">
                  <span className="card-category category-knowledge">Coalesce knowledge</span>
                  <span className="card-watch">Watch ↗</span>
                </div>
                <div className="card-title">Add these meeting notes...</div>
                <div className="card-desc">Extract key insights from meeting transcripts and organize them in structured docs.</div>
              </div>
            </div>
          </div>

          <div className="templates-handle" onClick={onOpenCreateFlow}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <polyline points="6 9 12 15 18 9" />
            </svg>
            <span>Templates</span>
          </div>
        </footer>
      </div>

      <style>{`
        .empty-verification-viewport {
          position: relative;
          width: 100%;
          height: 100%;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background: #F6F6F4 !important;
          color: #181818;
          overflow: hidden;
        }

        .shadow-atmosphere {
          position: fixed;
          top: 0;
          left: 0;
          width: 100vw;
          height: 100vh;
          pointer-events: none;
          z-index: 0;
          overflow: hidden;
          background: #F6F6F4 !important;
          filter: blur(8px);
        }

        .sunlight-svg {
          width: 100%;
          height: 100%;
          opacity: 0.95;
          animation: motionBlurDrift 28s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate;
          will-change: transform, filter;
        }

        @keyframes motionBlurDrift {
          0% {
            transform: translate(0px, 0px) rotate(0deg) scale(1);
            filter: drop-shadow(-10px 5px 25px rgba(0,0,0,0.04));
          }
          50% {
            transform: translate(-25px, 15px) rotate(0.8deg) scale(1.025);
            filter: drop-shadow(-20px 10px 35px rgba(0,0,0,0.06));
          }
          100% {
            transform: translate(-45px, 28px) rotate(1.5deg) scale(1.04);
            filter: drop-shadow(-35px 18px 45px rgba(0,0,0,0.07));
          }
        }

        .app-viewport {
          position: relative;
          z-index: 1;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          align-items: center;
          padding: 24px 20px 16px 20px;
          background: transparent !important;
        }

        .top-controls {
          position: absolute;
          top: 22px;
          left: 24px;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .icon-btn {
          background: none;
          border: none;
          cursor: pointer;
          color: #555555;
          opacity: 0.75;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: opacity 0.2s ease, color 0.2s ease;
          padding: 2px;
        }

        .icon-btn:hover {
          opacity: 1;
          color: #111111;
        }

        .icon-btn svg {
          width: 16px;
          height: 16px;
          stroke-width: 1.4px;
        }

        .workspace-center {
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 100%;
          max-width: 720px;
          margin-top: auto;
          margin-bottom: auto;
          transform: translateY(-12px);
          background: transparent !important;
        }

        .upgrade-pill {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          height: 31px;
          padding: 0 14px;
          border-radius: 9999px;
          background: rgba(255, 255, 255, 0.82);
          border: 1px solid rgba(0, 0, 0, 0.06);
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
          font-size: 11px;
          font-weight: 500;
          color: #333333;
          letter-spacing: -0.01em;
          cursor: pointer;
          margin-bottom: 24px;
          backdrop-filter: blur(8px);
          transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1), background-color 0.2s ease;
        }

        .upgrade-pill:hover {
          background: rgba(255, 255, 255, 0.95);
          transform: translateY(-1px);
        }

        .flower-icon {
          width: 13px;
          height: 13px;
          display: inline-block;
          flex-shrink: 0;
        }

        .greeting-heading {
          font-size: 51px;
          font-weight: 400;
          letter-spacing: -0.04em;
          color: #181818;
          text-align: center;
          line-height: 1.08;
          margin-bottom: 14px;
        }

        .greeting-subtitle {
          font-size: 12.5px;
          font-weight: 400;
          color: #5E5E5E;
          letter-spacing: -0.012em;
          text-align: center;
          margin-bottom: 48px;
        }

        .composer-card {
          width: min(670px, 100%);
          height: 130px;
          background: rgba(255, 255, 255, 0.94);
          border: 1px solid rgba(0, 0, 0, 0.055);
          border-radius: 9px;
          box-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.03),
            0 10px 28px rgba(0, 0, 0, 0.035);
          padding: 18px 20px 14px 20px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          backdrop-filter: blur(12px);
          transition: border-color 0.25s ease, box-shadow 0.25s ease;
          position: relative;
        }

        .composer-card:focus-within, .composer-card:hover {
          border-color: rgba(0, 0, 0, 0.1);
          box-shadow: 
            0 4px 8px rgba(0, 0, 0, 0.04),
            0 14px 32px rgba(0, 0, 0, 0.05);
        }

        .composer-input {
          width: 100%;
          border: none;
          outline: none;
          background: transparent;
          font-family: inherit;
          font-size: 13.5px;
          font-weight: 400;
          color: #1a1a1a;
          letter-spacing: -0.01em;
          resize: none;
          line-height: 1.4;
        }

        .composer-input::placeholder {
          color: #222222;
        }

        .composer-bottom {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: auto;
        }

        .composer-tools {
          display: flex;
          align-items: center;
          gap: 14px;
        }

        .tool-icon-btn {
          background: none;
          border: none;
          cursor: pointer;
          color: #777777;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0;
          transition: color 0.15s ease;
        }

        .tool-icon-btn:hover {
          color: #222222;
        }

        .tool-icon-btn svg {
          width: 16px;
          height: 16px;
          stroke-width: 1.5px;
        }

        .mini-toggle {
          width: 28px;
          height: 16px;
          background-color: #E2E2DF;
          border-radius: 999px;
          position: relative;
          cursor: pointer;
          transition: background-color 0.2s ease;
        }

        .mini-toggle::after {
          content: '';
          position: absolute;
          top: 2px;
          left: 2px;
          width: 12px;
          height: 12px;
          background-color: #FFFFFF;
          border-radius: 50%;
          box-shadow: 0 1px 2px rgba(0,0,0,0.15);
          transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1);
        }

        .mini-toggle.active {
          background-color: #111111;
        }

        .mini-toggle.active::after {
          transform: translateX(12px);
        }

        .send-btn {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          background-color: #111111;
          border: none;
          color: #FFFFFF;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1), background-color 0.15s ease;
        }

        .send-btn:hover {
          transform: scale(1.04);
          background-color: #000000;
        }

        .send-btn svg {
          width: 16px;
          height: 16px;
          stroke-width: 2.2px;
        }

        .bottom-section {
          width: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
          margin-top: auto;
          padding-bottom: 4px;
          background: transparent !important;
        }

        .template-rail-wrapper {
          width: 100vw;
          overflow-x: auto;
          padding: 4px 24px;
          display: flex;
          justify-content: center;
          scrollbar-width: none;
          background: transparent !important;
        }

        .template-rail-wrapper::-webkit-scrollbar {
          display: none;
        }

        .template-rail {
          display: flex;
          gap: 12px;
          max-width: 1240px;
          background: transparent !important;
        }

        .template-card {
          width: 260px;
          height: 112px;
          background: rgba(255, 255, 255, 0.35);
          border: 1px solid rgba(0, 0, 0, 0.055);
          border-radius: 8px;
          padding: 12px 14px;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          backdrop-filter: blur(8px);
          cursor: pointer;
          flex-shrink: 0;
          transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1), background-color 0.25s ease, border-color 0.25s ease;
        }

        .template-card:hover {
          transform: translateY(-2px);
          background: rgba(255, 255, 255, 0.65);
          border-color: rgba(0, 0, 0, 0.09);
        }

        .card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: 100%;
        }

        .card-category {
          font-size: 10.5px;
          font-weight: 500;
          letter-spacing: -0.01em;
        }

        .category-engineering { color: #8B5CF6; }
        .category-build { color: #3B82F6; }
        .category-design { color: #10B981; }
        .category-knowledge { color: #F43F5E; }
        .category-cofounder { color: #64748B; }

        .card-watch {
          font-size: 10px;
          font-weight: 400;
          color: #333333;
          display: flex;
          align-items: center;
          gap: 2px;
          opacity: 0.85;
        }

        .card-title {
          font-size: 12.5px;
          font-weight: 500;
          color: #222222;
          letter-spacing: -0.015em;
          margin-top: 4px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .card-desc {
          font-size: 10.5px;
          line-height: 1.42;
          color: #777777;
          letter-spacing: -0.005em;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        .templates-handle {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 11px;
          font-weight: 500;
          color: #555555;
          cursor: pointer;
          opacity: 0.85;
          padding: 4px 8px;
          border-radius: 4px;
          transition: opacity 0.15s ease, background-color 0.15s ease;
        }

        .templates-handle:hover {
          opacity: 1;
          background: rgba(0, 0, 0, 0.03);
        }

        .templates-handle svg {
          width: 12px;
          height: 12px;
          stroke-width: 1.8px;
        }

        @media (max-width: 640px) {
          .greeting-heading {
            font-size: 36px;
          }
          .composer-card {
            height: 120px;
          }
          .template-rail-wrapper {
            justify-content: flex-start;
          }
        }
      `}</style>
    </div>
  )
}
