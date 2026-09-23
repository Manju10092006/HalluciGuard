import React from 'react'
import { VerificationOrb } from './VerificationOrb'
import { VerificationComposer } from '../composer/VerificationComposer'
import { QuickActions } from '../composer/QuickActions'
import { RecentVerificationStrip } from './RecentVerificationStrip'

export function EmptyVerificationState({
  onSendMessage,
  recentSessions,
  onSelectSession,
  disabled = false,
}) {
  const [composerSeed, setComposerSeed] = React.useState('')

  const handleSelectQuickPrompt = (prompt) => {
    setComposerSeed(prompt)
  }

  return (
    <div className="empty-state-wrapper">
      <div className="empty-state-content">
        {/* Soft iridescent verification orb */}
        <VerificationOrb size={64} />

        {/* Editorial serif headline (Manus-style) */}
        <h1 className="empty-headline">
          Don't trust the answer.<br />
          <span className="headline-accent">Trace the evidence.</span>
        </h1>

        {/* Supporting statement */}
        <p className="empty-subheading">
          Verify AI-generated claims against evidence you can inspect.
        </p>

        {/* Centered Composer */}
        <VerificationComposer
          onSendMessage={onSendMessage}
          disabled={disabled}
          initialText={composerSeed}
          isSticky={false}
          placeholder="Paste an AI-generated answer to verify..."
        />

        {/* Quick prompt suggestions */}
        <QuickActions onSelectPrompt={handleSelectQuickPrompt} />

        {/* Secondary recent verifications strip */}
        <RecentVerificationStrip
          items={recentSessions}
          onSelectSession={onSelectSession}
        />
      </div>

      <style>{`
        .empty-state-wrapper {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 40px 24px;
          min-height: calc(100vh - 56px);
        }

        .empty-state-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          width: 100%;
          max-width: var(--composer-max-width);
          margin-top: -30px;
        }

        .empty-headline {
          font-family: var(--font-serif);
          font-size: 58px;
          font-weight: 400;
          letter-spacing: -0.015em;
          line-height: 1.04;
          color: var(--text-primary);
          margin: 22px 0 14px;
        }

        .headline-accent {
          color: var(--text-primary);
          font-style: italic;
        }

        .empty-subheading {
          font-size: 15px;
          font-weight: 400;
          color: var(--text-secondary);
          margin-bottom: 30px;
          max-width: 520px;
          line-height: 1.55;
          letter-spacing: -0.005em;
        }

        @media (max-width: 640px) {
          .empty-headline {
            font-size: 40px;
          }
          .empty-subheading {
            font-size: 14px;
          }
        }
      `}</style>
    </div>
  )
}
