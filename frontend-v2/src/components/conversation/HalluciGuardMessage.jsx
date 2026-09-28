import React from 'react'
import { BrandMark } from '../common/BrandMark'
import { VerificationStatus } from './VerificationStatus'
import { EvidenceTrace } from './EvidenceTrace'
import { ClaimList } from './ClaimList'
import { CorrectionView } from './CorrectionView'
import { AgentActivity } from './AgentActivity'

export function HalluciGuardMessage({ message, onOpenSource }) {
  return (
    <div className="halluciguard-message-row">
      <div className="message-header-row">
        <BrandMark size={18} />
        <span className="agent-name-tag">HalluciGuard</span>
        {message.timestamp && (
          <span className="message-time-tag">· {message.timestamp}</span>
        )}
      </div>

      <div className="message-body">
        {/* Priority 1: Answer text */}
        <div className="answer-text">
          <p>{message.text}</p>
        </div>

        {/* Priority 2: Verification status */}
        {message.status && (
          <VerificationStatus
            status={message.status}
            title={message.statusTitle}
            description={message.statusDescription}
          />
        )}

        {/* Correction before/after if present */}
        {message.correction && (
          <CorrectionView correction={message.correction} />
        )}

        {/* Priority 3: Evidence Trace */}
        {message.sources && message.sources.length > 0 && (
          <EvidenceTrace
            summary={message.evidenceSummary}
            sources={message.sources}
            onOpenSource={onOpenSource}
          />
        )}

        {/* Priority 4: Claim-level verification */}
        {message.claims && message.claims.length > 0 && (
          <ClaimList claims={message.claims} />
        )}

        {/* Priority 5: Agent activity pipeline disclosure */}
        <AgentActivity />
      </div>

      <style>{`
        .halluciguard-message-row {
          display: flex;
          flex-direction: column;
          width: 100%;
          margin: 20px 0;
          animation: message-fade 220ms ease-out;
        }

        .message-header-row {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }

        .agent-name-tag {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          letter-spacing: 0.02em;
        }

        .message-time-tag {
          font-size: 11px;
          color: var(--text-muted);
        }

        .message-body {
          padding-left: 26px;
          color: var(--text-primary);
        }

        .answer-text {
          font-size: 15px;
          line-height: 1.6;
          color: var(--text-primary);
        }

        .answer-text p {
          margin: 0 0 10px;
        }

        @keyframes message-fade {
          from {
            opacity: 0;
            transform: translateY(6px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  )
}
