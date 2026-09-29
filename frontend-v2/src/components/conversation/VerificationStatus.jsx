import React from 'react'
import { ShieldCheck, AlertOctagon, HelpCircle, AlertTriangle } from 'lucide-react'
import { VERIFICATION_STATUS } from '../../types/verification'
import { ShineBorder } from '@/components/ui/shine-border'
import { NumberTicker } from '@/components/ui/number-ticker'
import { GaugeChart } from '@/components/ui/charts'

/**
 * Verdict surface — Magic UI "Shine Border" (zero-dep CSS) gives the active
 * verdict pill its single slow sheen along the border ring.
 *
 * Optional `confidence` (0–100): when the backend supplies a real confidence
 * value it renders a Magic UI Number Ticker + Animata Gauge Chart summary.
 * It is NEVER invented here — absent data means no instrument is shown.
 */
export function VerificationStatus({ status, title, description, confidence }) {
  const getStatusConfig = () => {
    switch (status) {
      case VERIFICATION_STATUS.SUPPORTED:
      case 'VERIFIED':
        return {
          icon: ShieldCheck,
          className: 'is-supported',
          defaultTitle: 'SUPPORTED',
          defaultDesc: 'The claim is fully supported by primary documentation.',
        }
      case VERIFICATION_STATUS.CONTRADICTED:
      case 'NEEDS_CORRECTION':
        return {
          icon: AlertOctagon,
          className: 'is-contradicted',
          defaultTitle: 'CONTRADICTED',
          defaultDesc: 'The claim conflicts with retrieved verified evidence.',
        }
      case VERIFICATION_STATUS.INSUFFICIENT_EVIDENCE:
        return {
          icon: HelpCircle,
          className: 'is-insufficient',
          defaultTitle: 'INSUFFICIENT EVIDENCE',
          defaultDesc: 'Retrieved records lack sufficient proof to confirm or refute.',
        }
      case VERIFICATION_STATUS.UNCERTAIN:
      default:
        return {
          icon: AlertTriangle,
          className: 'is-uncertain',
          defaultTitle: 'UNCERTAIN',
          defaultDesc: 'Available records present unresolved variance.',
        }
    }
  }

  const config = getStatusConfig()
  const Icon = config.icon
  const hasConfidence =
    typeof confidence === 'number' && Number.isFinite(confidence)

  return (
    <ShineBorder
      className={`verification-status-surface ${config.className}`}
      duration={14}
    >
      <div className="status-layout">
        <div className="status-main">
          <div className="status-header-row">
            <Icon size={16} className="status-indicator-icon" />
            <span className="status-label">{title || config.defaultTitle}</span>
          </div>
          <p className="status-explanation">{description || config.defaultDesc}</p>
          {hasConfidence && (
            <div className="confidence-row">
              <span className="confidence-label">Confidence</span>
              <NumberTicker
                value={confidence}
                suffix="%"
                className="confidence-value"
              />
            </div>
          )}
        </div>
        {hasConfidence && (
          <div className="gauge-wrap">
            <GaugeChart value={confidence} size={92} strokeWidth={8} />
          </div>
        )}
      </div>

      <style>{`
        .verification-status-surface {
          border-radius: var(--radius-inline);
          padding: 12px 16px;
          margin: 14px 0 16px;
          transition: all 140ms ease;
        }

        .status-layout {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .status-main {
          flex: 1;
          min-width: 0;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .status-header-row {
          display: flex;
          align-items: center;
          gap: 7px;
        }

        .status-label {
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        .status-explanation {
          font-size: 13px;
          line-height: 1.45;
          margin: 0;
          opacity: 0.9;
        }

        .confidence-row {
          display: flex;
          align-items: baseline;
          gap: 8px;
          margin-top: 6px;
        }

        .confidence-label {
          font-family: var(--font-mono);
          font-size: 10.5px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          opacity: 0.75;
        }

        .confidence-value {
          font-family: var(--font-mono);
          font-size: 20px;
          font-weight: 700;
          letter-spacing: -0.02em;
        }

        .gauge-wrap {
          flex-shrink: 0;
          color: currentColor;
        }

        /* Supported */
        .verification-status-surface.is-supported {
          background: var(--supported-bg);
          border: 1px solid var(--supported-border);
          color: var(--supported);
        }
        .verification-status-surface.is-supported .status-explanation {
          color: var(--text-primary);
        }

        /* Contradicted */
        .verification-status-surface.is-contradicted {
          background: var(--contradicted-bg);
          border: 1px solid var(--contradicted-border);
          color: var(--contradicted);
        }
        .verification-status-surface.is-contradicted .status-explanation {
          color: var(--text-primary);
        }

        /* Insufficient */
        .verification-status-surface.is-insufficient {
          background: var(--insufficient-bg);
          border: 1px solid var(--insufficient-border);
          color: var(--insufficient);
        }
        .verification-status-surface.is-insufficient .status-explanation {
          color: var(--text-primary);
        }

        /* Uncertain */
        .verification-status-surface.is-uncertain {
          background: var(--uncertain-bg);
          border: 1px solid var(--uncertain-border);
          color: var(--uncertain);
        }
        .verification-status-surface.is-uncertain .status-explanation {
          color: var(--text-primary);
        }

        @media (prefers-reduced-motion: reduce) {
          .verification-status-surface {
            transition: none;
          }
        }
      `}</style>
    </ShineBorder>
  )
}
