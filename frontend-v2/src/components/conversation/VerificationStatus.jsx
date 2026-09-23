import React from 'react'
import { ShieldCheck, AlertOctagon, HelpCircle, AlertTriangle } from 'lucide-react'
import { VERIFICATION_STATUS } from '../../types/verification'

export function VerificationStatus({ status, title, description }) {
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

  return (
    <div className={`verification-status-surface ${config.className}`}>
      <div className="status-header-row">
        <Icon size={16} className="status-indicator-icon" />
        <span className="status-label">{title || config.defaultTitle}</span>
      </div>
      <p className="status-explanation">{description || config.defaultDesc}</p>

      <style>{`
        .verification-status-surface {
          border-radius: var(--radius-inline);
          padding: 12px 16px;
          margin: 14px 0 16px;
          display: flex;
          flex-direction: column;
          gap: 4px;
          transition: all 140ms ease;
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
      `}</style>
    </div>
  )
}
