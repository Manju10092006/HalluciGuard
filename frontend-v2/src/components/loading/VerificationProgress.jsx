import React, { useState, useEffect } from 'react'
import { ShieldCheck } from 'lucide-react'

const STAGES = [
  'Analyzing claim structure and boundaries...',
  'Tracing primary evidence records...',
  'Comparing source citations and entailment...',
  'Checking contextual consistency and verdict...',
]

export function VerificationProgress() {
  const [currentStep, setCurrentStep] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentStep((prev) => (prev < STAGES.length - 1 ? prev + 1 : prev))
    }, 950)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="verification-progress-card">
      <div className="progress-top-row">
        <div className="progress-beacon">
          <ShieldCheck size={16} className="beacon-icon" />
          <span className="beacon-ping" />
        </div>
        <div className="stage-text-container">
          <span className="stage-label-fade" key={currentStep}>
            {STAGES[currentStep]}
          </span>
        </div>
      </div>

      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${((currentStep + 1) / STAGES.length) * 100}%` }}
        />
      </div>

      <style>{`
        .verification-progress-card {
          padding: 16px 20px;
          margin: 16px 0;
          border-radius: var(--radius-inline);
          background: var(--surface);
          border: 1px solid var(--border);
          box-shadow: var(--shadow-subtle);
          max-width: 520px;
        }

        .progress-top-row {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 12px;
        }

        .progress-beacon {
          position: relative;
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: var(--accent-light);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--accent);
          flex-shrink: 0;
        }

        .beacon-ping {
          position: absolute;
          inset: -3px;
          border-radius: 50%;
          border: 1.5px solid var(--accent);
          opacity: 0.5;
          animation: beacon-ripple 1.6s cubic-bezier(0.2, 0.8, 0.2, 1) infinite;
        }

        .stage-text-container {
          flex: 1;
          height: 20px;
          overflow: hidden;
          position: relative;
        }

        .stage-label-fade {
          display: block;
          font-size: 13.5px;
          font-weight: 500;
          color: var(--text-primary);
          animation: crossfade-in 220ms ease-out forwards;
        }

        .progress-bar-track {
          width: 100%;
          height: 3px;
          border-radius: var(--radius-pill);
          background: var(--surface-sunken);
          overflow: hidden;
        }

        .progress-bar-fill {
          height: 100%;
          background: var(--accent);
          border-radius: var(--radius-pill);
          transition: width 400ms cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes crossfade-in {
          from {
            opacity: 0;
            transform: translateY(4px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes beacon-ripple {
          0% {
            transform: scale(0.9);
            opacity: 0.8;
          }
          100% {
            transform: scale(1.4);
            opacity: 0;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .beacon-ping {
            display: none;
          }
          .stage-label-fade {
            animation: none !important;
          }
        }
      `}</style>
    </div>
  )
}
