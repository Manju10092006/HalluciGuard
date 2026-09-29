import React, { useState, useEffect } from 'react'
import { ShieldCheck } from 'lucide-react'
import { Progress } from '@/components/ui/progress'

const STAGES = [
  'Analyzing claim structure and boundaries...',
  'Tracing primary evidence records...',
  'Comparing source citations and entailment...',
  'Checking contextual consistency and verdict...',
]

/**
 * Verification run loading card.
 * - hover.dev "Bar Loader": thin indeterminate top bar (CSS keyframes only).
 * - shadcn "Progress": determinate thin bar + mono percentage (Radix).
 * - hover.dev "Cutout Text Loader": typographic "Verifying" headline.
 * Reduced-motion: static final state, no loops.
 */
export function VerificationProgress() {
  const [currentStep, setCurrentStep] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentStep((prev) => (prev < STAGES.length - 1 ? prev + 1 : prev))
    }, 950)
    return () => clearInterval(timer)
  }, [])

  const percent = Math.round(((currentStep + 1) / STAGES.length) * 100)

  return (
    <div className="verification-progress-card">
      {/* Indeterminate top bar loader */}
      <div className="bar-loader-track" aria-hidden="true">
        <div className="bar-loader-segment" />
      </div>

      <div className="progress-top-row">
        <div className="progress-beacon">
          <ShieldCheck size={16} className="beacon-icon" />
          <span className="beacon-ping" />
        </div>
        <div className="stage-text-container">
          <span className="cutout-headline">Verifying</span>
          <span className="stage-label-fade" key={currentStep}>
            {STAGES[currentStep]}
          </span>
        </div>
        <span className="progress-percent" aria-label={`${percent} percent complete`}>
          {percent}%
        </span>
      </div>

      <Progress value={percent} aria-label="Verification progress" />

      <style>{`
        .verification-progress-card {
          position: relative;
          padding: 16px 20px;
          margin: 16px 0;
          border-radius: var(--radius-inline);
          background: var(--surface);
          border: 1px solid var(--border);
          box-shadow: var(--shadow-subtle);
          max-width: 520px;
          overflow: hidden;
        }

        /* hover.dev Bar Loader — thin indeterminate segment */
        .bar-loader-track {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 2px;
          background: transparent;
          overflow: hidden;
        }
        .bar-loader-segment {
          position: absolute;
          top: 0;
          left: 0;
          height: 100%;
          width: 32%;
          background: var(--accent);
          border-radius: 2px;
          animation: bar-slide 1.4s cubic-bezier(0.45, 0, 0.55, 1) infinite;
        }
        @keyframes bar-slide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(320%); }
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
          min-height: 40px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 1px;
          overflow: hidden;
          position: relative;
        }

        /* hover.dev Cutout Text Loader — typographic liveness */
        .cutout-headline {
          font-family: var(--font-display);
          font-size: 15px;
          font-weight: 600;
          letter-spacing: 0.01em;
          color: var(--text-primary);
          background: linear-gradient(100deg, var(--text-primary) 30%, var(--text-muted) 50%, var(--text-primary) 70%);
          background-size: 220% 100%;
          -webkit-background-clip: text;
          background-clip: text;
          -webkit-text-fill-color: transparent;
          animation: cutout-sheen 2.2s ease-in-out infinite;
        }
        @keyframes cutout-sheen {
          0% { background-position: 110% 0; }
          100% { background-position: -110% 0; }
        }

        .stage-label-fade {
          display: block;
          font-size: 12.5px;
          font-weight: 400;
          color: var(--text-secondary);
          animation: crossfade-in 220ms ease-out forwards;
        }

        .progress-percent {
          font-family: var(--font-mono);
          font-size: 11.5px;
          font-weight: 600;
          color: var(--text-muted);
          flex-shrink: 0;
          min-width: 38px;
          text-align: right;
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
          .beacon-ping,
          .bar-loader-segment {
            display: none;
          }
          .stage-label-fade {
            animation: none !important;
          }
          .cutout-headline {
            animation: none;
            background: none;
            -webkit-text-fill-color: currentColor;
          }
        }
      `}</style>
    </div>
  )
}
