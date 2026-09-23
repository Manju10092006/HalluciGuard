import React from 'react'

export function VerificationOrb({ size = 68 }) {
  return (
    <div className="verification-orb-container" aria-hidden="true">
      <div
        className="verification-orb"
        style={{
          width: size,
          height: size,
        }}
      >
        {/* Glow halo */}
        <div className="orb-halo" />
        {/* Main 3D spherical gradient */}
        <div className="orb-sphere">
          <div className="orb-specular" />
          <div className="orb-reflection" />
        </div>
        {/* Subtle shadow underneath */}
        <div className="orb-shadow" />
      </div>

      <style>{`
        .verification-orb-container {
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 24px;
        }

        .verification-orb {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          animation: orb-float 6s ease-in-out infinite;
        }

        .orb-halo {
          position: absolute;
          inset: -12px;
          border-radius: 50%;
          background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
          filter: blur(10px);
          opacity: 0.7;
          pointer-events: none;
        }

        .orb-sphere {
          position: relative;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          background: radial-gradient(circle at 35% 30%, #ffffff 0%, #c4b5fd 35%, #7c3aed 70%, #4338ca 100%);
          box-shadow:
            inset -6px -6px 14px rgba(30, 20, 80, 0.4),
            inset 6px 6px 14px rgba(255, 255, 255, 0.8),
            0 12px 28px rgba(109, 94, 245, 0.28);
          overflow: hidden;
        }

        [data-theme='dark'] .orb-sphere {
          background: radial-gradient(circle at 35% 30%, #f5f3ff 0%, #a78bfa 35%, #6d28d9 75%, #312e81 100%);
          box-shadow:
            inset -6px -6px 16px rgba(0, 0, 0, 0.7),
            inset 6px 6px 16px rgba(255, 255, 255, 0.5),
            0 12px 32px rgba(139, 124, 246, 0.35);
        }

        .orb-specular {
          position: absolute;
          top: 15%;
          left: 22%;
          width: 28%;
          height: 20%;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.85);
          filter: blur(2px);
          transform: rotate(-30deg);
        }

        .orb-reflection {
          position: absolute;
          bottom: 12%;
          right: 22%;
          width: 35%;
          height: 20%;
          border-radius: 50%;
          background: rgba(167, 139, 250, 0.45);
          filter: blur(4px);
        }

        .orb-shadow {
          position: absolute;
          bottom: -14px;
          width: 55%;
          height: 8px;
          border-radius: 50%;
          background: radial-gradient(ellipse at center, rgba(0, 0, 0, 0.15) 0%, transparent 75%);
          filter: blur(3px);
          animation: orb-shadow-pulse 6s ease-in-out infinite;
        }

        [data-theme='dark'] .orb-shadow {
          background: radial-gradient(ellipse at center, rgba(0, 0, 0, 0.45) 0%, transparent 75%);
        }

        @keyframes orb-float {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-5px);
          }
        }

        @keyframes orb-shadow-pulse {
          0%, 100% {
            transform: scale(1);
            opacity: 0.8;
          }
          50% {
            transform: scale(0.85);
            opacity: 0.5;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .verification-orb, .orb-shadow {
            animation: none !important;
          }
        }
      `}</style>
    </div>
  )
}
