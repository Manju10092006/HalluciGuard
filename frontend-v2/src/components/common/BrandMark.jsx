import React from 'react'

export function BrandMark({ size = 28, className = '' }) {
  return (
    <div
      className={`brand-mark-wrapper ${className}`}
      style={{
        width: size,
        height: size,
        borderRadius: size * 0.28,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%)',
        boxShadow: '0 2px 8px var(--accent-glow)',
        flexShrink: 0,
      }}
      aria-hidden="true"
    >
      <svg
        width={size * 0.65}
        height={size * 0.65}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Verification Shield & Trace Ring */}
        <path
          d="M12 2L4 5.5V11C4 16.5 7.5 21 12 22C16.5 21 20 16.5 20 11V5.5L12 2Z"
          stroke="#FFFFFF"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.95"
        />
        <path
          d="M9 11.5L11 13.5L15.5 9"
          stroke="#FFFFFF"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  )
}
