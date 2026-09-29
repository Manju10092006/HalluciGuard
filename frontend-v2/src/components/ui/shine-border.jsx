"use client";

import { cn } from "@/lib/utils";

/**
 * Magic UI "Shine Border" — vendored, zero-dependency CSS.
 * A single slow monochrome light sweep along the element's border ring.
 * Retokened: uses `currentColor` (the verdict color on status surfaces),
 * slow period, disabled under prefers-reduced-motion.
 * MIT (magicuidesign/magicui).
 */
export function ShineBorder({
  className,
  children,
  borderWidth = 1,
  duration = 14,
  ...props
}) {
  return (
    <div
      className={cn("hg-shine-border", className)}
      style={
        {
          "--shine-border-width": `${borderWidth}px`,
          "--shine-duration": `${duration}s`,
        }
      }
      {...props}
    >
      {children}
      <style>{`
        .hg-shine-border {
          position: relative;
        }
        .hg-shine-border::after {
          content: "";
          position: absolute;
          inset: 0;
          border-radius: inherit;
          padding: var(--shine-border-width, 1px);
          background: linear-gradient(
            110deg,
            transparent 42%,
            currentColor 50%,
            transparent 58%
          );
          background-size: 300% 300%;
          -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          -webkit-mask-composite: xor;
          mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          mask-composite: exclude;
          opacity: 0.45;
          pointer-events: none;
          animation: hg-shine-sweep var(--shine-duration, 14s) linear infinite;
        }
        @keyframes hg-shine-sweep {
          0% { background-position: 120% 0; }
          100% { background-position: -120% 0; }
        }
        @media (prefers-reduced-motion: reduce) {
          .hg-shine-border::after {
            animation: none;
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}
