"use client";

import { useRef } from "react";
import { cn } from "@/lib/utils";

/**
 * React Bits "SpotlightCard" — vendored, zero-dependency.
 * Soft cursor-following light wash on cards: pure light, no blur glass,
 * no color. Gated behind (pointer: fine) + prefers-reduced-motion.
 * MIT + Commons Clause v1.0 (DavidHDev/react-bits).
 */
export function SpotlightCard({
  children,
  className,
  spotlightColor = "rgba(26, 26, 24, 0.055)",
  ...props
}) {
  const ref = useRef(null);

  const handleMouseMove = (e) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    el.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`);
    el.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`);
  };

  return (
    <div
      ref={ref}
      onMouseMove={handleMouseMove}
      className={cn("hg-spotlight", className)}
      style={{ "--spotlight-color": spotlightColor }}
      {...props}
    >
      {children}
      <style>{`
        .hg-spotlight {
          position: relative;
        }
        .hg-spotlight::before {
          content: "";
          position: absolute;
          inset: 0;
          border-radius: inherit;
          pointer-events: none;
          background: radial-gradient(
            260px circle at var(--mouse-x, 50%) var(--mouse-y, 50%),
            var(--spotlight-color, rgba(26, 26, 24, 0.055)),
            transparent 70%
          );
          opacity: 0;
          transition: opacity 220ms ease;
        }
        .hg-spotlight:hover::before {
          opacity: 1;
        }
        @media (pointer: coarse) {
          .hg-spotlight::before {
            display: none;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .hg-spotlight::before {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}
