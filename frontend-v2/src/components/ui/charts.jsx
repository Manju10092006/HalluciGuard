"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

/**
 * Animata "Donut Chart" — vendored, zero-dependency SVG ring.
 * Animates once on mount via CSS transition; resolves to the final state
 * under prefers-reduced-motion. MIT (codse/animata).
 */
export function DonutChart({
  value,
  size = 40,
  strokeWidth = 4,
  className,
  label,
  ...props
}) {
  const [mounted, setMounted] = useState(false);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    setReduced(
      typeof window !== "undefined" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
    const raf = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  const clamped = Math.max(0, Math.min(100, Number(value) || 0));
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const shown = mounted || reduced ? clamped : 0;
  const offset = circumference - (circumference * shown) / 100;

  return (
    <span
      role="img"
      aria-label={label ?? `${clamped}% confidence`}
      className={cn("inline-flex shrink-0", className)}
      {...props}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--border)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={
            reduced
              ? undefined
              : { transition: "stroke-dashoffset 700ms cubic-bezier(0.16,1,0.3,1)" }
          }
        />
      </svg>
    </span>
  );
}

/**
 * Animata "Gauge Chart" — vendored, zero-dependency SVG dial.
 * 270° arc with a gap at the bottom; centered JetBrains Mono value.
 * MIT (codse/animata).
 */
export function GaugeChart({
  value,
  size = 120,
  strokeWidth = 10,
  className,
  label,
  ...props
}) {
  const [mounted, setMounted] = useState(false);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    setReduced(
      typeof window !== "undefined" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
    const raf = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  const clamped = Math.max(0, Math.min(100, Number(value) || 0));
  const radius = (size - strokeWidth) / 2;
  // 270-degree arc: start at 135°, sweep 270°
  const startAngle = 135;
  const sweep = 270;
  const polar = (angleDeg) => {
    const a = ((angleDeg - 90) * Math.PI) / 180;
    return {
      x: size / 2 + radius * Math.cos(a),
      y: size / 2 + radius * Math.sin(a),
    };
  };
  const start = polar(startAngle);
  const end = polar(startAngle + sweep);
  const arcPath = `M ${start.x} ${start.y} A ${radius} ${radius} 0 1 1 ${end.x} ${end.y}`;
  const shown = mounted || reduced ? clamped : 0;

  return (
    <span
      role="img"
      aria-label={label ?? `${clamped}% confidence`}
      className={cn("relative inline-flex shrink-0 items-center justify-center", className)}
      style={{ width: size, height: size }}
      {...props}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path
          d={arcPath}
          fill="none"
          stroke="var(--border)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        <path
          d={arcPath}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          pathLength={100}
          strokeDasharray={100}
          strokeDashoffset={100 - shown}
          style={
            reduced
              ? undefined
              : { transition: "stroke-dashoffset 800ms cubic-bezier(0.16,1,0.3,1)" }
          }
        />
      </svg>
      <span
        className="absolute font-mono text-[15px] font-semibold text-hg-text"
        style={{ fontFamily: "var(--font-mono)" }}
      >
        {Math.round(clamped)}
      </span>
    </span>
  );
}
