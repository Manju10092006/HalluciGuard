"use client";

import { useEffect, useRef, useState } from "react";
import { useInView, useMotionValue, useSpring, useReducedMotion } from "motion/react";
import { cn } from "@/lib/utils";

/**
 * Magic UI "Number Ticker" — vendored, spring count-up.
 * Renders the final value immediately under prefers-reduced-motion.
 * MIT (magicuidesign/magicui).
 */
export function NumberTicker({
  value,
  className,
  decimalPlaces = 0,
  prefix = "",
  suffix = "",
  ...props
}) {
  const ref = useRef(null);
  const reduceMotion = useReducedMotion();
  const motionValue = useMotionValue(0);
  const springValue = useSpring(motionValue, { damping: 60, stiffness: 120 });
  const isInView = useInView(ref, { once: true, margin: "0px" });
  const [display, setDisplay] = useState((0).toFixed(decimalPlaces));

  useEffect(() => {
    if (isInView) motionValue.set(value);
  }, [isInView, value, motionValue]);

  useEffect(() => {
    if (reduceMotion) {
      setDisplay(Number(value).toFixed(decimalPlaces));
      return;
    }
    const unsub = springValue.on("change", (v) =>
      setDisplay(Number(v).toFixed(decimalPlaces))
    );
    return unsub;
  }, [springValue, value, decimalPlaces, reduceMotion]);

  return (
    <span
      ref={ref}
      className={cn("tabular-nums", className)}
      aria-label={`${prefix}${Number(value).toFixed(decimalPlaces)}${suffix}`}
      {...props}
    >
      {prefix}
      {reduceMotion ? Number(value).toFixed(decimalPlaces) : display}
      {suffix}
    </span>
  );
}
