"use client";

import React from "react";
import { motion, useReducedMotion } from "motion/react";
import { cn } from "@/lib/utils";

/**
 * Magic UI "Animated List" — vendored, data-driven variant.
 * Unlike the original (timer-grown notifications), every item exists in
 * markup and staggers in on mount — so reduced-motion users get the full
 * list instantly. MIT (magicuidesign/magicui).
 */
export function AnimatedList({ children, className, stagger = 0.07, ...props }) {
  const reduceMotion = useReducedMotion();
  const items = React.Children.toArray(children);

  if (reduceMotion) {
    return (
      <div className={cn(className)} {...props}>
        {items}
      </div>
    );
  }

  return (
    <motion.div
      className={cn(className)}
      initial="hidden"
      animate="show"
      variants={{
        show: { transition: { staggerChildren: stagger } },
      }}
      {...props}
    >
      {items.map((child, i) => (
        <motion.div
          key={i}
          variants={{
            hidden: { opacity: 0, y: 14 },
            show: {
              opacity: 1,
              y: 0,
              transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] },
            },
          }}
        >
          {child}
        </motion.div>
      ))}
    </motion.div>
  );
}
