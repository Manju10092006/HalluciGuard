import { cn } from "@/lib/utils";

export function LogoMark({ className, size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" className={cn("shrink-0", className)} aria-hidden="true">
      <rect x="1.5" y="1.5" width="29" height="29" rx="9" fill="var(--accent)" />
      <rect x="1.5" y="1.5" width="29" height="29" rx="9" fill="url(#hg-sheen)" />
      <path d="M9.5 16.5l4 4L22.5 11" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9.5 21.5h6" stroke="white" strokeOpacity=".55" strokeWidth="2" strokeLinecap="round" />
      <defs>
        <linearGradient id="hg-sheen" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="white" stopOpacity=".28" />
          <stop offset="1" stopColor="white" stopOpacity="0" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export function Wordmark({ className }) {
  return (
    <span className={cn("font-semibold tracking-[-0.02em] text-hg-text", className)}>
      Halluci<span className="text-hg-accent">Guard</span>
    </span>
  );
}

export function Orb({ className }) {
  return (
    <div className={cn("relative flex items-center justify-center", className)} aria-hidden="true">
      <div className="relative h-16 w-16 motion-safe:animate-hg-drift">
        {/* Soft iridescent glow halo */}
        <div className="pointer-events-none absolute -inset-3 rounded-full bg-[radial-gradient(circle,rgba(var(--accent-rgb),0.35)_0%,transparent_70%)] blur-md opacity-80" />
        
        {/* 3D spherical iridescent core */}
        <div
          className="relative h-full w-full rounded-full overflow-hidden shadow-[0_14px_30px_rgba(var(--accent-rgb),0.3)]"
          style={{
            background: "radial-gradient(circle at 35% 30%, #ffffff 0%, #dcdad2 32%, var(--accent) 72%, #000000 100%)",
            boxShadow: "inset -5px -5px 12px rgba(0, 0, 0, 0.4), inset 5px 5px 12px rgba(255, 255, 255, 0.75), 0 12px 28px rgba(var(--accent-rgb), 0.35)",
          }}
        >
          {/* Specular highlight */}
          <div
            className="absolute top-[15%] left-[22%] h-[20%] w-[28%] -rotate-[30deg] rounded-full bg-white/90 blur-[1.5px]"
          />
          {/* Internal reflection */}
          <div
            className="absolute bottom-[14%] right-[22%] h-[20%] w-[35%] rounded-full bg-white/25 blur-[3px]"
          />
        </div>

        {/* Soft ground shadow underneath */}
        <div
          className="absolute -bottom-3.5 left-1/2 h-2 w-9 -translate-x-1/2 rounded-[50%] bg-[radial-gradient(ellipse_at_center,rgba(0,0,0,0.2)_0%,transparent_75%)] blur-[2.5px] dark:bg-[radial-gradient(ellipse_at_center,rgba(0,0,0,0.5)_0%,transparent_75%)]"
        />
      </div>
    </div>
  );
}
