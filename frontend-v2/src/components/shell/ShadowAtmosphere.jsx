"use client";

/**
 * Architectural sunlight + window-blind directional motion-blur atmosphere.
 * Ports the Cofounder / HalluciGuard editorial spec:
 *  - directional Gaussian motion blur (stdDeviation 55 18) on 34° diagonal slats
 *  - soft foliage corner blur (stdDeviation 35 12)
 * Rendered fixed behind the app shell; purely decorative.
 */
export function ShadowAtmosphere() {
  return (
    <div className="shadow-atmosphere" aria-hidden="true">
      <svg className="sunlight-svg" viewBox="0 0 1440 900" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
        <defs>
          <filter id="directionalMotionBlur" x="-30%" y="-30%" width="160%" height="160%" filterUnits="objectBoundingBox">
            <feGaussianBlur stdDeviation="55 18" />
          </filter>
          <filter id="softFoliageMotionBlur" x="-40%" y="-40%" width="180%" height="180%" filterUnits="objectBoundingBox">
            <feGaussianBlur stdDeviation="35 12" />
          </filter>
        </defs>

        {/* Base soft sunlight pool */}
        <ellipse cx="950" cy="220" rx="700" ry="480" fill="#FFFFFF" opacity="0.5" filter="url(#directionalMotionBlur)" />

        {/* Architectural window-blind shadows (directional motion-blur slats) */}
        <g opacity="0.08" filter="url(#directionalMotionBlur)">
          <rect x="420" y="-120" width="85" height="1300" transform="rotate(34 420 -120)" fill="#111111" />
          <rect x="570" y="-120" width="80" height="1300" transform="rotate(34 570 -120)" fill="#111111" />
          <rect x="720" y="-120" width="90" height="1300" transform="rotate(34 720 -120)" fill="#111111" />
          <rect x="870" y="-120" width="75" height="1300" transform="rotate(34 870 -120)" fill="#111111" />
          <rect x="1020" y="-120" width="95" height="1300" transform="rotate(34 1020 -120)" fill="#111111" />
          <rect x="1170" y="-120" width="85" height="1300" transform="rotate(34 1170 -120)" fill="#111111" />
          <rect x="1320" y="-120" width="105" height="1300" transform="rotate(34 1320 -120)" fill="#111111" />
        </g>

        {/* Organic foliage / window-structure motion-blurred shadows */}
        <g opacity="0.09" filter="url(#softFoliageMotionBlur)">
          <path d="M1080,40 Q1240,110 1420,70 Q1310,260 1460,420 Q1190,320 1080,40 Z" fill="#050505" />
          <path d="M1220,-60 Q1340,140 1490,190 Q1370,360 1495,570 Q1240,420 1220,-60 Z" fill="#050505" />
          <ellipse cx="1340" cy="190" rx="110" ry="80" transform="rotate(-15 1340 190)" fill="#050505" />
          <ellipse cx="1420" cy="330" rx="130" ry="95" transform="rotate(-20 1420 330)" fill="#050505" />
        </g>
      </svg>
    </div>
  );
}
