/** @type {import('next').NextConfig} */
const backendUrl =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "https://shannon-diagnosis-ment-striking.trycloudflare.com";

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_GOOGLE_CLIENT_ID:
      process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ||
      "88154202029-lrr58hkhqmu7td24ln93i6t21jp8hki2.apps.googleusercontent.com",
  },
  // Vercel builds run type-checking and lint by default. Keep type safety on;
  // don't fail the production build on lint-only issues.
  eslint: {
    ignoreDuringBuilds: true,
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/auth/:path*",
        destination: `${backendUrl}/auth/:path*`,
      },
      {
        source: "/api/v1/auth/:path*",
        destination: `${backendUrl}/api/v1/auth/:path*`,
      },
      {
        source: "/verify",
        destination: `${backendUrl}/verify`,
      },
      {
        source: "/api/history/:path*",
        destination: `${backendUrl}/api/history/:path*`,
      },
      {
        source: "/api/history",
        destination: `${backendUrl}/api/history`,
      },
      {
        source: "/health",
        destination: `${backendUrl}/health`,
      },
    ];
  },
};

export default nextConfig;
