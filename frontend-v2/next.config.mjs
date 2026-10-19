import { fileURLToPath } from 'node:url';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  outputFileTracingRoot: fileURLToPath(new URL('.', import.meta.url)),
  distDir: process.env.HALLUCIGUARD_NEXT_DIST_DIR || '.next',
  output: 'standalone',
};

export default nextConfig;
