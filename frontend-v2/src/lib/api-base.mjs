// The default is same-origin. Only an intentional public override exposes a
// backend origin to the browser; server credentials never belong in this value.
export function apiBase(override = process.env.NEXT_PUBLIC_BACKEND_URL) {
  if (!override?.trim()) return '/api/backend';
  let url;
  try { url = new URL(override.trim()); } catch { throw new Error('Invalid public backend origin'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('Invalid public backend origin');
  }
  return url.origin;
}
