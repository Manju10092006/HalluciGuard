// One transport for authentication, verification, health and history.
// The default stays on the Vercel origin; BACKEND_URL is server-side only.
export const AUTH_TOKEN_KEY = "hg.auth.jwt.v1";
export const AUTH_USER_KEY = "hg.auth.user.v1";
// A stale public backend URL must never send browser requests to a retired
// service. All traffic goes through the configured server-side origin.
const base = "/api/backend";

export async function backendRequest(path, { token, timeout, ...options } = {}) {
  const configuredTimeout = Number(process.env.NEXT_PUBLIC_API_TIMEOUT_MS || 290000);
  const timeoutMs = timeout ?? (Number.isFinite(configuredTimeout) && configuredTimeout > 0 ? configuredTimeout : 290000);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (options.body) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  try {
    const response = await fetch(`${base}${path}`, { ...options, headers, signal: controller.signal, cache: "no-store" });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = typeof payload?.detail === "string" ? payload.detail : `Backend request failed (${response.status}).`;
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }
    if (!payload || typeof payload !== "object") throw new Error("The backend returned an invalid response.");
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Verification timed out. Please retry; no verdict was produced.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
