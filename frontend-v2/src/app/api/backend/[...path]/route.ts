import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const methods: Record<string, string[]> = {
  health: ["GET"], verify: ["POST"],
  "auth/register": ["POST"], "auth/login": ["POST"],
  "auth/google": ["POST"], "auth/me": ["GET"], "auth/logout": ["POST"],
  "api/history": ["GET", "POST", "DELETE"],
};

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const path = (await context.params).path.join("/");
  if (!methods[path]?.includes(request.method)) {
    return NextResponse.json({ detail: "Unsupported backend endpoint." }, { status: 404 });
  }
  const configured = process.env.BACKEND_URL?.trim();
  if (!configured) return NextResponse.json({ detail: "The verification backend is not configured." }, { status: 503 });
  let origin: URL;
  try {
    origin = new URL(configured);
    if (!['http:', 'https:'].includes(origin.protocol) || origin.username || origin.password ||
        (process.env.VERCEL && origin.protocol !== 'https:') || origin.origin === request.nextUrl.origin ||
        origin.pathname !== '/' || origin.search || origin.hash) throw new Error("Invalid backend origin");
  } catch {
    return NextResponse.json({ detail: "The verification backend configuration is invalid." }, { status: 503 });
  }
  const target = new URL(`/${path}`, origin);
  target.search = request.nextUrl.search;
  const headers = new Headers({ Accept: "application/json" });
  for (const name of ["authorization", "content-type", "x-request-id"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const body = request.method === "GET" ? undefined : await request.text();
  if (body && new TextEncoder().encode(body).length > 1_000_000) {
    return NextResponse.json({ detail: "Request is too large." }, { status: 413 });
  }
  try {
    const upstream = await fetch(target, {
      method: request.method, headers, body, cache: "no-store", redirect: "manual",
      signal: AbortSignal.timeout(path === "verify" ? 285000 : 20000),
    });
    if (upstream.status >= 300 && upstream.status < 400) throw new Error("Unexpected backend redirect");
    const responseHeaders = new Headers({ "Cache-Control": "no-store" });
    responseHeaders.set("Content-Type", upstream.headers.get("content-type") || "application/json");
    for (const name of ["x-request-id", "www-authenticate"]) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch (error) {
    const timedOut = error instanceof Error && ["TimeoutError", "AbortError"].includes(error.name);
    return NextResponse.json({ detail: timedOut ? "The backend timed out; no verdict was produced." : "The verification backend is unavailable." }, { status: timedOut ? 504 : 502 });
  }
}

export { proxy as GET, proxy as POST, proxy as DELETE };
