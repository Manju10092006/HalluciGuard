"use client";

import * as React from "react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth/AuthContext";
import { ShieldCheck, ArrowRight, Loader2, AlertCircle } from "lucide-react";
import { GoogleSignInButton } from "@/components/auth/GoogleSignInButton";

export default function LandingPage() {
  const { status, signIn, signUp } = useAuth();
  const router = useRouter();

  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // If already authenticated, automatically redirect to the app workspace.
  useEffect(() => {
    if (status === "authenticated") {
      router.replace("/app");
    }
  }, [status, router]);

  if (status === "loading" || status === "authenticated") {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <Loader2 className="h-6 w-6 animate-spin text-signal" />
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setSubmitting(true);

    try {
      if (mode === "signin") {
        const res = await signIn(email.trim(), password);
        if (!res.ok) {
          setErrorMessage(res.error || "Invalid email or password");
        } else {
          router.replace("/app");
        }
      } else {
        const res = await signUp(email.trim(), password, name.trim());
        if (!res.ok) {
          setErrorMessage(res.error || "Registration failed");
        } else {
          router.replace("/app");
        }
      }
    } catch (err: any) {
      setErrorMessage(err?.message || "An unexpected error occurred. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex h-full min-h-screen w-full flex-col bg-background text-ink">
      <div className="flex flex-1 flex-col items-center justify-center px-4 py-12">
        {/* Brand Hero */}
        <div className="mb-10 flex flex-col items-center text-center">
          <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-panel-raised border border-line shadow-sm">
            <ShieldCheck className="h-7 w-7 text-signal" />
          </div>
          <h1 className="mb-2 text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            HalluciGuard
          </h1>
          <p className="max-w-md text-sm text-ink-muted leading-relaxed">
            Real-time verification against authoritative sources, cross-encoder reranking, and NLI entailment.
          </p>
        </div>

        {/* Auth Card */}
        <div className="w-full max-w-sm rounded-2xl border border-line bg-panel p-6 sm:p-8 shadow-md">
          {/* Google Sign-In Primary Option */}
          <div className="mb-5">
            <GoogleSignInButton
              onSuccess={() => router.replace("/app")}
              onError={(err) => setErrorMessage(err)}
            />
          </div>

          {/* Divider */}
          <div className="relative my-5 flex items-center justify-center">
            <div className="w-full border-t border-line" />
            <span className="absolute bg-panel px-3 text-[11px] font-medium uppercase tracking-wider text-ink-dim">
              or continue with email
            </span>
          </div>

          {/* Mode Switcher */}
          <div className="mb-6 grid grid-cols-2 gap-1 rounded-lg bg-panel-raised p-1 border border-line">
            <button
              type="button"
              onClick={() => {
                setMode("signin");
                setErrorMessage(null);
              }}
              className={`rounded-md py-1.5 text-xs font-medium transition-colors ${
                mode === "signin"
                  ? "bg-signal text-white shadow-xs"
                  : "text-ink-muted hover:text-ink"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => {
                setMode("signup");
                setErrorMessage(null);
              }}
              className={`rounded-md py-1.5 text-xs font-medium transition-colors ${
                mode === "signup"
                  ? "bg-signal text-white shadow-xs"
                  : "text-ink-muted hover:text-ink"
              }`}
            >
              Create Account
            </button>
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div className="mb-5 flex items-start gap-2.5 rounded-lg border border-contradicted/30 bg-contradicted/10 p-3 text-xs text-contradicted">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <div className="leading-snug">{errorMessage}</div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            {mode === "signup" && (
              <div>
                <label
                  htmlFor="name"
                  className="mb-1.5 block text-xs font-medium text-ink-muted"
                >
                  Full Name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Jane Doe"
                  className="w-full rounded-lg border border-line bg-panel-raised px-3 py-2 text-sm text-ink placeholder:text-ink-dim outline-none focus:border-signal focus:ring-1 focus:ring-signal"
                />
              </div>
            )}

            <div>
              <label
                htmlFor="email"
                className="mb-1.5 block text-xs font-medium text-ink-muted"
              >
                Email Address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="demo@halluciguard.ai"
                required
                className="w-full rounded-lg border border-line bg-panel-raised px-3 py-2 text-sm text-ink placeholder:text-ink-dim outline-none focus:border-signal focus:ring-1 focus:ring-signal"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="mb-1.5 block text-xs font-medium text-ink-muted"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                minLength={6}
                className="w-full rounded-lg border border-line bg-panel-raised px-3 py-2 text-sm text-ink placeholder:text-ink-dim outline-none focus:border-signal focus:ring-1 focus:ring-signal"
              />
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-signal px-4 py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-95 disabled:opacity-50 cursor-pointer"
            >
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  {mode === "signin" ? "Sign In" : "Get Started"}
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>

          {/* Quick Demo Hint */}
          <div className="mt-5 rounded-lg border border-line bg-panel-raised/50 p-2.5 text-center text-xs text-ink-dim">
            Default Demo: <span className="font-mono text-ink">demo@halluciguard.ai</span> /{" "}
            <span className="font-mono text-ink">password123</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="flex justify-center pb-6 text-xs text-ink-dim">
        HalluciGuard Production Engine · End-to-End Grounded Verification
      </footer>
    </div>
  );
}
