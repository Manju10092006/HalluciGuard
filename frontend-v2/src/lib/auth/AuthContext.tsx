"use client";

import * as React from "react";
import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
} from "react";

export type AuthStatus = "loading" | "anonymous" | "authenticated";

export interface AuthUser {
  id: string;
  sub: string;
  email: string;
  name: string;
  picture?: string;
  created_at?: string;
}

export interface AuthResult {
  ok: boolean;
  error?: string;
}

export interface AuthContextValue {
  enabled: boolean;
  status: AuthStatus;
  user: AuthUser | null;
  token: string | null;
  signIn: (email: string, password: string) => Promise<AuthResult>;
  signUp: (email: string, password: string, name?: string) => Promise<AuthResult>;
  signInWithGoogle: (credential: string) => Promise<AuthResult>;
  signOut: () => Promise<void>;
}

import { backendRequest, AUTH_TOKEN_KEY, AUTH_USER_KEY } from "@/lib/backend";

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);

  const clearSession = useCallback(() => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
    setToken(null);
    setUser(null);
    setStatus("anonymous");
  }, []);

  // Stored display data is never authentication. Validate the server JWT.
  useEffect(() => {
    let alive = true;
    const storedToken = localStorage.getItem(AUTH_TOKEN_KEY);
    if (!storedToken) { setStatus("anonymous"); return; }
    backendRequest("/auth/me", { token: storedToken, timeout: 20000 })
      .then((data: any) => {
        if (!alive) return;
        setToken(storedToken);
        setUser(data.user);
        setStatus("authenticated");
      })
      .catch(() => { if (alive) clearSession(); });
    return () => { alive = false; };
  }, [clearSession]);

  const authenticate = useCallback(async (path: string, body: object): Promise<AuthResult> => {
    try {
      const data = await backendRequest(path, { method: "POST", body: JSON.stringify(body), timeout: 20000 });
      if (!data.access_token || !data.user?.id) throw new Error("Invalid authentication response");
      localStorage.setItem(AUTH_TOKEN_KEY, data.access_token);
      localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
      setToken(data.access_token);
      setUser(data.user);
      setStatus("authenticated");
      return { ok: true };
    } catch (error: any) {
      return { ok: false, error: error.message || "Authentication failed" };
    }
  }, []);

  const signIn = useCallback((email: string, password: string) =>
    authenticate("/auth/login", { email: email.trim(), password }), [authenticate]);
  const signUp = useCallback((email: string, password: string, name?: string) =>
    authenticate("/auth/register", { email: email.trim(), password, name }), [authenticate]);
  const signInWithGoogle = useCallback((credential: string) =>
    authenticate("/auth/google", { credential }), [authenticate]);
  const signOut = useCallback(async () => { clearSession(); }, [clearSession]);

  const value: AuthContextValue = {
    enabled: true,
    status,
    user,
    token,
    signIn,
    signUp,
    signInWithGoogle,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    return {
      enabled: false,
      status: "anonymous",
      user: null,
      token: null,
      signIn: async () => ({ ok: false, error: "Auth context missing" }),
      signUp: async () => ({ ok: false, error: "Auth context missing" }),
      signInWithGoogle: async () => ({ ok: false, error: "Auth context missing" }),
      signOut: async () => {},
    };
  }
  return ctx;
}
