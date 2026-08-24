"use client";

import * as React from "react";
import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
} from "react";
import {
  apiGetMe,
  apiGoogleAuth,
  apiLogin,
  apiLogout,
  apiRegister,
  getStoredToken,
  setStoredToken,
} from "@/lib/api/client";

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

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);

  // Validate stored JWT on initial mount
  useEffect(() => {
    let active = true;
    const storedToken = getStoredToken();

    if (!storedToken) {
      setStatus("anonymous");
      return;
    }

    setToken(storedToken);
    apiGetMe(storedToken)
      .then((res) => {
        if (!active) return;
        if (res.user) {
          setUser({
            id: res.user.id,
            sub: res.user.sub || res.user.id,
            email: res.user.email,
            name: res.user.name,
            picture: res.user.picture,
            created_at: res.user.created_at,
          });
          setStatus("authenticated");
        } else {
          setStoredToken(null);
          setToken(null);
          setUser(null);
          setStatus("anonymous");
        }
      })
      .catch(() => {
        if (!active) return;
        setStoredToken(null);
        setToken(null);
        setUser(null);
        setStatus("anonymous");
      });

    return () => {
      active = false;
    };
  }, []);

  const signInWithGoogle = useCallback(async (credential: string): Promise<AuthResult> => {
    try {
      const res = await apiGoogleAuth(credential);
      if (res.access_token && res.user) {
        setStoredToken(res.access_token);
        setToken(res.access_token);
        setUser({
          id: res.user.id,
          sub: res.user.sub || res.user.id,
          email: res.user.email,
          name: res.user.name,
          picture: res.user.picture,
          created_at: res.user.created_at,
        });
        setStatus("authenticated");
        return { ok: true };
      }
      return { ok: false, error: "Google authentication failed: No token returned." };
    } catch (err: any) {
      return {
        ok: false,
        error: err?.detail || err?.message || "Google authentication failed",
      };
    }
  }, []);

  const signIn = useCallback(async (email: string, password: string): Promise<AuthResult> => {
    try {
      const res = await apiLogin(email, password);
      if (res.access_token && res.user) {
        setStoredToken(res.access_token);
        setToken(res.access_token);
        setUser({
          id: res.user.id,
          sub: res.user.sub || res.user.id,
          email: res.user.email,
          name: res.user.name,
          picture: res.user.picture,
          created_at: res.user.created_at,
        });
        setStatus("authenticated");
        return { ok: true };
      }
      return { ok: false, error: "Authentication failed: No token returned." };
    } catch (err: any) {
      return {
        ok: false,
        error: err?.detail || err?.message || "Invalid email or password",
      };
    }
  }, []);

  const signUp = useCallback(
    async (email: string, password: string, name?: string): Promise<AuthResult> => {
      try {
        const res = await apiRegister(email, password, name);
        if (res.access_token && res.user) {
          setStoredToken(res.access_token);
          setToken(res.access_token);
          setUser({
            id: res.user.id,
            sub: res.user.sub || res.user.id,
            email: res.user.email,
            name: res.user.name,
            picture: res.user.picture,
            created_at: res.user.created_at,
          });
          setStatus("authenticated");
          return { ok: true };
        }
        return { ok: false, error: "Registration failed: No token returned." };
      } catch (err: any) {
        return {
          ok: false,
          error: err?.detail || err?.message || "Registration failed",
        };
      }
    },
    []
  );

  const signOut = useCallback(async () => {
    await apiLogout();
    setStoredToken(null);
    setToken(null);
    setUser(null);
    setStatus("anonymous");
  }, []);

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
