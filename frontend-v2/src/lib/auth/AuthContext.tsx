"use client";
import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
export type AuthStatus = "loading" | "anonymous" | "authenticated";
export interface AuthUser { id: string; sub?: string; email: string; name: string; picture?: string; created_at?: string; }
export interface AuthResult { ok: boolean; error?: string; }
export interface AuthContextValue { enabled: boolean; status: AuthStatus; user: AuthUser | null; token: string | null; signIn: (email:string,password:string)=>Promise<AuthResult>; signUp:(email:string,password:string,name?:string)=>Promise<AuthResult>; signInWithGoogle:(credential:string)=>Promise<AuthResult>; signOut:()=>Promise<void>; }
const KEY = "hg.auth.jwt.v1", USER = "hg.auth.user.v1";
const BASE = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");
const Context = createContext<AuthContextValue | null>(null);
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status,setStatus] = useState<AuthStatus>("loading"), [user,setUser] = useState<AuthUser|null>(null), [token,setToken] = useState<string|null>(null);
  const clear = useCallback(() => { localStorage.removeItem(KEY); localStorage.removeItem(USER); setToken(null); setUser(null); setStatus("anonymous"); },[]);
  useEffect(() => {
    const stored = localStorage.getItem(KEY); let active = true;
    if (!stored) { setStatus("anonymous"); return; }
    fetch(BASE+"/auth/me", { headers: {Authorization:`Bearer ${stored}`} }).then(async r => {
      if (!r.ok) throw new Error("Session invalid"); const data = await r.json();
      if (active) { setToken(stored); setUser(data.user); setStatus("authenticated"); }
    }).catch(() => { if (active) clear(); });
    return () => { active = false; };
  },[clear]);
  const authenticate = useCallback(async (path:string, body:object):Promise<AuthResult> => {
    try {
      const r = await fetch(BASE+path,{ method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
      const data = await r.json(); if (!r.ok || !data.access_token || !data.user) throw new Error(data.detail || "Authentication failed");
      localStorage.setItem(KEY,data.access_token); localStorage.setItem(USER,JSON.stringify(data.user));
      setToken(data.access_token); setUser(data.user); setStatus("authenticated"); return {ok:true};
    } catch (e:any) { return {ok:false,error:e.message || "Authentication service unavailable"}; }
  },[]);
  const value:AuthContextValue = { enabled:true,status,user,token,
    signIn:(email,password)=>authenticate('/auth/login',{email,password}),
    signUp:(email,password,name)=>authenticate('/auth/register',{email,password,name}),
    signInWithGoogle:credential=>authenticate('/auth/google',{credential}),
    signOut:async()=>clear(),
  };
  return <Context.Provider value={value}>{children}</Context.Provider>;
}
export function useAuth():AuthContextValue { const value=useContext(Context); if (!value) throw new Error("AuthProvider required"); return value; }
