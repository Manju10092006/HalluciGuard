import axios from "axios";
import { verificationView } from "./verification-contract.mjs";
const API_BASE = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");
const http = axios.create({ baseURL: API_BASE, timeout: 300000 });
const STORAGE = "hg_local_conversations_v2", PROJECTS = "hg_local_projects_v2";
let conversations = [], projects = [], loaded = false, namespace = "";
function scope() { try { return JSON.parse(localStorage.getItem("hg.auth.user.v1") || "null")?.id || "guest"; } catch { return "guest"; } }
function load() {
  if (typeof window === "undefined") return;
  const nextScope = scope();
  if (loaded && namespace === nextScope) return;
  namespace = nextScope;
  try { conversations = JSON.parse(localStorage.getItem(STORAGE+namespace) || "[]"); } catch { conversations = []; }
  try { projects = JSON.parse(localStorage.getItem(PROJECTS+namespace) || "[]"); } catch { projects = []; }
  loaded = true;
}
function save() {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE+namespace, JSON.stringify(conversations));
  localStorage.setItem(PROJECTS+namespace, JSON.stringify(projects));
}
function find(id) { load(); const c = conversations.find(c => c.id === id); if (!c) throw new Error("Conversation not found"); return c; }
const identifier = () => globalThis.crypto.randomUUID();
// Conversations are browser-local history. Only the backend establishes truth.
export const api = {
  listConversations: async (archived = false) => { load(); return conversations.filter(c => !!c.archived === archived); },
  getConversation: async id => structuredClone(find(id)),
  createConversation: async (body = {}) => {
    load(); const now = new Date().toISOString();
    const c = { id: identifier(), title: body.title || "New verification", created_at: now, updated_at: now, pinned: false, archived: false, project_id: null, status: "reviewing", messages: [] };
    conversations.unshift(c); save(); return structuredClone(c);
  },
  updateConversation: async (id, patch) => { const c = find(id); for (const k of ["title", "pinned", "archived", "project_id"]) if (k in patch) c[k] = patch[k]; save(); return structuredClone(c); },
  deleteConversation: async id => { load(); conversations = conversations.filter(c => c.id !== id); save(); return { ok: true }; },
  sendMessage: async (id, content, mode = "standard", signal) => {
    const c = find(id);
    const { data } = await http.post("/api/v1/verify", { user_query: content, generation_mode: "normal", conversation_history: c.messages.slice(-28).map(m => ({ role: m.role, content: m.content })) }, { signal, headers: typeof window !== "undefined" && localStorage.getItem("hg.auth.jwt.v1") ? { Authorization: `Bearer ${localStorage.getItem("hg.auth.jwt.v1")}` } : {} });
    const verification = verificationView(data, mode), now = new Date().toISOString();
    const user_message = { id: identifier(), role: "user", content, created_at: now };
    const assistant_message = { id: identifier(), role: "assistant", content: verification.answer, verification, created_at: now };
    c.messages.push(user_message, assistant_message); c.status = verification.verdict === "supported" ? "verified" : verification.verdict === "failed" ? "failed" : "withheld"; c.updated_at = now; save();
    return { user_message, assistant_message, conversation: structuredClone(c) };
  },
  setFeedback: async (id, msgId, feedback) => { const m = find(id).messages.find(m => m.id === msgId); if (!m) throw new Error("Message not found"); m.feedback = feedback; save(); return { ok: true }; },
  listProjects: async () => { load(); return projects; },
  createProject: async name => { load(); const p = { id: identifier(), name }; projects.push(p); save(); return p; },
  deleteProject: async id => { load(); projects = projects.filter(p => p.id !== id); conversations.forEach(c => { if (c.project_id === id) c.project_id = null; }); save(); return { ok: true }; },
  listEvidence: async () => { load(); return conversations.flatMap(c => c.messages.flatMap(m => (m.verification?.evidence || []).map(e => ({ ...e, conversation_id: c.id, conversation_title: c.title, verdict: m.verification.verdict, created_at: m.created_at })))); },
};
