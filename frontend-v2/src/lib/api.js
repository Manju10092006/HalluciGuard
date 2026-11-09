import { backendRequest, AUTH_TOKEN_KEY } from "./backend.js";
import { normalizeVerification } from "./verification.js";

// Compatibility adapter for the earlier views. Conversations and projects are
// browser-local; factual results always come from the canonical backend.
function read(key) {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(key) || "[]"); } catch { return []; }
}
function write(key, value) { localStorage.setItem(key, JSON.stringify(value)); }
const key = "hg.legacy.conversations.v2";
const projectsKey = "hg.legacy.projects.v2";
function get(id) {
  const conv = read(key).find(c => c.id === id);
  if (!conv) throw new Error("Conversation not found");
  return conv;
}
function save(conv) {
  write(key, [conv, ...read(key).filter(c => c.id !== conv.id)]);
  return conv;
}
export const api = {
  listConversations: async (archived = false) => read(key).filter(c => !!c.archived === archived),
  getConversation: async id => get(id),
  createConversation: async (body = {}) => save({ id: crypto.randomUUID(), title: body.title || "New verification", created_at: new Date().toISOString(), updated_at: new Date().toISOString(), archived: false, messages: [], ...body }),
  updateConversation: async (id, patch) => save({ ...get(id), ...patch, updated_at: new Date().toISOString() }),
  deleteConversation: async id => { write(key, read(key).filter(c => c.id !== id)); },
  sendMessage: async (id, content) => {
    const conv = get(id);
    const result = await backendRequest("/verify", { method: "POST", token: localStorage.getItem(AUTH_TOKEN_KEY), body: JSON.stringify({ user_query: content, generation_mode: "normal", domain: "general" }) });
    const msg = normalizeVerification(result);
    const user_message = { id: crypto.randomUUID(), role: "user", content, created_at: new Date().toISOString() };
    const assistant_message = { id: msg.id, role: "assistant", content: msg.text, created_at: new Date().toISOString(), verification: {
      verdict: msg.status === "SUPPORTED" ? "supported" : msg.status === "CONTRADICTED" ? "contradicted" : "insufficient_evidence",
      answer: msg.text, explanation: msg.statusDescription, correction: msg.correction?.corrected,
      claims: msg.claims, evidence: msg.sources, pipeline: msg.pipeline, raw: result,
    } };
    const conversation = save({ ...conv, updated_at: new Date().toISOString(), messages: [...conv.messages, user_message, assistant_message] });
    return { user_message, assistant_message, conversation };
  },
  setFeedback: async (id, msgId, feedback) => save({ ...get(id), messages: get(id).messages.map(m => m.id === msgId ? { ...m, feedback } : m) }),
  listProjects: async () => read(projectsKey),
  createProject: async name => { const project = { id: crypto.randomUUID(), name }; write(projectsKey, [...read(projectsKey), project]); return project; },
  deleteProject: async id => { write(projectsKey, read(projectsKey).filter(p => p.id !== id)); },
  listEvidence: async () => read(key).flatMap(c => c.messages.flatMap(m => (m.verification?.evidence || []).map(e => ({ ...e, conversation_id: c.id })))),
};
