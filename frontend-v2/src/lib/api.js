import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${API_BASE}/api`;
const http = axios.create({ baseURL: API, timeout: 8000 });
const data = (p) => p.then((r) => r.data);

const SEED_CONVERSATIONS = [
  {
    id: "demo-1",
    title: "Java creator claim",
    created_at: new Date(Date.now() - 3600000 * 24).toISOString(),
    updated_at: new Date(Date.now() - 3600000 * 24).toISOString(),
    pinned: true,
    archived: false,
    project_id: null,
    status: "needs_correction",
    message_count: 2,
    messages: [
      { id: "m1", role: "user", content: "Java was created by Snehith.", created_at: new Date(Date.now() - 3600000 * 24).toISOString() },
      {
        id: "m2",
        role: "assistant",
        content: "That claim doesn't appear to be supported by retrieved evidence.",
        created_at: new Date(Date.now() - 3600000 * 24).toISOString(),
        verification: {
          verdict: "contradicted",
          answer: "That claim doesn't appear to be supported.",
          explanation: "The claim conflicts with retrieved evidence.",
          correction: "Java was created by James Gosling and his team at Sun Microsystems in 1995.",
          original_claim: "Java was created by Snehith.",
          claims: [
            { id: "c1", text: "Java was created by the named person.", status: "contradicted", evidence_ids: ["e1", "e2"] },
            { id: "c2", text: "Java is a programming language.", status: "supported", evidence_ids: ["e1"] }
          ],
          evidence: [
            { id: "e1", source: "Wikipedia", domain: "en.wikipedia.org", title: "Java (programming language)", relationship: "contradicts", excerpt: "Java was originally developed by James Gosling at Sun Microsystems and released in May 1995 as a core component of Sun's Java platform.", url: "https://en.wikipedia.org/wiki/Java_(programming_language)" },
            { id: "e2", source: "Oracle", domain: "oracle.com", title: "History of Java Technology", relationship: "contradicts", excerpt: "The Java language project was initiated in June 1991 by James Gosling, Mike Sheridan, and Patrick Naughton.", url: "https://oracle.com" }
          ],
          pipeline: [
            { agent: "Detector", detail: "Extracted 2 checkable claims" },
            { agent: "Verifier", detail: "Retrieved 2 evidence passages" },
            { agent: "Judge", detail: "Verdict: contradicted" },
            { agent: "Corrector", detail: "Rewrote the unsupported claim" },
            { agent: "ReVerifier", detail: "Correction verified against evidence" },
            { agent: "Memory", detail: "Verification stored to history" }
          ],
          mode: "standard",
          counts: { sources: 2, supporting: 0, contradicting: 2 }
        }
      }
    ]
  },
  {
    id: "demo-2",
    title: "Apollo 11 Moon landing",
    created_at: new Date(Date.now() - 3600000 * 48).toISOString(),
    updated_at: new Date(Date.now() - 3600000 * 48).toISOString(),
    pinned: false,
    archived: false,
    project_id: null,
    status: "verified",
    message_count: 2,
    messages: [
      { id: "m3", role: "user", content: "Apollo 11 landed on the Moon in July 1969 and Neil Armstrong was the first to walk on it.", created_at: new Date(Date.now() - 3600000 * 48).toISOString() },
      {
        id: "m4",
        role: "assistant",
        content: "This claim is consistent with the retrieved evidence.",
        created_at: new Date(Date.now() - 3600000 * 48).toISOString(),
        verification: {
          verdict: "supported",
          answer: "This claim is consistent with the retrieved evidence.",
          explanation: "Multiple independent sources corroborate the claim.",
          correction: null,
          original_claim: null,
          claims: [
            { id: "c3", text: "Apollo 11 landed humans on the Moon in July 1969.", status: "supported", evidence_ids: ["e3", "e4"] },
            { id: "c4", text: "Neil Armstrong was the first person to walk on the lunar surface.", status: "supported", evidence_ids: ["e3"] }
          ],
          evidence: [
            { id: "e3", source: "NASA", domain: "nasa.gov", title: "Apollo 11 Mission Overview", relationship: "supports", excerpt: "Apollo 11 launched on July 16, 1969. Neil Armstrong and Buzz Aldrin landed the Lunar Module Eagle on July 20, 1969.", url: "https://nasa.gov" },
            { id: "e4", source: "Wikipedia", domain: "en.wikipedia.org", title: "Apollo 11", relationship: "supports", excerpt: "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.", url: "https://en.wikipedia.org/wiki/Apollo_11" }
          ],
          pipeline: [
            { agent: "Detector", detail: "Extracted 2 checkable claims" },
            { agent: "Verifier", detail: "Retrieved 2 evidence passages" },
            { agent: "Judge", detail: "Verdict: supported" },
            { agent: "Memory", detail: "Verification stored to history" }
          ],
          mode: "standard",
          counts: { sources: 2, supporting: 2, contradicting: 0 }
        }
      }
    ]
  }
];

let localStore = [...SEED_CONVERSATIONS];
let localProjects = [
  { id: "p1", name: "AI Research", created_at: new Date().toISOString() }
];

// Map the REAL backend POST /verify response to the UI's `verification` shape.
// This is the production path; nothing here fabricates a verdict.
function mapVerifyResponse(resp, content, mode) {
  const vs = String(resp?.verification_status || "unverified").toLowerCase();
  const verdict =
    vs.includes("contradict") || vs.includes("hallucinat") ? "contradicted" :
    vs.includes("conflict") ? "conflicted" :
    (vs.includes("verif") && !vs.includes("unverif")) || vs === "supported" ? "supported" :
    vs.includes("human_review") || vs.includes("agent_failed") || vs.includes("reject") ? "needs_review" :
    "unverified";
  const reports = resp?.verifier?.claim_evidence || [];
  const evidence = [];
  let evSeq = 0;
  const claims = reports.map((r, i) => {
    const evIds = (r.evidence || []).map((e) => {
      const id = e.evidence_id || `e${++evSeq}`;
      const rel = String(e.entailment_label || "").toLowerCase();
      let domain = e.source || "";
      try { domain = new URL(e.url).hostname; } catch {}
      evidence.push({
        id, source: e.source || "Source", domain,
        title: e.title || "Reference passage",
        relationship: rel.includes("entail") || rel.includes("support") ? "supports"
          : rel.includes("contra") ? "contradicts" : "neutral",
        excerpt: e.snippet || "", url: e.url || "",
      });
      return id;
    });
    return {
      id: r.claim_id || `c${i + 1}`,
      text: r.claim_text || content,
      status: String(r.verdict || "unverified").toLowerCase().replace(/^verdictlabel\./, ""),
      evidence_ids: evIds,
    };
  });
  const corr = resp?.corrector || resp?.correction_result || {};
  const pipeline = Object.entries(resp?.agent_outcomes || {})
    .filter(([, o]) => o && o.executed)
    .map(([agent, o]) => ({ agent: agent.replace(/_/g, " "), detail: String(o.status || "completed") }));
  return {
    verdict,
    answer: resp?.final_response || resp?.draft_response || "",
    explanation: resp?.judge?.reason || resp?.judge_result?.reason || resp?.detector_note || "",
    correction: corr.corrected_text || null,
    original_claim: verdict === "contradicted" ? content : null,
    claims,
    evidence,
    pipeline: pipeline.length ? pipeline : [{ agent: "verifier", detail: verdict }],
    mode,
    counts: {
      sources: evidence.length,
      supporting: claims.filter((c) => c.status.includes("verif") && !c.status.includes("unverif")).length,
      contradicting: claims.filter((c) => c.status.includes("contradict")).length,
    },
    verification_status: resp?.verification_status,
  };
}

export const api = {
  listConversations: async (archived = false) => {
    try {
      return await data(http.get("/conversations", { params: { archived } }));
    } catch {
      return localStore
        .filter((c) => Boolean(c.archived) === archived)
        .map(({ messages, ...rest }) => ({ ...rest, message_count: messages ? messages.length : 0 }));
    }
  },
  getConversation: async (id) => {
    try {
      return await data(http.get(`/conversations/${id}`));
    } catch {
      const found = localStore.find((c) => c.id === id);
      if (!found) throw new Error("Conversation not found");
      return found;
    }
  },
  createConversation: async (body = {}) => {
    try {
      return await data(http.post("/conversations", body));
    } catch {
      const newConv = {
        id: `conv-${Date.now()}`,
        title: (body.title || "New verification").slice(0, 80),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        pinned: false,
        archived: false,
        project_id: body.project_id || null,
        status: "reviewing",
        messages: []
      };
      localStore.unshift(newConv);
      return newConv;
    }
  },
  updateConversation: async (id, body) => {
    try {
      return await data(http.patch(`/conversations/${id}`, body));
    } catch {
      const idx = localStore.findIndex((c) => c.id === id);
      if (idx !== -1) {
        localStore[idx] = { ...localStore[idx], ...body, updated_at: new Date().toISOString() };
        return localStore[idx];
      }
      throw new Error("Conversation not found");
    }
  },
  deleteConversation: async (id) => {
    try {
      return await data(http.delete(`/conversations/${id}`));
    } catch {
      localStore = localStore.filter((c) => c.id !== id);
      return { ok: true };
    }
  },
  sendMessage: async (id, content, mode = "standard") => {
    // Call the REAL backend verifier (POST /verify) and render its result.
    // Conversation state is persisted client-side (the backend exposes /verify +
    // /api/history, not a conversation API), but the `verification` is now genuine
    // — it is never fabricated on the production path.
    const genMode = mode === "stress_test" ? "stress_test" : "normal";
    let verification;
    try {
      const resp = await data(
        axios.post(`${API_BASE}/verify`, { user_query: content, generation_mode: genMode }, { timeout: 180000 })
      );
      verification = mapVerifyResponse(resp, content, mode);
    } catch (err) {
      // Production surfaces the real failure rather than inventing a verdict.
      // The legacy regex "mock" is DEV-ONLY (NEXT_PUBLIC_ALLOW_MOCK=1).
      if (process.env.NEXT_PUBLIC_ALLOW_MOCK !== "1") throw err;
      const isContradicted = /java.*(created|invented)|snehith|great wall/i.test(content);
      const verdict = isContradicted ? "contradicted" : "supported";
      verification = {
        verdict,
        answer: isContradicted ? "That claim doesn't appear to be supported." : "This claim is consistent with retrieved evidence.",
        explanation: "[DEV MOCK] backend unavailable; this is NOT a real verification.",
        correction: isContradicted ? "Java was created by James Gosling at Sun Microsystems in 1995." : null,
        original_claim: isContradicted ? content : null,
        claims: [{ id: `c-${Date.now()}`, text: content, status: verdict, evidence_ids: [] }],
        evidence: [],
        pipeline: [{ agent: "mock", detail: "DEV MOCK — backend unavailable" }],
        mode,
        counts: { sources: 0, supporting: verdict === "supported" ? 1 : 0, contradicting: verdict === "contradicted" ? 1 : 0 },
      };
    }
    let conv = localStore.find((c) => c.id === id);
    if (!conv) {
      conv = {
        id, title: content.slice(0, 60), created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(), pinned: false, archived: false,
        project_id: null, status: "reviewing", messages: [],
      };
      localStore.unshift(conv);
    }
    const userMsg = { id: `m-${Date.now()}-u`, role: "user", content, created_at: new Date().toISOString() };
    const assistantMsg = {
      id: `m-${Date.now()}-a`, role: "assistant",
      content: verification.answer || (verification.verdict === "contradicted"
        ? "That claim is not supported by the retrieved evidence."
        : "Verification complete."),
      created_at: new Date().toISOString(),
      verification,
    };
    conv.messages.push(userMsg, assistantMsg);
    conv.status = verification.verdict === "contradicted" ? "needs_correction"
      : verification.verdict === "supported" ? "verified" : "reviewing";
    conv.updated_at = new Date().toISOString();
    return { user_message: userMsg, assistant_message: assistantMsg, conversation: conv };
  },
  setFeedback: async (convId, msgId, feedback) => {
    try {
      return await data(http.patch(`/conversations/${convId}/messages/${msgId}`, { feedback }));
    } catch {
      return { ok: true };
    }
  },
  listProjects: async () => {
    try {
      return await data(http.get("/projects"));
    } catch {
      return localProjects;
    }
  },
  createProject: async (name) => {
    try {
      return await data(http.post("/projects", { name }));
    } catch {
      const p = { id: `proj-${Date.now()}`, name, created_at: new Date().toISOString() };
      localProjects.push(p);
      return p;
    }
  },
  deleteProject: async (id) => {
    try {
      return await data(http.delete(`/projects/${id}`));
    } catch {
      localProjects = localProjects.filter((p) => p.id !== id);
      return { ok: true };
    }
  },
  listEvidence: async () => {
    try {
      return await data(http.get("/evidence"));
    } catch {
      const items = [];
      for (const conv of localStore) {
        for (const msg of conv.messages || []) {
          for (const ev of (msg.verification?.evidence || [])) {
            items.push({ ...ev, conversation_id: conv.id, conversation_title: conv.title, verdict: msg.verification.verdict, created_at: msg.created_at });
          }
        }
      }
      return items;
    }
  }
};
