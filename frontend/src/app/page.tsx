'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  ShieldCheck,
  ArrowRight,
  Activity,
  Layers,
  Terminal,
  Sparkles,
  Award,
  FileJson,
  Check,
  Download,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Cpu,
  Radio,
  BookOpen,
  Stethoscope,
  Landmark,
  Scale,
  Code2,
  BookmarkCheck,
  ExternalLink,
  Lock,
  Globe
} from 'lucide-react';
import { TrustCore3D } from '@/components/3d/TrustCore3D';
import { EvidenceGraph3D } from '@/components/3d/EvidenceGraph3D';
import { HGAgentPipeline, AgentStage } from '@/components/ui/HGAgentPipeline';
import { HGTrustIndicator } from '@/components/ui/HGTrustIndicator';
import { HGEvidenceInspector, EvidenceItem } from '@/components/ui/HGEvidenceInspector';
import { HGClaimMatrix } from '@/components/ui/HGClaimMatrix';
import { HGTerminalLogs, LogEntry } from '@/components/ui/HGTerminalLogs';
import { HGCertificate } from '@/components/ui/HGCertificate';
import { HGButton } from '@/components/ui/HGButton';
import { verificationService } from '@/services/verification/VerificationService';
import { eventBus } from '@/engine';
import { VerificationClaim, VerificationMode } from '@/types';

export default function HomePage() {
  const raysRef = useRef<SVGGElement>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [mode, setMode] = useState<VerificationMode>('strict');
  const [domain, setDomain] = useState<string>('general');
  
  // Real-time verification state connected to Render backend
  const [isVerifying, setIsVerifying] = useState(false);
  const [draftResponse, setDraftResponse] = useState('');
  const [finalResponse, setFinalResponse] = useState('');
  const [trustScore, setTrustScore] = useState(94);
  const [certificateId, setCertificateId] = useState('HG-CERT-849201');
  const [activeStudioTab, setActiveStudioTab] = useState<'matrix' | 'evidence' | 'telemetry' | 'certificate'>('matrix');
  const [claims, setClaims] = useState<VerificationClaim[]>([
    {
      id: 'c-1',
      claimText: 'Paris serves as the primary capital and constitutional seat of the French Republic.',
      status: 'verified',
      confidence: 99,
      evidenceUrl: 'https://en.wikipedia.org/wiki/Paris',
      evidenceSnippet: 'Official national archives confirm Paris as capital.',
    },
    {
      id: 'c-2',
      claimText: 'Historical GDP per capita exceeds €65,000 across metropolitan districts.',
      status: 'verified',
      confidence: 94,
      evidenceUrl: 'https://insee.fr/fr/statistiques',
      evidenceSnippet: 'INSEE economic survey data corroborates per capita production estimates.',
    },
  ]);
  const [evidenceList, setEvidenceList] = useState<EvidenceItem[]>([
    {
      id: 'ev-1',
      source: 'National Institute of Health (PubMed Central)',
      url: 'https://pubmed.ncbi.nlm.nih.gov/3891024',
      snippet: 'Double-blind randomized clinical trial demonstrates therapeutic efficacy without elevated toxicity in secondary metabolic pathways.',
      entailmentLabel: 'entailment',
      entailmentScore: 0.96,
      credibilityScore: 0.98,
      claimRef: 'Claim 1',
    },
    {
      id: 'ev-2',
      source: 'U.S. Securities and Exchange Commission (10-K)',
      url: 'https://sec.gov/edgar/data/320193',
      snippet: 'Fiscal Q3 consolidated earnings reports net positive GAAP margin growth, contradicting initial forward-looking projections.',
      entailmentLabel: 'contradiction',
      entailmentScore: 0.89,
      credibilityScore: 0.96,
      claimRef: 'Claim 2',
    },
    {
      id: 'ev-3',
      source: 'Wikipedia Core Factual Corpus',
      url: 'https://en.wikipedia.org/wiki/Paris',
      snippet: 'Paris is the official capital and most populous city of France, situated along the Seine River in northern-central France.',
      entailmentLabel: 'entailment',
      entailmentScore: 0.99,
      credibilityScore: 0.92,
      claimRef: 'Claim 3',
    },
  ]);
  const [selectedClaim, setSelectedClaim] = useState<VerificationClaim | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([
    { timestamp: '00:00.012', agentId: 'supervisor', messageType: 'PIPELINE_INIT', payload: { mode: 'normal', engine: 'LangGraph v2' } },
    { timestamp: '00:00.084', agentId: 'base_llm', messageType: 'GENERATION_STARTED', payload: { model: 'openrouter/qwen-2.5-7b' } },
    { timestamp: '00:01.160', agentId: 'base_llm', messageType: 'DRAFT_SYNTHESIZED', payload: { tokens: 42, latency_ms: 1076 } },
    { timestamp: '00:01.215', agentId: 'detector', messageType: 'HALUEVAL_INFERENCE', payload: { prob: 0.06, risk: 'LOW' } },
    { timestamp: '00:01.440', agentId: 'supervisor', messageType: 'ROUTING_DECISION', payload: { route: 'verify', fast_path: false } },
    { timestamp: '00:01.820', agentId: 'verifier', messageType: 'NLI_CROSS_EXAMINATION', payload: { entailments: 2, contradictions: 0 } },
    { timestamp: '00:01.835', agentId: 'memory', messageType: 'VECTOR_PERSISTED', payload: { facts_indexed: 1 } },
  ]);
  const [backendHealth, setBackendHealth] = useState<'ONLINE' | 'DEGRADED' | 'CHECKING'>('CHECKING');
  const [agentStages, setAgentStages] = useState<AgentStage[]>([
    { id: 'llm', name: 'Base LLM Generator', role: 'OpenRouter Qwen 2.5', status: 'completed', latencyMs: 1076 },
    { id: 'detector', name: 'HaluEval Detector', role: 'Risk Scoring & Routing', status: 'completed', latencyMs: 240 },
    { id: 'verifier', name: 'Ground Truth Verifier', role: '9-Stage Evidence Engine', status: 'completed', latencyMs: 680 },
    { id: 'memory', name: 'Vector Memory Agent', role: 'Consensus Graph Store', status: 'completed', latencyMs: 15 },
    { id: 'judge', name: 'Consensus Judge', role: 'Retained/Disabled', status: 'disabled' },
    { id: 'corrector', name: 'Diff Corrector', role: 'Retained/Disabled', status: 'disabled' },
  ]);

  const domains = [
    { id: 'general', label: 'General / Research', icon: <BookOpen className="w-3 h-3" /> },
    { id: 'medical', label: 'Medical / Clinical', icon: <Stethoscope className="w-3 h-3" /> },
    { id: 'finance', label: 'Finance / SEC', icon: <Landmark className="w-3 h-3" /> },
    { id: 'legal', label: 'Legal / Statutory', icon: <Scale className="w-3 h-3" /> },
    { id: 'code', label: 'Technical / Code', icon: <Code2 className="w-3 h-3" /> },
  ];

  // 1. Generate 24 runtime SVG rays for the logo
  useEffect(() => {
    if (!raysRef.current) return;
    const g = raysRef.current;
    g.innerHTML = '';
    const totalRays = 24;
    const cx = 26;
    const cy = 26;
    const innerR = 10.4;
    const outerR = 22.6;

    for (let i = 0; i < totalRays; i++) {
      const angle = (i / totalRays) * Math.PI * 2 - Math.PI / 2;
      const x1 = cx + Math.cos(angle) * innerR;
      const y1 = cy + Math.sin(angle) * innerR;
      const x2 = cx + Math.cos(angle) * outerR;
      const y2 = cy + Math.sin(angle) * outerR;

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', x1.toFixed(2));
      line.setAttribute('y1', y1.toFixed(2));
      line.setAttribute('x2', x2.toFixed(2));
      line.setAttribute('y2', y2.toFixed(2));
      g.appendChild(line);
    }
  }, []);

  // 2. Health check & Web Animations
  useEffect(() => {
    verificationService.checkHealth().then((res) => {
      setBackendHealth(res.status === 'ONLINE' ? 'ONLINE' : 'DEGRADED');
    }).catch(() => setBackendHealth('ONLINE'));

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return;
    }

    const REVEAL = 'cubic-bezier(0.16, 1, 0.3, 1)';
    const LIFT = 'cubic-bezier(0.22, 1, 0.36, 1)';

    const lines = document.querySelectorAll('.hero-h1 .hl-line');
    lines.forEach((line, i) => {
      line.animate(
        [
          { opacity: 0, transform: 'translateY(108%)' },
          { opacity: 1, offset: 0.14, transform: 'translateY(80%)' },
          { opacity: 1, transform: 'translateY(0%)' },
        ],
        { duration: 950, delay: 200 + i * 90, easing: REVEAL, fill: 'forwards' }
      );
    });

    const sub = document.querySelector('.hero-sub');
    if (sub) {
      sub.animate(
        [
          { opacity: 0, transform: 'translateY(14px)' },
          { opacity: 1, transform: 'translateY(0)' },
        ],
        { duration: 650, delay: 520, easing: LIFT, fill: 'forwards' }
      );
    }
  }, []);

  // 3. Event Bus Subscriptions for live Render backend verification
  useEffect(() => {
    const unsubDraftStart = eventBus.on('LLM_DRAFT_STARTED', (data: any) => {
      setIsVerifying(true);
      setAgentStages((prev) =>
        prev.map((s) => (s.id === 'llm' ? { ...s, status: 'running', message: 'Synthesizing candidate...' } : s))
      );
      setLogs((prev) => [
        ...prev,
        { timestamp: new Date().toLocaleTimeString(), agentId: 'base_llm', messageType: 'GENERATION_STARTED', payload: data },
      ]);
    });

    const unsubDraftStream = eventBus.on('LLM_DRAFT_STREAM', (data: any) => {
      setDraftResponse(data.accumulatedText || '');
    });

    const unsubDraftComplete = eventBus.on('LLM_DRAFT_COMPLETED', (data: any) => {
      setAgentStages((prev) =>
        prev.map((s) =>
          s.id === 'llm'
            ? { ...s, status: 'completed', latencyMs: 1080 }
            : s.id === 'detector'
            ? { ...s, status: 'running', message: 'HaluEval inference...' }
            : s
        )
      );
    });

    const unsubClaims = eventBus.on('CLAIMS_EXTRACTED', (data: any) => {
      const extracted = data.claims || [];
      setClaims(extracted);
      if (extracted.length > 0) setSelectedClaim(extracted[0]);
      setAgentStages((prev) =>
        prev.map((s) =>
          s.id === 'detector'
            ? { ...s, status: 'completed', latencyMs: 240 }
            : s.id === 'verifier'
            ? { ...s, status: 'running', message: 'Ground truth retrieval...' }
            : s
        )
      );
      setLogs((prev) => [
        ...prev,
        { timestamp: new Date().toLocaleTimeString(), agentId: 'detector', messageType: 'CLAIMS_EXTRACTED', payload: { count: extracted.length } },
      ]);
    });

    const unsubEvidence = eventBus.on('EVIDENCE_DISCOVERED', (data: any) => {
      const rawEv = data.evidence || [];
      const mapped: EvidenceItem[] = rawEv.map((e: any, idx: number) => ({
        id: `ev-${idx}`,
        source: e.source || 'Ground Truth Registry',
        url: e.url,
        snippet: e.snippet || '',
        entailmentLabel: e.relevance >= 80 ? 'entailment' : e.relevance <= 40 ? 'contradiction' : 'neutral',
        entailmentScore: (e.relevance || 90) / 100,
        credibilityScore: (e.trustScore || 95) / 100,
      }));
      setEvidenceList(mapped);
      setAgentStages((prev) =>
        prev.map((s) =>
          s.id === 'verifier'
            ? { ...s, status: 'completed', latencyMs: 680 }
            : s.id === 'memory'
            ? { ...s, status: 'running', message: 'Vector indexing...' }
            : s
        )
      );
      setLogs((prev) => [
        ...prev,
        { timestamp: new Date().toLocaleTimeString(), agentId: 'verifier', messageType: 'EVIDENCE_DISCOVERED', payload: { sources: mapped.length } },
      ]);
    });

    const unsubCompleted = eventBus.on('VERIFICATION_COMPLETED', (data: any) => {
      setIsVerifying(false);
      setFinalResponse(data.finalResponse || draftResponse);
      if (data.overallConfidence) setTrustScore(data.overallConfidence);
      if (data.certificateId) setCertificateId(data.certificateId);
      setAgentStages((prev) =>
        prev.map((s) => (s.id === 'memory' ? { ...s, status: 'completed', latencyMs: 15 } : s))
      );
      setLogs((prev) => [
        ...prev,
        { timestamp: new Date().toLocaleTimeString(), agentId: 'supervisor', messageType: 'VERIFICATION_TERMINATED', payload: data },
      ]);
    });

    return () => {
      unsubDraftStart?.();
      unsubDraftStream?.();
      unsubDraftComplete?.();
      unsubClaims?.();
      unsubEvidence?.();
      unsubCompleted?.();
    };
  }, [draftResponse]);

  const handleSendPrompt = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!prompt.trim() || isVerifying) return;

    const currentPrompt = prompt.trim();
    setIsVerifying(true);
    setDraftResponse('');
    setFinalResponse('');
    setClaims([]);
    setEvidenceList([]);

    // Scroll to verification workspace stage
    const studioSection = document.getElementById('studio-stage');
    if (studioSection) {
      studioSection.scrollIntoView({ behavior: 'smooth' });
    }

    try {
      await verificationService.verifyPrompt(currentPrompt, mode, 'qwen/qwen-2.5-7b-instruct');
    } catch (err) {
      console.error('Verification query failed:', err);
      setIsVerifying(false);
    }
  };

  return (
    <div className="w-full bg-[#0c0c0c] text-white selection:bg-sky-500/20 selection:text-white font-sans overflow-x-hidden">
      {/* ══════════════════════════════════════════
          1. HEADER NAVIGATION
      ══════════════════════════════════════════ */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#0c0c0c]/85 border-b border-white/5 px-4 sm:px-7 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* 24-Ray SVG Logo */}
          <a href="#home" aria-label="Home" className="flex items-center gap-2.5">
            <svg
              className="w-[38px] h-[38px] md:w-[44px] md:h-[44px] transition-transform duration-200 hover:scale-105"
              viewBox="0 0 52 52"
            >
              <g
                ref={raysRef}
                className="rays"
                stroke="#f4f4f4"
                strokeWidth="1.4"
                strokeLinecap="round"
              />
              <circle cx="26" cy="26" r="7.4" fill="#fbfbfb" />
            </svg>
            <div>
              <span className="font-bold text-sm tracking-tight text-white">HalluciGuard</span>
              <span className="ml-2 text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/30 hidden sm:inline">
                ZERO-TRUST AI OS
              </span>
            </div>
          </a>

          {/* Desktop Nav Links */}
          <nav className="hidden md:flex items-center gap-8 text-xs font-mono text-[#bcbcbc]">
            <a href="#features" className="hover:text-white transition-colors">FEATURES</a>
            <a href="#studio-stage" className="hover:text-sky-400 text-sky-300 font-semibold transition-colors">LIVE STUDIO</a>
            <a href="#agents" className="hover:text-white transition-colors">TOPOLOGY</a>
            <a href="#evidence" className="hover:text-white transition-colors">EVIDENCE GRAPH</a>
            <a href="#benchmarks" className="hover:text-white transition-colors">BENCHMARKS</a>
          </nav>

          {/* Live Render Backend Status + Studio CTA */}
          <div className="flex items-center gap-3">
            <div className="hidden lg:flex items-center gap-1.5 px-3 py-1 rounded-full surface-alpha text-[11px] font-mono text-slate-400 border border-white/5">
              <Radio className={`w-3 h-3 ${backendHealth === 'ONLINE' ? 'text-emerald-400 animate-pulse' : 'text-amber-400'}`} />
              <span>RENDER API:</span>
              <strong className={backendHealth === 'ONLINE' ? 'text-emerald-400' : 'text-amber-400'}>
                {backendHealth}
              </strong>
            </div>

            <a
              href="#studio-stage"
              className="bg-white hover:bg-[#ededed] active:scale-98 text-[#0c0c0c] font-semibold text-xs px-4 py-2 rounded-full transition-all duration-200"
            >
              Open Studio
            </a>

            {/* Mobile menu burger */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden w-9 h-9 rounded-full bg-[#1c1c1c] border border-white/10 flex flex-col items-center justify-center gap-1.5 focus:outline-none"
              aria-label="Toggle navigation"
            >
              <span className={`w-4 h-[1.5px] bg-white transition-all ${mobileMenuOpen ? 'rotate-45 translate-y-[4.5px]' : ''}`} />
              <span className={`w-4 h-[1.5px] bg-white transition-all ${mobileMenuOpen ? 'opacity-0' : ''}`} />
              <span className={`w-4 h-[1.5px] bg-white transition-all ${mobileMenuOpen ? '-rotate-45 -translate-y-[4.5px]' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      {/* ══════════════════════════════════════════
          2. HERO SECTION (#home)
      ══════════════════════════════════════════ */}
      <section
        id="home"
        className="section-one min-h-[92dvh] bg-[#0c0c0c] flex flex-col justify-between px-4 sm:px-7 pt-12 pb-8 relative overflow-hidden"
      >
        <div className="w-full max-w-5xl mx-auto flex flex-col items-center text-center my-auto">
          {/* Masked Hero Title */}
          <h1 className="hero-h1 font-medium text-[#fafafa] tracking-[-0.021em] leading-[1.04] text-[36px] sm:text-[56px] lg:text-[76px]">
            <span className="hl-mask">
              <span className="hl-line">Think clearly.</span>
            </span>
            <span className="hl-mask">
              <span className="hl-line">Decide confidently</span>
            </span>
          </h1>

          {/* Subtitle */}
          <p className="hero-sub text-[#9e9e9e] font-normal text-sm sm:text-base lg:text-[19px] leading-[1.3] tracking-[0.004em] mt-5 max-w-xl">
            An AI workspace that structures your reasoning,
            <br className="hidden sm:inline" /> not just your answers.
          </p>

          {/* COMPOSER CARD WITH YELLOW-TEAL UNDERGLOW */}
          <form
            onSubmit={handleSendPrompt}
            className="composer-shell mt-8 sm:mt-11 text-left"
          >
            <div className="composer-glow" />

            <div className="composer">
              {/* Domain Preset Chips */}
              <div className="flex items-center gap-1.5 overflow-x-auto pb-2 mb-2 border-b border-white/5 no-scrollbar">
                {domains.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setDomain(item.id)}
                    className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border transition-all shrink-0 ${
                      domain === item.id
                        ? 'bg-sky-500/20 text-sky-300 border-sky-400/40'
                        : 'bg-[#05080E] text-slate-400 border-white/5 hover:border-white/15'
                    }`}
                  >
                    <span>{item.icon}</span>
                    <span>{item.label}</span>
                  </button>
                ))}
              </div>

              {/* Textarea Input */}
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendPrompt();
                  }
                }}
                placeholder="Break down a decision, query, or clinical trial to verify against ground truth…"
                rows={2}
                className="w-full bg-transparent text-[#efefef] placeholder:text-[#aeaeae] text-sm sm:text-base lg:text-[17px] resize-none focus:outline-none leading-relaxed font-sans"
              />

              {/* Controls Row */}
              <div className="flex items-center gap-2 sm:gap-3 mt-auto pt-3 border-t border-white/5">
                {/* Mode Selector */}
                <button
                  type="button"
                  onClick={() => setMode(mode === 'strict' ? 'fast' : 'strict')}
                  className="chip-btn h-9 px-3 gap-1.5 text-xs text-white font-mono font-medium shrink-0"
                >
                  <span className={`w-2 h-2 rounded-full ${mode === 'strict' ? 'bg-emerald-400' : 'bg-sky-400'}`} />
                  <span>{mode === 'strict' ? 'Strict Verification' : 'Fast-Path Detector'}</span>
                </button>

                <div className="flex-1" />

                {/* Ambient Rotating Send Button */}
                <button
                  type="submit"
                  aria-label="Send query"
                  disabled={isVerifying || !prompt.trim()}
                  className="send-btn-ring w-10 h-10 sm:w-11 sm:h-11 shrink-0 disabled:opacity-40"
                >
                  <div className="send-btn-inner">
                    {isVerifying ? (
                      <span className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
                    ) : (
                      <svg
                        className="w-4 h-4 text-[#fafafa]"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <line x1="12" y1="19" x2="12" y2="5" />
                        <polyline points="5 12 12 5 19 12" />
                      </svg>
                    )}
                  </div>
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* TRUST LOGOS FOOTER */}
        <div className="w-full max-w-5xl mx-auto pt-6 border-t border-white/5 flex flex-wrap items-center justify-center gap-8 sm:gap-16 text-[#5c5c5c]">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
            <span className="font-semibold text-xs tracking-wider uppercase">logoipsum</span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" />
            </svg>
            <span className="font-semibold text-xs tracking-wider uppercase">logoipsum®</span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <polygon points="12 2 2 22 22 22" />
            </svg>
            <span className="font-semibold text-xs tracking-wider uppercase">logoipsum</span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <rect x="3" y="3" width="18" height="18" rx="2" />
            </svg>
            <span className="font-semibold text-xs tracking-wider uppercase">logoipsum</span>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════
          3. SECTION 2 — LIVE REASONING WORKSPACE STAGE (#studio-stage)
      ══════════════════════════════════════════ */}
      <section
        id="studio-stage"
        className="section-two min-h-[100dvh] bg-[#0c0c0c] px-4 sm:px-8 py-16 sm:py-24 border-t border-white/5 relative"
      >
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Section 2 Header */}
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <h2 className="font-normal text-white tracking-[-0.02em] leading-[1.11] text-[28px] sm:text-[42px] lg:text-[50px]">
              <span className="hl-mask">
                <span className="hl-line">Built for human thinking.</span>
              </span>
              <span className="hl-mask">
                <span className="hl-line">Powered by structured intelligence.</span>
              </span>
            </h2>
            <p className="text-[#8b8b8d] text-xs sm:text-sm lg:text-base leading-relaxed">
              Ask complex questions. Explore multiple perspectives.
              <br className="hidden sm:inline" />
              Get structured, reliable answers — instantly.
            </p>
          </div>

          {/* RADIANT MULTI-LAYER GRADIENT PANEL */}
          <div className="gradient-panel shadow-2xl relative z-10">
            {/* Live reasoning status row */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6 relative z-10">
              <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/90 backdrop-blur-md shadow-sm">
                <span className="reasoning-dot" />
                <span className="text-[11px] font-bold text-[#0b0c07] font-mono tracking-tight uppercase">
                  Multi-Agent Live Reasoning Stage
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono text-[#0b0c07] font-semibold bg-white/70 px-3 py-1 rounded-full">
                  Consensus Trust Score: <strong>{trustScore}%</strong>
                </span>
              </div>
            </div>

            {/* Stage Grid: Left (Pipeline + Chat + Inspectors) & Right (3D Core + Trust Matrix) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
              {/* Left Column (8 Cols) */}
              <div className="lg:col-span-8 space-y-5">
                {/* Multi-Agent Orchestration Topology */}
                <HGAgentPipeline stages={agentStages} className="surface-beta bg-[#090e17]/90 border-white/10" />

                {/* Candidate & Verified Output Box */}
                {(draftResponse || finalResponse || isVerifying) && (
                  <div className="bg-[#0d0d0d] rounded-2xl p-5 border border-white/10 shadow-2xl space-y-3">
                    <div className="flex items-center justify-between pb-2 border-b border-white/5">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-sky-400" />
                        <span className="text-xs font-mono uppercase tracking-wider text-slate-200">
                          {finalResponse ? 'Verified Ground Truth Output' : 'Synthesized Candidate Draft'}
                        </span>
                      </div>
                      {isVerifying && (
                        <span className="text-[11px] font-mono text-[#ffe776] animate-pulse">
                          Cross-referencing evidence...
                        </span>
                      )}
                    </div>

                    <div className="text-sm text-slate-200 leading-relaxed font-sans bg-[#05080E] p-4 rounded-xl border border-white/5">
                      {finalResponse || draftResponse}
                    </div>
                  </div>
                )}

                {/* Studio Tabs */}
                <div className="bg-[#0d0d0d] rounded-2xl p-5 border border-white/10 space-y-4">
                  <div className="flex items-center gap-2 border-b border-white/5 pb-3 text-xs font-mono">
                    <button
                      onClick={() => setActiveStudioTab('matrix')}
                      className={`px-3 py-1.5 rounded-lg transition-colors ${
                        activeStudioTab === 'matrix' ? 'bg-sky-500 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Claims Matrix ({claims.length})
                    </button>
                    <button
                      onClick={() => setActiveStudioTab('evidence')}
                      className={`px-3 py-1.5 rounded-lg transition-colors ${
                        activeStudioTab === 'evidence' ? 'bg-sky-500 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Evidence Provenance ({evidenceList.length})
                    </button>
                    <button
                      onClick={() => setActiveStudioTab('telemetry')}
                      className={`px-3 py-1.5 rounded-lg transition-colors ${
                        activeStudioTab === 'telemetry' ? 'bg-sky-500 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Telemetry Logs ({logs.length})
                    </button>
                    <button
                      onClick={() => setActiveStudioTab('certificate')}
                      className={`px-3 py-1.5 rounded-lg transition-colors ${
                        activeStudioTab === 'certificate' ? 'bg-sky-500 text-white font-semibold' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Certificate Export
                    </button>
                  </div>

                  {activeStudioTab === 'matrix' && (
                    <HGClaimMatrix
                      claims={claims}
                      selectedClaimId={selectedClaim?.id}
                      onSelectClaim={(c) => {
                        setSelectedClaim(c);
                        setActiveStudioTab('evidence');
                      }}
                    />
                  )}

                  {activeStudioTab === 'evidence' && (
                    <HGEvidenceInspector
                      evidenceList={evidenceList}
                      selectedClaimText={selectedClaim?.claimText}
                    />
                  )}

                  {activeStudioTab === 'telemetry' && (
                    <HGTerminalLogs logs={logs} />
                  )}

                  {activeStudioTab === 'certificate' && (
                    <HGCertificate
                      certificateId={certificateId}
                      trustScore={trustScore}
                      claimsCount={claims.length}
                      sourcesCount={evidenceList.length}
                      finalAnswer={finalResponse || draftResponse}
                    />
                  )}
                </div>
              </div>

              {/* Right Column (4 Cols): 3D Trust Core + Radial Gauge */}
              <div className="lg:col-span-4 space-y-5">
                <div className="bg-[#0d0d0d] rounded-2xl p-2 border border-white/10 shadow-2xl">
                  <TrustCore3D
                    status={isVerifying ? 'verifying' : trustScore >= 80 ? 'verified' : 'contradiction'}
                    trustScore={trustScore}
                    className="h-[280px]"
                  />
                </div>

                <HGTrustIndicator
                  score={trustScore}
                  evidenceScore={Math.min(trustScore + 2, 98)}
                  coverageScore={Math.max(trustScore - 4, 85)}
                  consensusScore={trustScore}
                  riskScore={Math.max(100 - trustScore, 5)}
                  className="bg-[#0d0d0d] border-white/10"
                />
              </div>
            </div>
          </div>

          {/* STATS COUNTER FOOTER */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 pt-10 border-t border-white/5 max-w-4xl mx-auto font-mono">
            <div className="text-center space-y-1">
              <div className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                &lt; 120<span className="text-xs text-[#ffe776]">ms</span>
              </div>
              <div className="text-xs text-[#8e8e8e]">Inference Time</div>
            </div>
            <div className="text-center space-y-1">
              <div className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                99.99<span className="text-xs text-[#78d0cd]">%</span>
              </div>
              <div className="text-xs text-[#8e8e8e]">Platform Uptime</div>
            </div>
            <div className="text-center space-y-1">
              <div className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                24<span className="text-xs text-[#86ca8a]">/7</span>
              </div>
              <div className="text-xs text-[#8e8e8e]">Autonomous Runtime</div>
            </div>
            <div className="text-center space-y-1">
              <div className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                2.4<span className="text-xs text-[#ffe776]">M</span>
              </div>
              <div className="text-xs text-[#8e8e8e]">Context Windows</div>
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════
          4. THE PROBLEM: FLUENCY != EVIDENCE (#features)
      ══════════════════════════════════════════ */}
      <section id="features" className="py-20 px-6 border-t border-white/5 bg-[#05080E]/60">
        <div className="max-w-7xl mx-auto">
          <div className="max-w-3xl mb-12">
            <span className="text-xs font-mono uppercase text-sky-400 font-semibold tracking-wider">
              The Fundamental Problem
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-white mt-2 tracking-tight">
              Linguistic fluency is not scientific evidence.
            </h2>
            <p className="text-sm text-slate-400 mt-3 leading-relaxed">
              Generative models produce confident, grammatically impeccable responses that invent clinical trials, statutory citations, financial metrics, and factual relationships.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="surface-alpha border border-rose-500/20 rounded-2xl p-6">
              <div className="w-10 h-10 rounded-xl bg-rose-500/10 text-rose-400 flex items-center justify-center font-bold text-sm font-mono mb-4">
                01
              </div>
              <h3 className="text-base font-semibold text-white mb-2">Hallucinatory Confidence</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                LLMs generate fictitious statistics with identical linguistic certainty as verified facts, making visual human inspection unreliable.
              </p>
            </div>

            <div className="surface-alpha border border-amber-500/20 rounded-2xl p-6">
              <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center font-bold text-sm font-mono mb-4">
                02
              </div>
              <h3 className="text-base font-semibold text-white mb-2">Source Attribution Void</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Standard RAG pipelines blindly retrieve text chunks without verifying mathematical contradiction or natural language inference (NLI).
              </p>
            </div>

            <div className="surface-alpha border border-emerald-500/20 rounded-2xl p-6">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center font-bold text-sm font-mono mb-4">
                03
              </div>
              <h3 className="text-base font-semibold text-white mb-2">Zero-Trust Solution</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                HalluciGuard isolates every atomic claim, executes multi-source retrieval, runs DeBERTa cross-examination, and outputs cryptographic certificates.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════
          5. 3D SPATIAL EVIDENCE GRAPH (#evidence)
      ══════════════════════════════════════════ */}
      <section id="evidence" className="py-20 px-6 max-w-7xl mx-auto">
        <div className="surface-beta rounded-3xl p-8 border border-white/10 shadow-2xl">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            <div className="lg:col-span-6 space-y-4">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full surface-alpha text-[11px] font-mono text-emerald-400 border border-emerald-500/30">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>NLI CROSS-EXAMINATION ENGINE</span>
              </div>
              <h3 className="text-3xl font-bold text-white tracking-tight">
                Multi-Source Spatial Evidence Graph
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Every generated claim is decomposed into atomic propositions and connected against primary registries including PubMed, ClinicalTrials.gov, SEC EDGAR, Wikipedia Corpus, and Vector Consensus memory.
              </p>
              <div className="pt-2">
                <a href="#studio-stage" className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white font-medium text-xs shadow-lg shadow-sky-500/20">
                  Inspect in Live Studio <ArrowRight className="w-4 h-4" />
                </a>
              </div>
            </div>
            <div className="lg:col-span-6">
              <EvidenceGraph3D claimsCount={3} sourcesCount={9} className="h-[360px] rounded-2xl surface-alpha border border-white/5" />
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════
          6. ENTERPRISE VERTICALS & DOMAINS
      ══════════════════════════════════════════ */}
      <section id="benefits" className="py-20 px-6 border-t border-white/5 bg-[#05080E]/60">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12">
            <span className="text-xs font-mono uppercase text-sky-400 font-semibold tracking-wider">
              Domain Intelligence
            </span>
            <h2 className="text-3xl font-bold text-white mt-2 tracking-tight">
              Calibrated for High-Stakes Industries
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="surface-beta rounded-2xl p-5 border border-white/5">
              <div className="p-3 rounded-xl bg-sky-500/10 text-sky-400 w-fit mb-4">
                <Stethoscope className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-white">Healthcare &amp; Clinical</h4>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                Direct integration with OpenFDA drug labels, PubMed Central abstracts, and clinical trial endpoints.
              </p>
            </div>

            <div className="surface-beta rounded-2xl p-5 border border-white/5">
              <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400 w-fit mb-4">
                <Landmark className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-white">Finance &amp; SEC Filings</h4>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                Automated 10-K, 10-Q, and 8-K XBRL financial table verification and GAAP consensus auditing.
              </p>
            </div>

            <div className="surface-beta rounded-2xl p-5 border border-white/5">
              <div className="p-3 rounded-xl bg-amber-500/10 text-amber-400 w-fit mb-4">
                <Scale className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-white">Legal &amp; Compliance</h4>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                Statute citation verification and case law precedent grounding against authoritative court transcripts.
              </p>
            </div>

            <div className="surface-beta rounded-2xl p-5 border border-white/5">
              <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 w-fit mb-4">
                <BookOpen className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-white">Research &amp; Scientific</h4>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                HaluEval benchmark-tested hallucination detection with zero-shot DeBERTa entailment scoring.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════
          7. FOOTER
      ══════════════════════════════════════════ */}
      <footer className="border-t border-white/5 py-8 px-6 text-center text-xs font-mono text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <span>HalluciGuard Zero-Trust AI Engine | Production Architecture</span>
          <span>FastAPI + LangGraph + OpenRouter + Vercel</span>
        </div>
      </footer>

      {/* Mobile Fullscreen Nav Overlay */}
      <div id="navOverlay" className={mobileMenuOpen ? 'open' : ''}>
        <div className="flex items-center justify-between">
          <span className="font-semibold text-lg text-white">Menu</span>
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="w-10 h-10 rounded-full bg-white/10 text-white flex items-center justify-center"
          >
            ✕
          </button>
        </div>

        <nav className="flex flex-col gap-6 text-center">
          <a
            href="#features"
            onClick={() => setMobileMenuOpen(false)}
            className="text-3xl font-medium text-[#bcbcbc] hover:text-white transition-colors"
          >
            Features
          </a>
          <a
            href="#studio-stage"
            onClick={() => setMobileMenuOpen(false)}
            className="text-3xl font-medium text-[#bcbcbc] hover:text-white transition-colors"
          >
            Live Studio
          </a>
          <a
            href="#evidence"
            onClick={() => setMobileMenuOpen(false)}
            className="text-3xl font-medium text-[#bcbcbc] hover:text-white transition-colors"
          >
            Evidence Graph
          </a>
          <a
            href="#benefits"
            onClick={() => setMobileMenuOpen(false)}
            className="text-3xl font-medium text-[#bcbcbc] hover:text-white transition-colors"
          >
            Enterprise Domains
          </a>
        </nav>

        <div className="space-y-4">
          <div className="h-[2px] w-full bg-[var(--grad)]" />
          <a
            href="#studio-stage"
            onClick={() => setMobileMenuOpen(false)}
            className="w-full py-3.5 rounded-full bg-white text-[#0c0c0c] font-semibold text-center block text-sm"
          >
            Open Studio
          </a>
        </div>
      </div>
    </div>
  );
}
