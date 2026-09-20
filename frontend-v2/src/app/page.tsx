"use client";

import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/lib/auth/AuthContext'
import { AuthDialog } from '@/components/auth/AuthDialog'
import Lenis from 'lenis'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Matter from 'matter-js'
import {
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  BookOpenCheck,
  Building2,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  FileSearch,
  FileText,
  GitBranch,
  Globe,
  Link2,
  Mail,
  Menu,
  Network,
  Phone,
  RefreshCw,
  Scale,
  ScanSearch,
  Send,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

const capabilities = [
  ['Atomic claim extraction', ScanSearch, 'coral', 'One answer becomes precise, testable statements.'],
  ['Primary-source retrieval', FileSearch, 'blue', 'Evidence is found at passage level, not page level.'],
  ['Claim-source matching', Link2, 'violet', 'Each citation is attached to the statement it addresses.'],
  ['Contradiction detection', AlertTriangle, 'amber', 'Disagreement stays visible instead of being averaged away.'],
  ['Context preservation', FileText, 'mint', 'Qualifiers survive every handoff between agents.'],
  ['Evidence grading', Scale, 'blue', 'Authority, recency, and directness are weighed independently.'],
  ['Multi-agent routing', Network, 'violet', 'The right specialist receives the claim at the right moment.'],
  ['Human escalation', CircleHelp, 'coral', 'Ambiguous cases can stop and ask for judgment.'],
  ['Correction loop', RefreshCw, 'mint', 'A repaired answer is verified again before release.'],
  ['Provenance trail', GitBranch, 'amber', 'Every decision keeps a readable chain of custody.'],
  ['Source quality checks', BookOpenCheck, 'blue', 'Primary records are distinguished from summaries.'],
  ['Auditable verdicts', ShieldCheck, 'violet', 'The conclusion arrives with its reasoning attached.'],
]

const formats = [
  { title: 'Claim dossier', code: 'C—01', copy: 'Atomic statements with the original wording preserved.', tone: 'peach' },
  { title: 'Evidence map', code: 'E—04', copy: 'Passages connected directly to the claims they address.', tone: 'lilac' },
  { title: 'Contradiction brief', code: 'R—02', copy: 'Disagreement isolated without discarding valid context.', tone: 'sky' },
  { title: 'Source lineage', code: 'S—11', copy: 'A visible trail from primary record to final verdict.', tone: 'mint' },
  { title: 'Uncertainty note', code: 'U—03', copy: 'What remains unresolved is stated with precision.', tone: 'butter' },
  { title: 'Correction record', code: 'V—05', copy: 'The repair, the reason, and the re-verification together.', tone: 'rose' },
]

const agents = [
  { number: '01', name: 'Detector', action: 'Breaks the answer into independently checkable claims while preserving the original context.', output: 'Atomic claim dossier', icon: ScanSearch },
  { number: '02', name: 'Verifier', action: 'Finds the strongest passages and tests whether they directly establish each claim.', output: 'Evidence packet', icon: ShieldCheck },
  { number: '03', name: 'Judge', action: 'Weighs support, contradiction, source quality, and uncertainty without hiding disagreement.', output: 'Claim-level verdict', icon: Scale },
  { number: '04', name: 'Corrector', action: 'Repairs unsupported language without changing the parts of the answer that survived scrutiny.', output: 'Corrected response', icon: RefreshCw },
  { number: '05', name: 'Memory Agent', action: 'Carries the verified result and its provenance forward so the same mistake does not return.', output: 'Verified memory', icon: GitBranch },
]

const faqs = [
  {
    q: 'How do I connect my financial data sources?',
    a: 'Connecting your data sources is straightforward. You can use our secure API integrations or pre-built connectors for major financial platforms to import your data in minutes.',
  },
  {
    q: 'Can I change or cancel my plan at any time?',
    a: 'Yes, you can upgrade, downgrade, or cancel your subscription at any time directly from your account dashboard with no hidden fees or lock-in periods.',
  },
  {
    q: 'How secure is my data?',
    a: 'We utilize bank-grade 256-bit encryption for all data in transit and at rest. Your information is isolated and processed strictly within compliant, certified infrastructure.',
  },
  {
    q: 'Does the platform support multiple team members?',
    a: 'Yes, multi-seat collaboration with role-based access control (RBAC) is supported, allowing your team to collaborate seamlessly while maintaining security controls.',
  },
  {
    q: 'What integrations are included?',
    a: 'Out of the box, we support integrations with accounting software, major data warehouses, storage providers, and standard REST APIs.',
  },
  {
    q: 'Do you offer onboarding support?',
    a: 'Yes, all plans include dedicated onboarding documentation and technical support, with custom onboarding assistance available for enterprise accounts.',
  },
]

function useSmoothScroll() {
  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined
    const lenis = new Lenis({ lerp: 0.085, smoothWheel: true, wheelMultiplier: 0.92, syncTouch: false })
    const update = (time: number) => lenis.raf(time * 1000)
    lenis.on('scroll', ScrollTrigger.update)
    gsap.ticker.add(update)
    gsap.ticker.lagSmoothing(0)
    return () => { gsap.ticker.remove(update); lenis.destroy() }
  }, [])
}

function useReveal() {
  useLayoutEffect(() => {
    const ctx = gsap.context(() => gsap.utils.toArray('[data-reveal]').forEach((item: any) => {
      gsap.fromTo(item, { autoAlpha: 0, y: 32 }, { autoAlpha: 1, y: 0, duration: 0.85, ease: 'power3.out', scrollTrigger: { trigger: item, start: 'top 88%', once: true, fastScrollEnd: true } })
    }))
    return () => { ctx.revert(); }
  }, [])
}

function SplitReveal({ children, className = '', as: Tag = 'span', by = 'word' }: { children: string; className?: string; as?: any; by?: string }) {
  const ref = useRef<HTMLElement>(null)
  const parts = useMemo(() => by === 'char' ? Array.from(children) : children.split(/(\s+)/), [children, by])
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return undefined
    const ctx = gsap.context(() => gsap.fromTo(el.querySelectorAll('[data-split-piece]'), { opacity: 0, yPercent: 105, rotateX: -42 }, { opacity: 1, yPercent: 0, rotateX: 0, duration: 0.85, stagger: by === 'char' ? 0.018 : 0.05, ease: 'power3.out', scrollTrigger: { trigger: el, start: 'top 86%', once: true } }), el)
    return () => { ctx.revert(); }
  }, [children, by])
  return <Tag ref={ref} className={`split-reveal ${className}`.trim()}>{parts.map((part: string, index: number) => /^\s+$/.test(part) ? part : <span className="split-mask" key={`${part}-${index}`}><span data-split-piece>{part}</span></span>)}</Tag>
}

function FoldText({ text, className = '' }: { text: string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  useLayoutEffect(() => {
    const pieces = ref.current?.querySelectorAll('i')
    if (!pieces?.length) return undefined
    const tween = gsap.fromTo(pieces, { opacity: 0, rotateX: -88, transformOrigin: '50% 0%' }, { opacity: 1, rotateX: 0, duration: 0.62, stagger: 0.025, ease: 'power3.out' })
    return () => { tween.kill(); }
  }, [text])
  return <span ref={ref} className={`fold-text ${className}`}>{Array.from(text).map((char, i) => <span key={i}><i>{char === ' ' ? '\u00a0' : char}</i></span>)}</span>
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  )
}

function Navbar({ onOpenAuth, authenticated }: { onOpenAuth: () => void; authenticated: boolean }) {
  const [open, setOpen] = useState(false)
  const links: [string, string][] = [
    ['The investigation', '#investigation'],
    ['The agents', '#agents'],
    ['The evidence', '#evidence'],
  ]
  return (
    <header className="nav-wrap">
      <nav className="nav" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="HalluciGuard home">
          <span>HalluciGuard</span>
          <span className="brand-dot" aria-hidden="true" />
        </a>
        <div className="desktop-links">
          {links.map(([label, href], index) => (
            <a className={`nav-link ${index === 0 ? 'active' : ''}`} key={href} href={href}>
              {label}
              {index === 0 && <span className="active-dot" />}
            </a>
          ))}
        </div>
        <button className="nav-cta cursor-pointer" type="button" onClick={onOpenAuth}>
          {authenticated ? 'Open Chat UI' : 'Sign In / Chat'} <ArrowUpRight size={14} />
        </button>
        <button className="menu-button" type="button" aria-label="Toggle navigation" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </nav>
      {open && (
        <div className="mobile-menu">
          {links.map(([label, href]) => <a key={href} href={href} onClick={() => setOpen(false)}>{label}</a>)}
          <button className="text-left font-medium p-3 rounded-lg border-0 bg-transparent text-[#14382e] hover:bg-[#dceae1] cursor-pointer" type="button" onClick={() => { setOpen(false); onOpenAuth(); }}>
            {authenticated ? 'Open Chat UI' : 'Sign In / Chat'}
          </button>
        </div>
      )}
    </header>
  )
}

function RotatingWheel() {
  const ticks = Array.from({ length: 120 }, (_, i) => i)
  return (
    <div className="wheel-container" aria-hidden="true">
      <svg viewBox="0 0 1000 1000" className="rotating-wheel-svg">
        <defs>
          <radialGradient id="wheelGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(255, 255, 255, 0)" />
            <stop offset="50%" stopColor="rgba(215, 238, 228, 0.45)" />
            <stop offset="85%" stopColor="rgba(195, 226, 212, 0.25)" />
            <stop offset="100%" stopColor="rgba(180, 218, 202, 0)" />
          </radialGradient>
        </defs>
        <circle cx="500" cy="500" r="460" fill="url(#wheelGrad)" />
        <circle cx="500" cy="500" r="475" stroke="rgba(41, 109, 91, 0.14)" strokeWidth="1.5" fill="none" />
        <circle cx="500" cy="500" r="435" stroke="rgba(41, 109, 91, 0.16)" strokeWidth="1.5" fill="none" />
        <circle cx="500" cy="500" r="395" stroke="rgba(41, 109, 91, 0.1)" strokeWidth="1" strokeDasharray="4 4" fill="none" />
        <circle cx="500" cy="500" r="350" stroke="rgba(41, 109, 91, 0.08)" strokeWidth="1" fill="none" />
        <g className="wheel-ticks-group">
          {ticks.map((i) => {
            const angle = (i * 360) / 120
            const isMajor = i % 5 === 0
            const tickLen = isMajor ? 28 : 16
            const r1 = 475
            const r2 = 475 - tickLen
            const rad = (angle * Math.PI) / 180
            const x1 = 500 + r1 * Math.cos(rad)
            const y1 = 500 + r1 * Math.sin(rad)
            const x2 = 500 + r2 * Math.cos(rad)
            const y2 = 500 + r2 * Math.sin(rad)
            return (
              <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={isMajor ? "rgba(41, 109, 91, 0.35)" : "rgba(41, 109, 91, 0.18)"}
                strokeWidth={isMajor ? 1.5 : 1}
              />
            )
          })}
        </g>
      </svg>
    </div>
  )
}

function InvestigationVisual() {
  const root = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const hero = root.current?.closest('.hero')
    if (!hero) return undefined
    const ctx = gsap.context(() => {
      gsap.to('.rotating-wheel-svg', { rotate: 38, ease: 'none', scrollTrigger: { trigger: hero, start: 'top top', end: 'bottom top', scrub: 1.3 } })
      gsap.to('.investigation-card', { yPercent: -8, rotate: -1.2, ease: 'none', scrollTrigger: { trigger: hero, start: 'top top', end: 'bottom top', scrub: 1.1 } })
    }, root)
    return () => { ctx.revert(); }
  }, [])
  return (
    <div className="hero-visual-wrapper" ref={root}>
      <RotatingWheel />
      <div className="investigation-card" aria-label="Example claim investigation">
        <div className="visual-toolbar">
          <span className="toolbar-title"><ScanSearch size={15} /> INVESTIGATION 001</span>
          <span className="pause-mark">00</span>
        </div>
        <div className="examination-section">
          <span className="strip-label">AN AI ANSWER, UNDER EXAMINATION</span>
          <p className="examination-quote">
            &ldquo;Apollo 11 landed in <u className="quote-underline">1969.</u><br />
            <mark className="quote-highlight">Buzz Aldrin stepped out first.&rdquo;</mark>
          </p>
        </div>
        <div className="card-divider" />
        <div className="status-row">
          <span className="status-text">Comparing independent evidence</span>
          <span className="status-step">03 / 04</span>
        </div>
        <div className="trace-route">
          <span>Answer</span>
          <span className="route-arrow">→</span>
          <span>Claims</span>
          <span className="route-arrow">→</span>
          <span>Evidence</span>
          <span className="route-arrow">→</span>
          <span>Verdict</span>
        </div>
      </div>
      <p className="demo-note">Animated demonstration · not a live fact-check</p>
    </div>
  )
}

function Hero({ onOpenAuth, authenticated }: { onOpenAuth: () => void; authenticated: boolean }) {
  return (
    <section className="hero" id="top">
      <div className="hero-copy">
        <h1 className="hero-heading">
          <SplitReveal>Don&apos;t trust</SplitReveal><br />
          <SplitReveal>the answer.</SplitReveal><br />
          <SplitReveal className="hero-serif-italic">Trace the evidence.</SplitReveal>
        </h1>
        <p className="hero-intro">
          An answer can sound right. Let&apos;s find out if it is.<br />
          Follow every claim from first question to final verdict.
        </p>
        <div className="hero-actions">
          <button className="button primary-green cursor-pointer border-0" type="button" onClick={onOpenAuth}>
            {authenticated ? 'Open Chat Workspace' : 'Launch Chat Workspace'} <ArrowRight size={16} />
          </button>
          <a className="button light-pill" href="#investigation">
            Follow an Investigation <span className="play-triangle">▶</span>
          </a>
        </div>
      </div>
      <div className="hero-stage">
        <InvestigationVisual />
      </div>
      <div className="hero-footer-left">
        Evidence grounded AI verification
      </div>
      <div className="hero-footer-right">
        Scroll to Discover ↓
      </div>
    </section>
  )
}

function InvestigationSequence() {
  const sectionRef = useRef<HTMLElement>(null)
  const [active, setActive] = useState(0)
  const acts: [string, string, string][] = [
    ['Answer', 'Start with the exact response.', 'The system preserves the original language before analysis begins.'],
    ['Claims', 'Separate what can be checked.', 'One fluent paragraph becomes atomic statements with their context intact.'],
    ['Evidence', 'Bring the record to the claim.', 'Relevant passages arrive with source quality and relationship clearly marked.'],
    ['Verdict', 'Let the evidence have a say.', 'Supported facts survive. Contradictions are corrected. Uncertainty remains visible.'],
  ]
  useLayoutEffect(() => {
    const mm = gsap.matchMedia()
    mm.add('(min-width: 900px) and (prefers-reduced-motion: no-preference)', () => {
      ScrollTrigger.create({ trigger: sectionRef.current, start: 'top top', end: '+=320%', pin: '.sequence-pin', anticipatePin: 1, onUpdate: (self) => setActive(Math.min(3, Math.floor(self.progress * 4))) })
    })
    return () => { mm.revert(); }
  }, [])
  return (
    <section className="sequence-section" id="investigation" ref={sectionRef}>
      <div className="sequence-pin">
        <div className="sequence-heading"><SplitReveal as="h2">A confident answer is only the beginning.</SplitReveal><p>Scroll to open the response and follow what survives.</p></div>
        <div className="sequence-layout">
          <div className="sequence-nav">{acts.map((act, index) => <button key={act[0]} className={active === index ? 'active' : ''} onClick={() => setActive(index)}><span>0{index + 1}</span><b>{act[0]}</b><i /></button>)}</div>
          <div className={`sequence-dossier stage-${active}`}>
            <div className="dossier-top"><span>HG / CASE 001</span><span>{String(active + 1).padStart(2, '0')} — 04</span></div>
            <div className="dossier-scene answer-scene"><span>ORIGINAL RESPONSE</span><blockquote>&ldquo;Apollo 11 landed in 1969. Buzz Aldrin was the first person to step onto the lunar surface.&rdquo;</blockquote></div>
            <div className="dossier-scene claims-scene"><span>CLAIM EXTRACTION</span><div className="claim-chip"><b>C—01</b>Apollo 11 landed in 1969.</div><div className="claim-chip flagged"><b>C—02</b>Buzz Aldrin stepped out first.</div></div>
            <div className="dossier-scene evidence-scene"><span>PRIMARY RECORD / NASA</span><div className="evidence-paper"><FileText size={20} /><p>Neil Armstrong was the first person to step onto the Moon, followed by Buzz Aldrin.</p><small>DIRECT CONTRADICTION · PRIMARY SOURCE</small></div></div>
            <div className="dossier-scene verdict-scene"><span>VERDICT ISSUED</span><div className="verdict-seal"><CheckCircle2 size={30} /><b>CORRECTED</b></div><p>Apollo 11 landed in 1969. <strong>Neil Armstrong</strong> was the first person to step onto the lunar surface.</p></div>
          </div>
          <div className="sequence-copy" key={acts[active][0]}><span>{acts[active][0]}</span><h3>{acts[active][1]}</h3><p>{acts[active][2]}</p><div className="sequence-progress"><i style={{ width: `${(active + 1) * 25}%` }} /></div></div>
        </div>
      </div>
    </section>
  )
}

function AgentSystem() {
  const sectionRef = useRef<HTMLElement>(null)
  const [active, setActive] = useState(0)
  const current = agents[active]
  const ActiveIcon = current.icon
  useLayoutEffect(() => {
    const mm = gsap.matchMedia()
    mm.add('(min-width: 900px) and (prefers-reduced-motion: no-preference)', () => {
      ScrollTrigger.create({ trigger: sectionRef.current, start: 'top top', end: '+=400%', pin: '.agent-pin', anticipatePin: 1, onUpdate: (self) => setActive(Math.min(4, Math.floor(self.progress * 5))) })
    })
    return () => { mm.revert(); }
  }, [])
  return (
    <section className="agents-section-v2" id="agents" ref={sectionRef}><div className="agent-pin">
      <div className="agent-section-head"><SplitReveal as="h2">Five specialists. One continuous evidence trail.</SplitReveal><p>Each scroll step hands the same claim to a new kind of intelligence.</p></div>
      <div className="agent-stage">
        <div className="agent-wheel">{agents.map((agent, index) => { const distance = index - active; const depth = Math.abs(distance); return <button key={agent.name} className={distance === 0 ? 'active' : ''} style={{ transform: `translate(${-depth * depth * 18}px, calc(-50% + ${distance * 92}px)) rotate(${-distance * 2.8}deg) scale(${1 - depth * .08})`, opacity: Math.max(.12, 1 - depth * .22) }} onClick={() => setActive(index)}><span>{agent.number}</span>{agent.name}</button> })}<div className="agent-wheel-focus" /></div>
        <div className="agent-core" key={current.name}><div className="agent-core-meta"><span>{current.number} / 05</span><span>ACTIVE SPECIALIST</span></div><div className="agent-icon"><ActiveIcon size={30} /></div><FoldText text={current.name} /><p>{current.action}</p><div className="agent-handoff"><span>OUTPUT</span><b>{current.output}</b><ArrowRight size={17} /></div></div>
        <div className="agent-route" aria-hidden="true"><span>CLAIM</span>{agents.map((agent, index) => <i key={agent.name} className={index <= active ? 'passed' : ''} />)}<span>MEMORY</span></div>
      </div>
    </div></section>
  )
}

function EvidenceFormats() {
  const motionRows = [formats, [...formats.slice(3), ...formats.slice(0, 3)]]

  return (
    <section className="formats-section-v2" id="evidence">
      <div className="section-shell">
        <div className="research-showcase">
          <div className="research-copy-panel">
            <div>
              <p className="mini-label"><i /> ANY EVIDENCE SHAPE</p>
              <h2>Research</h2>
              <p>Evidence that moves with the claim. Every output stays readable: what was claimed, which passage was found, and what the evidence establishes.</p>
            </div>
            <a className="research-cta" href="#relationships">Explore the records <ArrowUpRight size={17} /></a>
          </div>
          <div className="research-motion-window" aria-label="HalluciGuard evidence records">
            <div className="research-glow research-glow-one" />
            <div className="research-glow research-glow-two" />
            {motionRows.map((row, rowIndex) => (
              <div className={`research-marquee-row ${rowIndex === 1 ? 'reverse' : ''}`} key={`research-row-${rowIndex}`}>
                <div className="research-marquee-track">
                  {[0, 1].map((copyIndex) => (
                    <div className="research-card-set" aria-hidden={copyIndex === 1} key={`research-set-${rowIndex}-${copyIndex}`}>
                      {row.map((item, index) => (
                        <article className={`research-record-card ${item.tone}`} key={`${item.code}-${copyIndex}-${index}`}>
                          <div className="research-record-top"><span>{item.code}</span><FileText size={21} /></div>
                          <h3>{item.title}</h3>
                          <p>{item.copy}</p>
                          <div className="research-record-foot"><span>{String(index + 1).padStart(2, '0')} / 06</span><ArrowUpRight size={16} /></div>
                        </article>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            ))}
            <div className="research-window-caption"><span>LIVE EVIDENCE DESK</span><span>CLAIM → SOURCE → VERDICT</span></div>
          </div>
        </div>
      </div>
    </section>
  )
}

function Capabilities() {
  const capabilityRows = Array.from({ length: 4 }, (_, index) => capabilities.slice(index * 3, index * 3 + 3))

  return (
    <section className="capabilities-section-v2">
      <div className="capability-title-wrap">
        <h2>Everything the agents can do.</h2>
        <p>Twelve coordinated capabilities, continuously moving through one evidence system.</p>
      </div>
      <div className="capability-flow" aria-label="HalluciGuard capabilities">
        {capabilityRows.map((row, rowIndex) => (
          <div className={`capability-marquee-row row-${rowIndex + 1} ${rowIndex % 2 ? 'reverse' : ''}`} key={`capability-row-${rowIndex}`}>
            <div className="capability-marquee-track">
              {[0, 1, 2].map((copyIndex) => (
                <div className="capability-pill-set" aria-hidden={copyIndex > 0} key={`capability-set-${rowIndex}-${copyIndex}`}>
                  {row.map(([title, Icon, tone, copy]: any, index: number) => (
                    <article className={`capability-pill ${tone}`} key={`${title}-${copyIndex}`}>
                      <div className="capability-icon-v2"><Icon size={22} /></div>
                      <div><h3>{title}</h3><p>{copy}</p></div>
                      <span>{String(rowIndex * 3 + index + 1).padStart(2, '0')}</span>
                    </article>
                  ))}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function EvidenceConstellation() {
  const ref = useRef<HTMLElement>(null)
  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ scrollTrigger: { trigger: ref.current, start: 'top 68%', once: true } })
      tl.from('.constellation-claim', { scale: .7, opacity: 0, duration: .7, ease: 'back.out(1.5)' }).from('.source-node', { opacity: 0, scale: .82, y: 24, stagger: .16, duration: .65, ease: 'power3.out' }, '-=.25').from('.constellation-lines path', { strokeDashoffset: 1, duration: .8, stagger: .1, ease: 'power2.inOut' }, '-=.45').from('.constellation-verdict', { opacity: 0, y: 20, duration: .65, ease: 'power3.out' }, '-=.2')
    }, ref)
    return () => { ctx.revert(); }
  }, [])
  return (
    <section className="constellation-section" id="relationships" ref={ref}><div className="section-shell constellation-shell">
      <div className="constellation-copy" data-reveal><SplitReveal as="h2">A citation is a link. The relationship is the proof.</SplitReveal><p>HalluciGuard does not count links. It asks what each passage actually establishes, then keeps disagreement visible.</p><div className="relationship-key"><span><i className="support" />Support</span><span><i className="contradict" />Contradiction</span><span><i className="context" />Context only</span></div></div>
      <div className="constellation-board"><svg className="constellation-lines" viewBox="0 0 800 610" preserveAspectRatio="none" aria-hidden="true"><path pathLength="1" d="M168 126 C276 158 286 248 392 298" /><path pathLength="1" d="M648 118 C548 170 522 234 407 297" /><path pathLength="1" d="M650 468 C540 420 518 360 408 316" /></svg><article className="source-node source-one"><span>S—01 / SUPPORT</span><b>NASA Mission Overview</b><p>Apollo 11 landed in July 1969.</p></article><article className="source-node source-two"><span>S—02 / CONTRADICTION</span><b>Primary mission record</b><p>Neil Armstrong stepped onto the surface first.</p></article><article className="source-node source-three"><span>S—03 / CONTEXT</span><b>Lunar module record</b><p>Buzz Aldrin followed Armstrong onto the Moon.</p></article><div className="constellation-claim"><span>CLAIM C—02</span><blockquote>&ldquo;Buzz Aldrin stepped out first.&rdquo;</blockquote></div><div className="constellation-verdict"><span>VERDICT</span><b>Contradicted</b><p>Correct the person. Preserve the mission and date.</p></div></div>
    </div></section>
  )
}

interface PillConfig {
  text: string
  bg: string
  color: string
  borderColor: string
}

function MatterPhysicsBox({ pills, isLeft }: { pills: PillConfig[]; isLeft: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isVisibleRef = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return

    const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1
    const width = container.clientWidth || 500
    const height = 440

    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { Engine, Bodies, Composite, Mouse, MouseConstraint } = Matter

    const engine = Engine.create({
      gravity: { x: 0, y: 0.95, scale: 0.001 }
    })

    // Static Boundaries (Invisible Walls)
    const wallThick = 100
    const ground = Bodies.rectangle(width / 2, height + wallThick / 2 - 4, width * 2, wallThick, {
      isStatic: true,
      friction: 0.8,
      restitution: 0.35
    })
    const leftWall = Bodies.rectangle(-wallThick / 2 + 4, height / 2, wallThick, height * 2, {
      isStatic: true,
      friction: 0.8,
      restitution: 0.35
    })
    const rightWall = Bodies.rectangle(width + wallThick / 2 - 4, height / 2, wallThick, height * 2, {
      isStatic: true,
      friction: 0.8,
      restitution: 0.35
    })
    const roof = Bodies.rectangle(width / 2, -wallThick / 2 - 250, width * 2, wallThick, {
      isStatic: true
    })

    Composite.add(engine.world, [ground, leftWall, rightWall, roof])

    // Mouse Dragging Constraint
    const mouse = Mouse.create(canvas)
    mouse.pixelRatio = dpr

    const mouseConstraint = MouseConstraint.create(engine, {
      mouse,
      constraint: {
        stiffness: 0.2,
        render: { visible: false }
      }
    })

    if ((mouseConstraint.mouse as any).element) {
      const el = (mouseConstraint.mouse as any).element
      el.removeEventListener('mousewheel', (mouseConstraint.mouse as any).mousewheel)
      el.removeEventListener('DOMMouseScroll', (mouseConstraint.mouse as any).mousewheel)
    }

    Composite.add(engine.world, mouseConstraint)

    // Measure pill widths dynamically using canvas context
    ctx.save()
    ctx.font = '600 13px "DM Sans", system-ui, sans-serif'
    const pillSpecs = pills.map((p) => {
      const textWidth = ctx.measureText(p.text).width
      const w = Math.max(textWidth + 36, 88)
      const h = 38
      return { ...p, w, h }
    })
    ctx.restore()

    let activeBodies: Matter.Body[] = []
    let spawnTimer: NodeJS.Timeout | null = null
    let resetTimer: NodeJS.Timeout | null = null
    let spawnIndex = 0

    const spawnNextPill = () => {
      if (spawnIndex >= pillSpecs.length) {
        // All pills spawned. Schedule reset after 10s
        resetTimer = setTimeout(() => {
          activeBodies.forEach((b) => Composite.remove(engine.world, b))
          activeBodies = []
          spawnIndex = 0
          if (isVisibleRef.current) {
            spawnTimer = setTimeout(spawnNextPill, 200)
          }
        }, 10000)
        return
      }

      const p = pillSpecs[spawnIndex]
      const spawnX = Math.random() * (width - p.w - 40) + p.w / 2 + 20
      const spawnY = -40

      const body = Bodies.rectangle(spawnX, spawnY, p.w, p.h, {
        chamfer: { radius: p.h / 2 },
        restitution: 0.42,
        friction: 0.55,
        frictionAir: 0.012,
        density: 0.002,
        angle: (Math.random() - 0.5) * 0.4
      })

      ;(body as any).pillData = {
        text: p.text,
        bg: p.bg,
        color: p.color,
        borderColor: p.borderColor,
        w: p.w,
        h: p.h
      }

      Composite.add(engine.world, body)
      activeBodies.push(body)
      spawnIndex++

      if (isVisibleRef.current) {
        spawnTimer = setTimeout(spawnNextPill, 220)
      }
    }

    // Animation Loop
    let animId: number
    const render = () => {
      if (!isVisibleRef.current) return

      Engine.update(engine, 1000 / 60)

      ctx.clearRect(0, 0, width * dpr, height * dpr)
      ctx.save()
      ctx.scale(dpr, dpr)

      const allBodies = Composite.allBodies(engine.world)
      for (let i = 0; i < allBodies.length; i++) {
        const b = allBodies[i]
        const pData = (b as any).pillData
        if (!pData) continue

        const { x, y } = b.position
        const angle = b.angle
        const { text, bg, color, borderColor, w, h } = pData

        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(angle)

        // Drop shadow for pills
        ctx.shadowColor = 'rgba(0, 0, 0, 0.07)'
        ctx.shadowBlur = 6
        ctx.shadowOffsetY = 3

        // Rounded pill shape
        ctx.beginPath()
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(-w / 2, -h / 2, w, h, h / 2)
        } else {
          const r = h / 2
          const left = -w / 2
          const top = -h / 2
          ctx.moveTo(left + r, top)
          ctx.lineTo(left + w - r, top)
          ctx.arcTo(left + w, top, left + w, top + h, r)
          ctx.lineTo(left + w, top + h - r)
          ctx.arcTo(left + w, top + h, left + w - r, top + h, r)
          ctx.lineTo(left + r, top + h)
          ctx.arcTo(left, top + h, left, top + h - r, r)
          ctx.lineTo(left, top + r)
          ctx.arcTo(left, top, left + r, top, r)
          ctx.closePath()
        }

        ctx.fillStyle = bg
        ctx.fill()

        ctx.shadowColor = 'transparent'
        ctx.lineWidth = 1.2
        ctx.strokeStyle = borderColor
        ctx.stroke()

        // Text
        ctx.fillStyle = color
        ctx.font = '600 13px "DM Sans", system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(text, 0, 1)

        ctx.restore()
      }

      ctx.restore()

      animId = requestAnimationFrame(render)
    }

    // Scroll Activation Observer
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0]
        if (entry.isIntersecting) {
          if (!isVisibleRef.current) {
            isVisibleRef.current = true
            render()
            if (activeBodies.length === 0 && spawnIndex === 0) {
              spawnNextPill()
            }
          }
        } else {
          isVisibleRef.current = false
          if (spawnTimer) clearTimeout(spawnTimer)
          if (resetTimer) clearTimeout(resetTimer)
        }
      },
      { threshold: 0.15 }
    )

    observer.observe(container)

    return () => {
      isVisibleRef.current = false
      if (spawnTimer) clearTimeout(spawnTimer)
      if (resetTimer) clearTimeout(resetTimer)
      observer.disconnect()
      cancelAnimationFrame(animId)
      Engine.clear(engine)
    }
  }, [pills, isLeft])

  return (
    <div className="matter-canvas-wrapper" ref={containerRef}>
      <canvas className="matter-canvas-element" ref={canvasRef} />
      <span className="matter-interactive-hint">Drag pills to interact</span>
    </div>
  )
}

function MatterComparisonSection() {
  const leftPills: PillConfig[] = [
    { text: "Claim", bg: "#14382e", color: "#ffffff", borderColor: "#0e2922" },
    { text: "Evidence", bg: "#2563eb", color: "#ffffff", borderColor: "#1d4ed8" },
    { text: "Primary Source", bg: "#7c3aed", color: "#ffffff", borderColor: "#6d28d9" },
    { text: "Verifier", bg: "#059669", color: "#ffffff", borderColor: "#047857" },
    { text: "Cross-check", bg: "#0d9488", color: "#ffffff", borderColor: "#0f766e" },
    { text: "NLI", bg: "#f59e0b", color: "#1e1b4b", borderColor: "#d97706" },
    { text: "Confidence", bg: "#10b981", color: "#ffffff", borderColor: "#059669" },
    { text: "Judge", bg: "#4c1d95", color: "#ffffff", borderColor: "#3b0764" },
    { text: "Correction", bg: "#f17f73", color: "#ffffff", borderColor: "#e15b4c" },
    { text: "Re-verification", bg: "#164e63", color: "#ffffff", borderColor: "#083344" },
    { text: "Provenance", bg: "#4f46e5", color: "#ffffff", borderColor: "#4338ca" },
    { text: "Verified", bg: "#2e7d63", color: "#ffffff", borderColor: "#1f5845" },
  ]

  const rightPills: PillConfig[] = [
    { text: "Confident answer", bg: "#94a3b8", color: "#ffffff", borderColor: "#64748b" },
    { text: "Unknown", bg: "#e2e8f0", color: "#334155", borderColor: "#cbd5e1" },
    { text: "Assumption", bg: "#64748b", color: "#ffffff", borderColor: "#475569" },
    { text: "Unsupported", bg: "#475569", color: "#ffffff", borderColor: "#334155" },
    { text: "Missing source", bg: "#cbd5e1", color: "#1e293b", borderColor: "#94a3b8" },
    { text: "Unverified", bg: "#e2e8f0", color: "#475569", borderColor: "#cbd5e1" },
    { text: "Ambiguous", bg: "#78716c", color: "#ffffff", borderColor: "#57534e" },
    { text: "Contradiction", bg: "#a8a29e", color: "#1c1917", borderColor: "#78716c" },
    { text: "Outdated", bg: "#57534e", color: "#ffffff", borderColor: "#44403c" },
    { text: "Risk", bg: "#334155", color: "#ffffff", borderColor: "#1e293b" },
  ]

  return (
    <section className="matter-comparison-section" id="verification-difference">
      <div className="section-shell">
        <div className="matter-comparison-head" data-reveal>
          <span className="section-index">THE VERIFICATION DIFFERENCE</span>
          <h2>An answer is only the beginning.</h2>
          <p>See what happens when every claim is given a chance to prove itself.</p>
        </div>

        <div className="matter-boxes-grid" data-reveal>
          {/* Left Box: With HalluciGuard */}
          <div className="matter-box-container left-box">
            <div className="matter-box-header">
              <span className="matter-box-badge">With HalluciGuard</span>
              <p className="matter-box-subtext">Claims are separated, verified, and traced to evidence.</p>
            </div>
            <MatterPhysicsBox pills={leftPills} isLeft={true} />
          </div>

          {/* Right Box: Without verification */}
          <div className="matter-box-container right-box">
            <div className="matter-box-header">
              <span className="matter-box-badge">Without verification</span>
              <p className="matter-box-subtext">Confident statements can pass through without being checked.</p>
            </div>
            <MatterPhysicsBox pills={rightPills} isLeft={false} />
          </div>
        </div>
      </div>
    </section>
  )
}

function FAQ() {
  const [open, setOpen] = useState(-1)
  const videoSrc = `/help-support.mp4`

  return (
    <section className="faq-section" id="questions">
      <div className="section-shell faq-grid-v2">
        <div className="faq-left">
          <h2>Help and <span className="support-underline">support</span></h2>
          <p className="faq-subtitle">Answers to common questions about setup, pricing, and how everything works.</p>
          <div className="faq-video-container">
            <video src={videoSrc} autoPlay loop muted playsInline preload="metadata" className="faq-video-element" />
          </div>
          <p className="still-questions-label">Still got questions?</p>
          <a className="button dark faq-contact-button" href="#contact">Contact us <ArrowRight size={15} /></a>
        </div>
        <div className="faq-right-card">
          {faqs.map((item, index) => (
            <article className={`faq-item-card ${open === index ? 'open' : ''}`} key={item.q}>
              <button
                className="faq-question-btn"
                onClick={() => setOpen(open === index ? -1 : index)}
                aria-expanded={open === index}
              >
                <span>{item.q}</span>
                <span className="plus-badge">{open === index ? '−' : '+'}</span>
              </button>
              <div className="faq-answer-collapse">
                <p>{item.a}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}

function Contact() {
  const [sent, setSent] = useState(false)
  const submit = (event: React.FormEvent) => {
    event.preventDefault()
    setSent(true)
  }
  const videoSrc = `/contact-support.mp4`

  return (
    <section className="contact-section" id="contact">
      <div className="section-shell">
        <div className="contact-header" data-reveal>
          <h2>Get in <span className="touch-underline">touch</span></h2>
          <p>Reach out to our team at any time for support or questions and we&apos;ll get back to you within 2 business days.</p>
        </div>
        <div className="contact-body">
          <aside className="contact-card-left" data-reveal>
            <div className="info-box">
              <Phone size={18} color="#444" />
              <span>412-483-8261</span>
            </div>
            <div className="info-box">
              <Mail size={18} color="#444" />
              <span>support@zovasaas.com</span>
            </div>
            <div className="info-box">
              <Building2 size={18} color="#444" />
              <span>210 Market St. Suite 402<br />San Francisco, CA</span>
            </div>
            <div className="video-box">
                  <video src={videoSrc} autoPlay loop muted playsInline preload="metadata" className="contact-video-media" />
            </div>
          </aside>
          <form className="contact-card-right" onSubmit={submit} data-reveal>
            <h3>How can we help you today?</h3>
            <div className="form-field">
              <label htmlFor="input-name">Name</label>
              <input id="input-name" name="name" required placeholder="Jane Smith" />
            </div>
            <div className="form-field">
              <label htmlFor="input-email">Email</label>
              <input id="input-email" name="email" type="email" required placeholder="jane@framer.com" />
            </div>
            <div className="form-field">
              <label htmlFor="input-topic">Topic</label>
              <select id="input-topic" name="topic" defaultValue="">
                <option value="" disabled>Select...</option>
                <option>Product demonstration</option>
                <option>Research workflow</option>
                <option>Enterprise verification</option>
                <option>Technical integration</option>
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="input-message">Message</label>
              <textarea id="input-message" name="message" required placeholder="Enter your message" rows={4} />
            </div>
            <button className="contact-submit-button" type="submit">
              {sent ? 'Submitted' : 'Submit'}
            </button>
            {sent && <p className="sent-confirmation">Thank you! We will get back to you within 2 business days.</p>}
          </form>
        </div>
      </div>
    </section>
  )
}

function OrbitingCirclesVisual() {
  const avatars = [
    { id: 1, url: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&auto=format&fit=crop&q=80' },
    { id: 2, url: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&auto=format&fit=crop&q=80' },
    { id: 3, url: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&auto=format&fit=crop&q=80' },
    { id: 4, url: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100&auto=format&fit=crop&q=80' },
    { id: 5, url: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=100&auto=format&fit=crop&q=80' },
    { id: 6, url: 'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=100&auto=format&fit=crop&q=80' },
  ]

  return (
    <div className="orbit-container" aria-hidden="true">
      {/* Dashed Orbital Rings */}
      <div className="orbit-ring ring-outer">
        <div className="orbit-spinner spinner-outer">
          <div className="avatar-node avatar-outer-1">
            <img src={avatars[4].url} alt="" />
          </div>
          <div className="avatar-node avatar-outer-2">
            <img src={avatars[5].url} alt="" />
          </div>
        </div>
      </div>

      <div className="orbit-ring ring-middle">
        <div className="orbit-spinner spinner-middle">
          <div className="avatar-node avatar-mid-1">
            <img src={avatars[2].url} alt="" />
          </div>
          <div className="avatar-node avatar-mid-2">
            <img src={avatars[3].url} alt="" />
          </div>
        </div>
      </div>

      <div className="orbit-ring ring-inner">
        <div className="orbit-spinner spinner-inner">
          <div className="avatar-node avatar-in-1">
            <img src={avatars[0].url} alt="" />
          </div>
          <div className="avatar-node avatar-in-2">
            <img src={avatars[1].url} alt="" />
          </div>
        </div>
      </div>

      {/* Center Brand Badge */}
      <div className="orbit-center">
        <div className="orbit-center-badge">
          <BrandMark />
        </div>
      </div>
    </div>
  )
}

function Footer() {
  const [newsletterEmail, setNewsletterEmail] = useState('')
  const [subscribed, setSubscribed] = useState(false)

  const handleSubscribe = (e: React.FormEvent) => {
    e.preventDefault()
    if (newsletterEmail) {
      setSubscribed(true)
      setNewsletterEmail('')
    }
  }

  return (
    <footer className="v2-footer">
      <div className="section-shell">
        {/* Affiliate / Partner Callout Card */}
        <div className="affiliate-card" data-reveal>
          <div className="affiliate-content">
            <span className="affiliate-tag">Become an Affiliate</span>
            <h2 className="affiliate-heading">Join our Affiliate Program</h2>
            <p className="affiliate-desc">
              Earn up to <strong>$200</strong> with our generous <strong>40% commission</strong> for every sale you drive with your referral link.
            </p>
            <a className="affiliate-cta-btn cursor-pointer" href="#contact">
              Become an affiliate <ArrowUpRight size={16} />
            </a>
          </div>
          <div className="affiliate-visual">
            <OrbitingCirclesVisual />
          </div>
        </div>

        {/* Footer Navigation Grid */}
        <div className="footer-grid-v2" data-reveal>
          {/* Column 1: Brand Info */}
          <div className="footer-col brand-col">
            <a className="brand footer-brand-logo" href="#top">
              <BrandMark />
              <span>HalluciGuard</span>
            </a>
            <p className="footer-tagline">
              The most Powerful AI Verification Platform &amp; Design System for researchers and developers.
            </p>
          </div>

          {/* Column 2: Company */}
          <div className="footer-col">
            <h4 className="footer-col-title">Company</h4>
            <ul className="footer-link-list">
              <li><a href="#investigation">Pricing</a></li>
              <li><a href="#contact">Contact Us</a></li>
              <li><a href="#contact">Become an Affiliate <ArrowUpRight size={13} /></a></li>
              <li><a href="#evidence">Projects <ArrowUpRight size={13} /></a></li>
            </ul>
          </div>

          {/* Column 3: Socials */}
          <div className="footer-col">
            <h4 className="footer-col-title">Socials</h4>
            <ul className="footer-link-list">
              <li><a href="https://behance.net" target="_blank" rel="noopener noreferrer">Behance <ArrowUpRight size={13} /></a></li>
              <li><a href="https://dribbble.net" target="_blank" rel="noopener noreferrer">Dribbble <ArrowUpRight size={13} /></a></li>
              <li><a href="https://twitter.com" target="_blank" rel="noopener noreferrer">Twitter/X <ArrowUpRight size={13} /></a></li>
              <li><a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub <ArrowUpRight size={13} /></a></li>
            </ul>
          </div>

          {/* Column 4: Newsletter */}
          <div className="footer-col newsletter-col">
            <h4 className="footer-col-title">Newsletter</h4>
            <p className="newsletter-desc">
              Receive product updates news, exclusive discounts and early access.
            </p>
            <form className="newsletter-form" onSubmit={handleSubscribe}>
              <div className="newsletter-input-wrap">
                <span className="email-prefix">@</span>
                <input
                  type="email"
                  placeholder="Enter your email..."
                  value={newsletterEmail}
                  onChange={(e) => setNewsletterEmail(e.target.value)}
                  required
                />
                <button type="submit" className="newsletter-submit-btn" aria-label="Subscribe">
                  <ArrowRight size={16} />
                </button>
              </div>
            </form>
            {subscribed && <p className="newsletter-success">Thanks for subscribing!</p>}
          </div>
        </div>

        {/* Sub-footer Bar */}
        <div className="sub-footer-bar">
          <p className="copyright-text">
            &copy; 2025 HalluciGuard &middot; All rights reserved &middot; Made with HalluciGuard
          </p>
          <div className="sub-footer-right">
            <span>Built for HalluciGuard</span>
            <div className="footer-social-icons">
              <a href="https://dribbble.com" target="_blank" rel="noopener noreferrer" aria-label="Dribbble"><Globe size={15} /></a>
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" aria-label="Twitter"><ArrowUpRight size={15} /></a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default function LandingPage() {
  const router = useRouter()
  const { status } = useAuth()
  const [authOpen, setAuthOpen] = useState(false)

  const handleOpenAuth = useCallback(() => {
    if (status === 'authenticated') {
      router.push('/chat')
    } else {
      setAuthOpen(true)
    }
  }, [router, status])

  useSmoothScroll()
  useReveal()

  return (
    <>
      <Navbar onOpenAuth={handleOpenAuth} authenticated={status === 'authenticated'} />
      <main>
        <Hero onOpenAuth={handleOpenAuth} authenticated={status === 'authenticated'} />
        <InvestigationSequence />
        <AgentSystem />
        <EvidenceFormats />
        <Capabilities />
        <EvidenceConstellation />
        <MatterComparisonSection />
        <FAQ />
        <Contact />
      </main>
      <Footer />
      <AuthDialog open={authOpen} onClose={() => setAuthOpen(false)} />
    </>
  )
}
