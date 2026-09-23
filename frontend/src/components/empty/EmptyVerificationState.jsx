import React, { useMemo, useRef, useState } from 'react'
import {
  ArrowUp, ChevronDown, FileSearch, FileText, Globe2, Menu, Paperclip,
  Scale, Settings2, ShieldCheck, Sparkles, SquarePen,
} from 'lucide-react'

const templates = [
  { eyebrow: 'Quick verification', icon: ShieldCheck, title: 'Verify a factual claim', description: 'Check a statement against credible, current sources.', prompt: 'Verify this claim and show the strongest supporting or contradicting evidence: ' },
  { eyebrow: 'Source audit', icon: FileSearch, title: 'Audit an AI answer', description: 'Break an answer into claims and trace every citation.', prompt: 'Audit this AI-generated answer. Extract each factual claim, verify it, and identify unsupported statements: ' },
  { eyebrow: 'Evidence brief', icon: FileText, title: 'Build an evidence brief', description: 'Turn a complex topic into an auditable source map.', prompt: 'Build an evidence brief for this topic using primary sources where possible: ' },
  { eyebrow: 'Contradictions', icon: Scale, title: 'Compare conflicting sources', description: 'Surface disagreement without flattening uncertainty.', prompt: 'Compare the evidence for these conflicting claims and explain which is better supported: ' },
  { eyebrow: 'Freshness check', icon: Globe2, title: 'Check what changed', description: 'Re-verify an older answer against today’s record.', prompt: 'Re-verify this older answer against the latest available authoritative sources: ' },
]

function getGreeting() {
  const hour = new Date().getHours()
  if (hour < 12) return 'Good morning'
  if (hour < 18) return 'Good afternoon'
  return 'Good evening'
}

export function EmptyVerificationState({
  onSendMessage,
  disabled = false,
  onToggleSidebar,
  onOpenSettings,
  onOpenCreateFlow,
  onOpenAuth,
  user,
}) {
  const [inputText, setInputText] = useState('')
  const [deepVerify, setDeepVerify] = useState(true)
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef(null)
  const userName = user?.name ? user.name.split(' ')[0] : 'there'
  const greeting = useMemo(getGreeting, [])

  const handleSend = () => {
    if (!inputText.trim() || disabled) return
    onSendMessage({ text: inputText.trim(), deepVerify })
    setInputText('')
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  const selectTemplate = (prompt) => {
    setInputText(prompt)
    window.requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(prompt.length, prompt.length)
    })
  }

  return (
    <div className="empty-verification-viewport">
      <div className="research-atmosphere" aria-hidden="true">
        <div className="sun-wash" />
        <div className="window-light"><i /><i /><i /><i /><i /><i /></div>
        <div className="paper-grain" />
      </div>

      <header className="home-command-bar" aria-label="Workspace controls">
        <div className="home-command-group">
          <button className="home-icon-button" onClick={onToggleSidebar} aria-label="Open sidebar" title="Open sidebar"><Menu size={17} /></button>
          <button className="home-icon-button" onClick={onOpenSettings} aria-label="Verification settings" title="Verification settings"><Settings2 size={17} /></button>
          <button className="home-icon-button" onClick={onOpenCreateFlow} aria-label="Create verification flow" title="Create verification flow"><SquarePen size={17} /></button>
        </div>
        <div className="workspace-status"><span /> Evidence workspace</div>
      </header>

      <main className="home-stage">
        <section className="hero-workspace" aria-labelledby="home-greeting">
          <button className="plan-pill" type="button" onClick={onOpenAuth}>
            <Sparkles size={12} fill="currentColor" /><span>HalluciGuard Pro</span><span className="plan-pill-divider" /><span>Upgrade</span>
          </button>

          <div className="greeting-block">
            <p className="greeting-kicker">A calm place for hard questions</p>
            <h1 id="home-greeting">{greeting}, {userName}.</h1>
            <p>What would you like to verify?</p>
          </div>

          <div className={`hero-composer ${isFocused ? 'is-focused' : ''}`}>
            <div className="composer-status-line">
              <span className="composer-mode-mark"><ShieldCheck size={13} /> Claim workspace</span>
              <span className="composer-hint">Enter to verify · Shift + Enter for a new line</span>
            </div>
            <textarea
              ref={textareaRef}
              rows={3}
              placeholder="Paste a claim, answer, or source you want to investigate…"
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              aria-label="Claim to verify"
            />
            <div className="composer-footer">
              <div className="composer-left-tools">
                <button className="composer-tool-button" type="button" title="Attach evidence"><Paperclip size={15} /><span>Attach</span></button>
                <button className={`verification-mode ${deepVerify ? 'active' : ''}`} type="button" onClick={() => setDeepVerify((value) => !value)} aria-pressed={deepVerify}>
                  <span className="mode-orb"><span /></span>Deep verify
                </button>
                <button className="source-scope" type="button" title="Choose source scope">All sources <ChevronDown size={12} /></button>
              </div>
              <button className={`composer-submit ${inputText.trim() ? 'is-ready' : ''}`} type="button" onClick={handleSend} disabled={disabled || !inputText.trim()} aria-label="Verify claim">
                <ArrowUp size={17} strokeWidth={2.2} />
              </button>
            </div>
          </div>

          <div className="trust-line" aria-label="Verification process">
            <span>Extract claims</span><i /><span>Trace sources</span><i /><span>Expose contradictions</span><i /><span>Deliver a verdict</span>
          </div>
        </section>
      </main>

      <div className="template-dock">
        <div className="template-heading-row">
          <span>Start with a verification</span>
          <button type="button" onClick={onOpenCreateFlow}>Explore workflows <ArrowUp size={12} className="diagonal-arrow" /></button>
        </div>
        <div className="template-track">
          {templates.map(({ eyebrow, icon: Icon, title, description, prompt }, index) => (
            <button className="verification-template" type="button" key={title} onClick={() => selectTemplate(prompt)} style={{ '--template-delay': `${180 + index * 70}ms` }}>
              <span className="template-topline"><Icon size={14} /><span>{eyebrow}</span><ArrowUp size={12} className="diagonal-arrow" /></span>
              <strong>{title}</strong><small>{description}</small>
            </button>
          ))}
        </div>
      </div>

      <style>{`
        .empty-verification-viewport { position: relative; isolation: isolate; width: 100%; height: 100%; min-height: 620px; overflow: hidden; color: #1d211d; background: #f7f6f0; font-family: "Instrument Sans", ui-sans-serif, system-ui, sans-serif; }
        .research-atmosphere, .research-atmosphere > * { position: absolute; inset: 0; pointer-events: none; }
        .research-atmosphere { z-index: -1; overflow: hidden; background: #f7f6f0; }
        .sun-wash { background: radial-gradient(circle at 48% 33%, rgba(255,255,255,.98) 0 15%, rgba(255,255,255,.62) 42%, transparent 70%), linear-gradient(112deg, rgba(255,255,255,.92), rgba(240,239,230,.46)); }
        .window-light { inset: -32% -18% -20% 18%; filter: blur(18px); opacity: .9; transform: rotate(-9deg); animation: windowDrift 22s ease-in-out infinite alternate; }
        .window-light i { position: absolute; top: -12%; bottom: -12%; width: 5.4%; background: rgba(48,58,46,.075); border-radius: 999px; transform: skewX(-18deg); }
        .window-light i:nth-child(1) { left: 16%; } .window-light i:nth-child(2) { left: 30%; width: 7%; } .window-light i:nth-child(3) { left: 45%; }
        .window-light i:nth-child(4) { left: 59%; width: 8%; } .window-light i:nth-child(5) { left: 75%; } .window-light i:nth-child(6) { left: 88%; width: 7%; }
        .paper-grain { opacity: .022; mix-blend-mode: multiply; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.78' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.8'/%3E%3C/svg%3E"); }
        .home-command-bar { position: absolute; z-index: 5; inset: 0 0 auto; height: 68px; display: flex; align-items: center; justify-content: space-between; padding: 0 26px; }
        .home-command-group { display: flex; align-items: center; gap: 5px; }
        .home-icon-button { width: 34px; height: 34px; border-radius: 999px; color: #666a64; display: grid; place-items: center; transition: color .2s ease, background .2s ease, transform .2s ease; }
        .home-icon-button:hover { color: #171a17; background: rgba(255,255,255,.68); transform: translateY(-1px); }
        .workspace-status { display: flex; align-items: center; gap: 7px; height: 30px; padding: 0 12px; border: 1px solid rgba(32,38,32,.08); border-radius: 999px; background: rgba(255,255,255,.38); color: #777b75; font-size: 11px; backdrop-filter: blur(10px); }
        .workspace-status span { width: 6px; height: 6px; border-radius: 50%; background: #227c68; box-shadow: 0 0 0 4px rgba(34,124,104,.09); }
        .home-stage { position: absolute; inset: 0; display: grid; place-items: center; padding: 78px 24px 190px; }
        .hero-workspace { width: min(720px, 100%); display: flex; flex-direction: column; align-items: center; animation: heroArrive .85s cubic-bezier(.2,.8,.2,1) both; }
        .plan-pill { height: 30px; padding: 0 11px; display: inline-flex; align-items: center; gap: 6px; color: #4e534d; background: rgba(255,255,255,.64); border: 1px solid rgba(34,40,34,.09); border-radius: 999px; box-shadow: 0 8px 24px rgba(38,41,35,.045); backdrop-filter: blur(14px); font-size: 11px; font-weight: 500; }
        .plan-pill:hover { background: rgba(255,255,255,.92); transform: translateY(-1px); } .plan-pill svg { color: #bd851c; }
        .plan-pill-divider { width: 1px; height: 11px; background: rgba(30,35,30,.12); } .plan-pill span:last-child { color: #227c68; }
        .greeting-block { margin: 25px 0 34px; text-align: center; }
        .greeting-kicker { margin-bottom: 9px !important; color: #858981 !important; font-size: 10px !important; font-weight: 500; letter-spacing: .14em; text-transform: uppercase; }
        .greeting-block h1 { margin: 0; color: #181b18; font-family: "Instrument Serif", Georgia, serif; font-size: clamp(43px, 4.2vw, 61px); font-weight: 400; line-height: 1; letter-spacing: -.045em; }
        .greeting-block > p:last-child { margin: 12px 0 0; color: #70746e; font-size: 14px; }
        .hero-composer { width: 100%; min-height: 158px; padding: 15px 16px 13px; display: flex; flex-direction: column; background: rgba(255,255,253,.86); border: 1px solid rgba(30,35,29,.11); border-radius: 17px; box-shadow: 0 22px 60px rgba(37,40,34,.08), 0 3px 9px rgba(37,40,34,.04), inset 0 1px 0 rgba(255,255,255,.85); backdrop-filter: blur(18px) saturate(1.1); transition: transform .3s cubic-bezier(.2,.8,.2,1), box-shadow .3s ease, border-color .3s ease; }
        .hero-composer:hover { border-color: rgba(30,35,29,.17); box-shadow: 0 26px 70px rgba(37,40,34,.095), 0 4px 11px rgba(37,40,34,.045); }
        .hero-composer.is-focused { transform: translateY(-2px); border-color: rgba(34,124,104,.38); box-shadow: 0 0 0 4px rgba(34,124,104,.07), 0 28px 72px rgba(37,40,34,.1); }
        .composer-status-line { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 8px; }
        .composer-mode-mark { display: inline-flex; align-items: center; gap: 6px; color: #227c68; font-size: 10.5px; font-weight: 500; letter-spacing: .025em; }
        .composer-hint { color: #a1a49e; font-size: 10px; }
        .hero-composer textarea { width: 100%; flex: 1; min-height: 62px; border: 0; outline: 0; resize: none; background: transparent; color: #20231f; font: 400 15px/1.52 "Instrument Sans", sans-serif; }
        .hero-composer textarea::placeholder { color: #8d918a; opacity: 1; }
        .composer-footer { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding-top: 8px; }
        .composer-left-tools { display: flex; align-items: center; gap: 7px; }
        .composer-tool-button, .verification-mode, .source-scope { height: 29px; padding: 0 9px; display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; color: #666a64; font-size: 11px; font-weight: 500; transition: background .18s ease, color .18s ease; }
        .composer-tool-button:hover, .source-scope:hover { color: #20241f; background: #f0efe9; } .composer-tool-button span { display: none; }
        .verification-mode { background: #f1f0ea; border: 1px solid transparent; }
        .verification-mode.active { color: #165f51; background: #e7f0ec; border-color: rgba(34,124,104,.13); }
        .mode-orb { width: 14px; height: 14px; padding: 2px; display: grid; place-items: center; border-radius: 50%; border: 1px solid currentColor; }
        .mode-orb span { width: 4px; height: 4px; border-radius: 50%; background: currentColor; box-shadow: 0 0 0 2px rgba(34,124,104,.1); }
        .source-scope { color: #858981; }
        .composer-submit { width: 36px; height: 36px; flex: 0 0 36px; display: grid; place-items: center; border-radius: 50%; color: #a7aaa5; background: #eeede7; transition: transform .22s cubic-bezier(.2,.8,.2,1), color .18s ease, background .18s ease, box-shadow .18s ease; }
        .composer-submit.is-ready { color: #fff; background: #20231f; box-shadow: 0 7px 18px rgba(28,31,27,.19); }
        .composer-submit.is-ready:hover { transform: translateY(-2px) scale(1.03); background: #0d100d; } .composer-submit:disabled { cursor: default; }
        .trust-line { margin-top: 16px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 8px; color: #8d918a; font-size: 10px; letter-spacing: .015em; }
        .trust-line i { width: 2px; height: 2px; border-radius: 50%; background: #a8aba5; }
        .template-dock { position: absolute; inset: auto 0 0; z-index: 2; padding: 0 24px 18px; }
        .template-heading-row { max-width: 1240px; margin: 0 auto 9px; display: flex; justify-content: space-between; align-items: center; color: #737770; font-size: 10.5px; font-weight: 500; }
        .template-heading-row > span { letter-spacing: .08em; text-transform: uppercase; }
        .template-heading-row button { display: inline-flex; align-items: center; gap: 4px; color: #72766f; font-size: 10.5px; } .template-heading-row button:hover { color: #1d211d; }
        .diagonal-arrow { transform: rotate(45deg); }
        .template-track { max-width: 1240px; margin: 0 auto; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 9px; }
        .verification-template { min-width: 0; height: 116px; padding: 13px 14px; display: flex; flex-direction: column; align-items: flex-start; text-align: left; color: #242823; background: rgba(255,255,253,.48); border: 1px solid rgba(31,37,30,.08); border-radius: 13px; backdrop-filter: blur(12px); animation: templateArrive .65s cubic-bezier(.2,.8,.2,1) both; animation-delay: var(--template-delay); transition: transform .25s cubic-bezier(.2,.8,.2,1), background .2s ease, box-shadow .2s ease, border-color .2s ease; }
        .verification-template:hover { transform: translateY(-5px); background: rgba(255,255,253,.91); border-color: rgba(31,37,30,.14); box-shadow: 0 16px 34px rgba(35,38,32,.075); }
        .template-topline { width: 100%; display: flex; align-items: center; gap: 6px; color: #227c68; font-size: 9.5px; font-weight: 500; } .template-topline span { flex: 1; }
        .template-topline .diagonal-arrow { color: #92968f; opacity: 0; transform: translate(-3px, 3px) rotate(45deg); transition: opacity .2s ease, transform .2s ease; }
        .verification-template:hover .template-topline .diagonal-arrow { opacity: 1; transform: translate(0,0) rotate(45deg); }
        .verification-template strong { width: 100%; margin-top: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #242823; font-size: 12px; font-weight: 500; letter-spacing: -.01em; }
        .verification-template small { margin-top: 5px; color: #777b74; font-size: 10px; line-height: 1.38; font-weight: 400; }
        @keyframes heroArrive { from { opacity: 0; transform: translateY(14px) scale(.99); } to { opacity: 1; transform: translateY(0) scale(1); } }
        @keyframes templateArrive { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes windowDrift { to { transform: translate(2.5%, 1.5%) rotate(-7.5deg) scale(1.025); } }
        @media (max-width: 1000px) {
          .template-track { display: flex; overflow-x: auto; margin-inline: -24px; padding: 0 24px 4px; scrollbar-width: none; } .template-track::-webkit-scrollbar { display: none; }
          .verification-template { min-width: 218px; } .template-heading-row { max-width: 720px; }
        }
        @media (max-height: 730px) {
          .home-stage { padding-bottom: 150px; } .greeting-block { margin: 17px 0 22px; } .greeting-kicker, .trust-line { display: none; }
          .hero-composer { min-height: 140px; } .verification-template { height: 98px; } .verification-template small { display: none; }
        }
        @media (max-width: 640px) {
          .empty-verification-viewport { min-height: 560px; overflow-y: auto; } .home-command-bar { height: 58px; padding: 0 14px; } .workspace-status { display: none; }
          .home-stage { position: relative; min-height: calc(100vh - 148px); padding: 86px 17px 40px; } .hero-workspace { width: 100%; }
          .greeting-block { margin: 20px 0 27px; } .greeting-block h1 { font-size: 42px; } .greeting-block > p:last-child { font-size: 13px; }
          .hero-composer { min-height: 165px; border-radius: 15px; } .composer-hint, .source-scope { display: none; } .composer-tool-button span { display: inline; }
          .trust-line { max-width: 320px; } .template-dock { position: relative; padding: 0 17px 16px; } .template-heading-row button { display: none; }
          .template-track { margin-inline: -17px; padding-inline: 17px; }
        }
      `}</style>
    </div>
  )
}
