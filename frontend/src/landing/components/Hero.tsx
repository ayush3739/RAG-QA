import { motion, useReducedMotion } from 'framer-motion'
import {
  ArrowRight, Github, FileText, Globe, Brain, Zap,
  CheckCircle2, LayoutDashboard, MessageSquare, FolderOpen,
  Settings, PanelRight, Plus, Database, Upload, Link2
} from 'lucide-react'

/* ─── Animation helpers ─────────────────────────────────── */
const ease = [0.23, 1, 0.32, 1] as const

const fadeUp = {
  hidden: { opacity: 0, y: 22 },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, ease, delay },
  }),
}

const fadeIn = {
  hidden: { opacity: 0 },
  visible: (delay = 0) => ({
    opacity: 1,
    transition: { duration: 0.6, ease, delay },
  }),
}

/* ─── Tool routing tag types ────────────────────────────── */
interface ToolTag { icon: React.ReactNode; label: string; color: string }

const TOOL_TAGS: Record<string, ToolTag> = {
  rag:    { icon: <Database size={9} />,  label: 'RAG Retrieval',   color: 'text-violet-400 bg-violet-400/10 border-violet-400/25' },
  web:    { icon: <Globe size={9} />,     label: 'Web Search',      color: 'text-sky-400 bg-sky-400/10 border-sky-400/25' },
  reason: { icon: <Brain size={9} />,     label: 'Direct Reasoning',color: 'text-amber-400 bg-amber-400/10 border-amber-400/25' },
  deep:   { icon: <Zap size={9} />,       label: 'Deep Research',   color: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25' },
}

/* ─── Sidebar nav items (matches actual app) ────────────── */
const NAV = [
  { id: 'dashboard', label: 'Dashboard',  icon: LayoutDashboard, count: null },
  { id: 'chats',     label: 'Chats',      icon: MessageSquare,   count: 3    },
  { id: 'documents', label: 'Documents',  icon: FolderOpen,      count: null },
  { id: 'settings',  label: 'Settings',   icon: Settings,        count: null },
]

/* ─── Mock messages ─────────────────────────────────────── */
interface Msg {
  role: 'user' | 'assistant'
  text: string
  toolTags?: ToolTag[]
  citations?: string[]
  routeStatus?: string
  isTyping?: boolean
}

const MESSAGES: Msg[] = [
  {
    role: 'user',
    text: 'Compare the chunking strategies across the uploaded research papers.',
  },
  {
    role: 'assistant',
    routeStatus: 'Deciding route',
    toolTags: [TOOL_TAGS.rag],
    text: 'Using hybrid RAG retrieval across your 3 documents. Found 7 relevant passages on chunking methodologies. Parent-child chunking outperforms fixed-size by 23% on recall…',
    citations: ['Smith et al. 2023, §3.2', 'Chen & Liu 2024, §4.1', 'Zhu 2024, §2'],
  },
  {
    role: 'user',
    text: "What's the latest on embedding models in 2025?",
  },
  {
    role: 'assistant',
    routeStatus: 'Searching web',
    toolTags: [TOOL_TAGS.web, TOOL_TAGS.deep],
    isTyping: true,
    text: '',
  },
]

/* ─── Single chat bubble ────────────────────────────────── */
function ChatBubble({ msg, idx }: { msg: Msg; idx: number }) {
  const reduced = useReducedMotion()

  if (msg.role === 'user') {
    return (
      <motion.div
        custom={0.55 + idx * 0.1}
        variants={fadeUp}
        initial="hidden"
        animate="visible"
        className="flex justify-end"
      >
        <div className="max-w-[78%] rounded-2xl rounded-tr-sm bg-[#5b4fe8] px-3.5 py-2.5 text-[11.5px] leading-relaxed text-white">
          {msg.text}
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      custom={0.55 + idx * 0.1}
      variants={fadeUp}
      initial="hidden"
      animate="visible"
      className="flex flex-col gap-1.5"
    >
      {/* Route status chip */}
      {msg.routeStatus && (
        <div className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-violet-500 animate-pulse" />
          <span className="font-mono text-[9.5px] text-zinc-600">{msg.routeStatus}</span>
        </div>
      )}

      {/* Tool tags */}
      {msg.toolTags && (
        <div className="flex flex-wrap gap-1.5">
          {msg.toolTags.map((t) => (
            <span
              key={t.label}
              className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] font-medium ${t.color}`}
            >
              {t.icon}
              {t.label}
            </span>
          ))}
        </div>
      )}

      {/* Body or typing indicator */}
      {msg.isTyping ? (
        <div className="flex items-center gap-1.5 px-0.5 py-1.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-1.5 w-1.5 rounded-full bg-zinc-600 animate-bounce"
              style={reduced ? {} : { animationDelay: `${i * 0.15}s` }}
            />
          ))}
          <span className="ml-1 text-[10.5px] text-zinc-600">Searching the web…</span>
        </div>
      ) : (
        <p className="text-[11.5px] leading-relaxed text-zinc-300">{msg.text}</p>
      )}

      {/* Citations */}
      {msg.citations && (
        <div className="mt-0.5 flex flex-wrap gap-1.5">
          {msg.citations.map((c) => (
            <span
              key={c}
              className="inline-flex items-center gap-1 rounded-md border border-white/[0.07] bg-white/[0.03] px-2 py-0.5 text-[9px] text-zinc-500"
            >
              <CheckCircle2 size={8} className="text-emerald-500 shrink-0" />
              {c}
            </span>
          ))}
        </div>
      )}
    </motion.div>
  )
}

/* ─── The app mockup ────────────────────────────────────── */
function AppMockup() {
  return (
    <div className="flex h-full divide-x divide-white/[0.06]">
      {/* ── Left sidebar (matches actual app) ── */}
      <aside className="hidden w-[200px] flex-col bg-[#08090c] p-4 md:flex">
        {/* Brand */}
        <div className="mb-5 flex items-center gap-2 px-1">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#5b4fe8]">
            <Database size={13} className="text-white" />
          </div>
          <span className="text-[12px] font-semibold tracking-tight text-white">DocuMind</span>
        </div>

        {/* New Chat button */}
        <button className="mb-4 flex items-center gap-2 rounded-xl border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-[11px] text-zinc-400 transition-colors hover:text-white">
          <Plus size={12} />
          New Chat
          <span className="ml-auto font-mono text-[9px] text-zinc-700">⌘K</span>
        </button>

        {/* Nav items */}
        <nav className="flex flex-col gap-0.5">
          {NAV.map((item) => {
            const Icon = item.icon
            const isActive = item.id === 'chats'
            return (
              <div
                key={item.id}
                className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[11.5px] transition-colors ${
                  isActive
                    ? 'bg-white/[0.06] text-white'
                    : 'text-zinc-600'
                }`}
              >
                <Icon size={13} className={isActive ? 'text-violet-400' : ''} />
                {item.label}
                {item.count != null && (
                  <span className="ml-auto flex h-4 min-w-4 items-center justify-center rounded-full bg-[#5b4fe8] px-1 text-[9px] font-medium text-white">
                    {item.count}
                  </span>
                )}
              </div>
            )
          })}
        </nav>

        {/* User chip at bottom */}
        <div className="mt-auto flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.03] px-2.5 py-2">
          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#5b4fe8] text-[10px] font-bold text-white">
            U
          </div>
          <div className="min-w-0">
            <p className="truncate text-[10.5px] font-medium text-zinc-300">User</p>
            <p className="truncate text-[9px] text-zinc-600">No email</p>
          </div>
        </div>
      </aside>

      {/* ── Main chat area ── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Chat top bar */}
        <div className="flex h-10 shrink-0 items-center gap-3 border-b border-white/[0.06] px-4">
          <span className="text-[12px] font-medium text-zinc-200">New Chat</span>
          <span className="rounded-full border border-white/[0.08] bg-white/[0.04] px-2 py-0.5 text-[9.5px] text-zinc-600">0 Docs</span>
          <div className="ml-auto flex items-center gap-2">
            <span className="font-mono text-[9.5px] text-zinc-700">Inspector</span>
            <PanelRight size={12} className="text-zinc-700" />
          </div>
        </div>

        {/* Messages */}
        <div className="flex flex-1 flex-col gap-4 overflow-hidden px-5 py-4">
          {MESSAGES.map((msg, i) => (
            <ChatBubble key={i} msg={msg} idx={i} />
          ))}
        </div>

        {/* Input bar */}
        <div className="shrink-0 border-t border-white/[0.06] px-4 py-3">
          <div className="flex items-center gap-2 rounded-2xl border border-white/[0.08] bg-white/[0.03] px-4 py-2.5">
            <span className="flex-1 text-[11.5px] text-zinc-700">Ask anything…</span>
            {/* Mode badge */}
            <span className="rounded-full border border-white/[0.08] bg-white/[0.04] px-2 py-0.5 text-[9px] font-medium text-zinc-500">
              AUTO
            </span>
            <button className="rounded-lg bg-[#5b4fe8] p-1.5">
              <ArrowRight size={11} className="text-white" />
            </button>
          </div>
        </div>
      </main>

      {/* ── Right inspector panel ── */}
      <aside className="hidden w-[180px] flex-col bg-[#08090c] p-3 lg:flex">
        <p className="mb-3 px-1 text-[9.5px] font-semibold uppercase tracking-widest text-zinc-700">Add Sources</p>

        {/* Upload / Link buttons */}
        <div className="mb-4 grid grid-cols-2 gap-2">
          <button className="flex flex-col items-center gap-1.5 rounded-xl border border-white/[0.07] bg-white/[0.02] py-3 text-[9.5px] text-zinc-500 transition-colors hover:text-zinc-300">
            <Upload size={14} />
            Upload File
          </button>
          <button className="flex flex-col items-center gap-1.5 rounded-xl border border-white/[0.07] bg-white/[0.02] py-3 text-[9.5px] text-zinc-500 transition-colors hover:text-zinc-300">
            <Link2 size={14} />
            Link Existing
          </button>
        </div>

        <p className="mb-2 px-1 text-[9px] font-semibold uppercase tracking-widest text-zinc-700">Active Documents (0)</p>
        <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-white/[0.06] py-6">
          <p className="text-center text-[10px] text-zinc-700">No documents<br />uploaded.</p>
        </div>
      </aside>
    </div>
  )
}

/* ─── Hero ──────────────────────────────────────────────── */
export default function Hero() {
  return (
    <section
      id="hero"
      className="relative flex min-h-screen flex-col items-center bg-[#060608]"
    >
      {/* ── Butter-smooth ambient hero spotlight (zero hard lines or banding) ── */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 -translate-x-1/2 top-0 h-[500px] w-[850px] -translate-y-1/4 rounded-full bg-gradient-to-b from-violet-600/30 via-indigo-900/15 to-transparent blur-[120px]"
      />
      {/* Star-dot pattern */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 opacity-[0.015]"
        style={{
          backgroundImage: 'radial-gradient(circle, #ffffff 1px, transparent 1px)',
          backgroundSize: '44px 44px',
        }}
      />

      {/* ── Text content ── */}
      <div className="relative z-10 mx-auto flex w-full max-w-4xl flex-col items-center px-6 pt-36 pb-10 text-center">
        {/* Editorial Badge Pill */}
        <motion.div
          custom={0}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="mb-8 inline-flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-3.5 py-1 backdrop-blur-md transition-all hover:bg-white/[0.06]"
        >
          <span
            className="h-1.5 w-1.5 rounded-full bg-violet-400"
            style={{ animation: 'pulse 2s infinite' }}
          />
          <span className="text-[11.5px] font-medium text-zinc-200">
            Agentic RAG
          </span>
          <span className="text-[10px] text-zinc-600 font-light">/</span>
          <span className="text-[11.5px] text-zinc-400">
            Tool-routing
          </span>
          <span className="text-[10px] text-zinc-600 font-light">/</span>
          <span className="text-[11.5px] text-zinc-400">
            Open source
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          custom={0.1}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="text-balance text-5xl font-semibold leading-[1.04] text-white md:text-[68px] lg:text-[76px]"
          style={{
            fontFamily: "'DM Sans', -apple-system, sans-serif",
            letterSpacing: '-0.035em',
            textWrap: 'balance',
          }}
        >
          The RAG workspace that{' '}
          <span style={{ color: '#818CF8' }}>
            thinks before it searches.
          </span>
        </motion.h1>

        {/* Sub-line with higher contrast weight & looser leading */}
        <motion.p
          custom={0.22}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="mt-6 max-w-[540px] text-[15px] md:text-[16.5px] font-normal leading-[1.65] text-zinc-400"
        >
          Agentic tool-routing decides between document retrieval, web search, deep research,
          and direct reasoning — each answer grounded with citations.
        </motion.p>

        {/* CTAs */}
        <motion.div
          custom={0.34}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          <div className="group relative rounded-full p-[1px] shadow-[0_0_20px_rgba(99,102,241,0.2)] transition-all duration-300 hover:shadow-[0_0_30px_rgba(99,102,241,0.35)] hover:scale-[1.02] active:scale-[0.98]">
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400/80 via-purple-400/80 to-blue-400/80 opacity-80 transition-opacity duration-300 group-hover:opacity-100" />
            <button
              onClick={() => { window.location.hash = '#/chat/new' }}
              className="relative flex items-center gap-2 rounded-full bg-[#060608]/90 px-6 py-2.5 text-sm font-medium backdrop-blur-md transition-colors hover:bg-[#060608]/70"
            >
              <span className="bg-gradient-to-r from-blue-100 to-purple-200 bg-clip-text text-transparent">
                Launch workspace
              </span>
              <ArrowRight className="h-4 w-4 text-purple-200 transition-transform duration-200 group-hover:translate-x-0.5" />
            </button>
          </div>
          <a
            href="https://github.com/ayush3739/RAG-QA"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-full border border-white/[0.08] px-6 py-2.5 text-sm font-medium text-zinc-400 backdrop-blur-sm transition-all duration-200 hover:border-white/[0.2] hover:bg-white/[0.04] hover:text-white hover:scale-[1.02] active:scale-[0.98]"
          >
            <Github className="h-4 w-4" />
            View on GitHub
          </a>
        </motion.div>

        {/* Trust line */}
        <motion.p
          custom={0.44}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          className="mt-5 text-[11px] text-zinc-700"
        >
          No credit card required · Open source · MIT licensed
        </motion.p>
      </div>

      {/* Bottom fade into next section for cohesive scroll transition */}
      <div 
        className="pointer-events-none absolute bottom-0 left-0 right-0 h-64 z-20"
        style={{ background: 'linear-gradient(to bottom, transparent 0%, #09090b 100%)' }} 
      />

      {/* ── Product mockup ── */}
      <motion.div
        custom={0.52}
        variants={fadeUp}
        initial="hidden"
        animate="visible"
        className="relative z-10 mx-auto w-full max-w-[1100px] px-4 pb-20"
      >
        {/* ── Premium Product Horizon Glow (tightly anchored to card rim) ── */}
        {/* Soft tight ambient bloom strictly behind top horizon rim */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-7 left-1/2 -translate-x-1/2 h-[90px] w-[75%] max-w-[750px] rounded-full bg-gradient-to-b from-violet-500/30 via-indigo-600/10 to-transparent blur-[25px]"
        />

        {/* Crisp hairline horizon light on top edge */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-[1px] left-1/2 -translate-x-1/2 h-[1px] w-[60%] max-w-[600px] bg-gradient-to-r from-transparent via-violet-300/70 to-transparent"
        />

        {/* Browser chrome with crisp 1px light border separating object from glow */}
        <div className="relative overflow-hidden rounded-[14px] border border-white/[0.08] border-t-white/[0.18] bg-[#0d0d10] shadow-[0_30px_90px_rgba(0,0,0,0.8)]">
          {/* Title bar */}
          <div className="flex h-10 items-center border-b border-white/[0.06] bg-[#09090d] px-4">
            <div className="flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
              <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
              <span className="h-3 w-3 rounded-full bg-[#28c840]" />
            </div>
            <div className="mx-auto flex h-5 w-[200px] items-center justify-center gap-1.5 rounded-md border border-white/[0.07] bg-white/[0.03] px-3">
              <span className="text-[10px] text-zinc-600">app.documind.ai</span>
            </div>
          </div>

          {/* App */}
          <div className="h-[480px] md:h-[540px]">
            <AppMockup />
          </div>
        </div>
      </motion.div>
    </section>
  )
}
