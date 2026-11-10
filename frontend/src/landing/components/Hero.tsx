import { navigate } from '../../lib/navigation'
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
  rag:    { icon: <Database size={9} aria-hidden="true" />,  label: 'RAG Retrieval',   color: 'text-primary bg-primary/10 border-primary/25' },
  web:    { icon: <Globe size={9} aria-hidden="true" />,     label: 'Web Search',      color: 'text-secondary bg-secondary/10 border-secondary/25' },
  reason: { icon: <Brain size={9} aria-hidden="true" />,     label: 'Direct Reasoning',color: 'text-warning bg-warning-subtle border-warning-subtle' },
  deep:   { icon: <Zap size={9} aria-hidden="true" />,       label: 'Deep Research',   color: 'text-success bg-success-subtle border-success-subtle' },
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
        <div className="max-w-[78%] rounded-2xl rounded-tr-sm bg-primary px-3.5 py-2.5 text-[11.5px] leading-relaxed text-primary-foreground">
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
          <span className="h-1.5 w-1.5 rounded-full bg-primary motion-safe:animate-pulse" />
          <span className="font-mono text-[9.5px] text-muted-foreground">{msg.routeStatus}</span>
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
              className="h-1.5 w-1.5 rounded-full bg-muted-foreground motion-safe:animate-pulse"
              style={reduced ? {} : { animationDelay: `${i * 0.15}s` }}
            />
          ))}
          <span className="ml-1 text-[10.5px] text-muted-foreground">Searching the web…</span>
        </div>
      ) : (
        <p className="text-[11.5px] leading-relaxed text-foreground">{msg.text}</p>
      )}

      {/* Citations */}
      {msg.citations && (
        <div className="mt-0.5 flex flex-wrap gap-1.5">
          {msg.citations.map((c) => (
            <span
              key={c}
              className="inline-flex items-center gap-1 rounded-md border border-border bg-surface-container px-2 py-0.5 text-[9px] text-muted-foreground"
            >
              <CheckCircle2 size={8} className="text-success shrink-0" aria-hidden="true" />
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
    <div className="flex h-full divide-x divide-border">
      {/* ── Left sidebar (matches actual app) ── */}
      <aside className="hidden w-[200px] flex-col bg-surface-container-low p-4 md:flex">
        {/* Brand */}
        <div className="mb-5 flex items-center gap-2 px-1">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary">
            <Database size={13} className="text-primary-foreground" aria-hidden="true" />
          </div>
          <span className="text-[12px] font-semibold tracking-tight text-foreground">KnowVex</span>
        </div>

        {/* New Chat button */}
        <button
          type="button"
          className="mb-4 flex items-center gap-2 rounded-xl border border-border bg-surface-container px-3 py-2 text-[11px] text-muted-foreground transition-colors duration-150 hover:text-foreground active:scale-[0.98] disabled:opacity-50"
        >
          <Plus size={12} aria-hidden="true" />
          New Chat
          <span className="ml-auto font-mono text-[9px] text-muted-foreground">⌘K</span>
        </button>

        {/* Nav items */}
        <nav className="flex flex-col gap-0.5">
          {NAV.map((item) => {
            const Icon = item.icon
            const isActive = item.id === 'chats'
            return (
              <div
                key={item.id}
                className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[11.5px] transition-colors duration-150 ${
                  isActive
                    ? 'bg-surface-container-high text-foreground'
                    : 'text-muted-foreground'
                }`}
              >
                <Icon size={13} className={isActive ? 'text-primary' : ''} aria-hidden="true" />
                {item.label}
                {item.count != null && (
                  <span className="ml-auto flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[9px] font-medium text-primary-foreground">
                    {item.count}
                  </span>
                )}
              </div>
            )
          })}
        </nav>

        {/* User chip at bottom */}
        <div className="mt-auto flex items-center gap-2 rounded-lg border border-border bg-surface-container px-2.5 py-2">
          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground">
            U
          </div>
          <div className="min-w-0">
            <p className="truncate text-[10.5px] font-medium text-foreground">User</p>
            <p className="truncate text-[9px] text-muted-foreground">No email</p>
          </div>
        </div>
      </aside>

      {/* ── Main chat area ── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Chat top bar */}
        <div className="flex h-10 shrink-0 items-center gap-3 border-b border-border px-4">
          <span className="text-[12px] font-medium text-foreground">New Chat</span>
          <span className="rounded-full border border-border bg-surface-container px-2 py-0.5 text-[9.5px] text-muted-foreground">0 Docs</span>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[9.5px] font-medium text-muted-foreground">Inspector</span>
            <PanelRight size={12} className="text-muted-foreground" aria-hidden="true" />
          </div>
        </div>

        {/* Messages */}
        <div className="flex flex-1 flex-col gap-4 overflow-hidden px-5 py-4">
          {MESSAGES.map((msg, i) => (
            <ChatBubble key={i} msg={msg} idx={i} />
          ))}
        </div>

        {/* Input bar */}
        <div className="shrink-0 border-t border-border px-4 py-3">
          <div className="flex items-center gap-2 rounded-2xl border border-border bg-surface-container px-4 py-2.5">
            <span className="flex-1 text-[11.5px] text-muted-foreground">Ask anything…</span>
            {/* Mode badge */}
            <span className="rounded-full border border-border bg-surface-container-high px-2 py-0.5 text-[9px] font-medium text-muted-foreground">
              AUTO
            </span>
            <button
              type="button"
              className="rounded-lg bg-primary p-1.5 active:scale-[0.98] disabled:opacity-50"
            >
              <ArrowRight size={11} className="text-primary-foreground" aria-hidden="true" />
            </button>
          </div>
        </div>
      </main>

      {/* ── Right inspector panel ── */}
      <aside className="hidden w-[180px] flex-col bg-surface-container-low p-3 lg:flex">
        <p className="mb-3 px-1 text-[10px] font-medium text-muted-foreground">Add sources</p>

        {/* Upload / Link buttons */}
        <div className="mb-4 grid grid-cols-2 gap-2">
          <button
            type="button"
            className="flex flex-col items-center gap-1.5 rounded-xl border border-border bg-surface-container py-3 text-[9.5px] text-muted-foreground transition-colors duration-150 hover:text-foreground active:scale-[0.98] disabled:opacity-50"
          >
            <Upload size={14} aria-hidden="true" />
            Upload File
          </button>
          <button
            type="button"
            className="flex flex-col items-center gap-1.5 rounded-xl border border-border bg-surface-container py-3 text-[9.5px] text-muted-foreground transition-colors duration-150 hover:text-foreground active:scale-[0.98] disabled:opacity-50"
          >
            <Link2 size={14} aria-hidden="true" />
            Link Existing
          </button>
        </div>

        <p className="mb-2 px-1 text-[10px] font-medium text-muted-foreground">Active documents (0)</p>
        <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-border py-6">
          <p className="text-center text-[10px] text-muted-foreground">No documents<br />uploaded.</p>
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
      className="relative flex min-h-dvh flex-col items-center bg-background overflow-hidden"
    >
      {/* ── Butter-smooth ambient hero spotlight (little blue halo) ── */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 -translate-x-1/2 top-0 h-[500px] w-[850px] -translate-y-1/4 rounded-full bg-gradient-to-b from-blue-500/20 via-primary/10 to-transparent blur-[120px]"
      />

      {/* ── Text content ── */}
      <div className="relative z-10 mx-auto flex w-full max-w-4xl flex-col items-center px-6 pt-32 pb-10 text-center">
        {/* Editorial Badge Pill */}
        <motion.div
          custom={0}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="mb-8 inline-flex items-center gap-2 rounded-full border border-border bg-surface-container px-3.5 py-1 text-xs text-foreground transition-colors duration-150"
        >
          <span
            className="h-1.5 w-1.5 rounded-full bg-primary motion-safe:animate-pulse"
          />
          <span className="text-[11.5px] font-medium text-foreground">
            Agentic RAG
          </span>
          <span className="text-[10px] text-muted-foreground font-light">/</span>
          <span className="text-[11.5px] text-muted-foreground">
            Tool-routing
          </span>
          <span className="text-[10px] text-muted-foreground font-light">/</span>
          <span className="text-[11.5px] text-muted-foreground">
            Open source
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          custom={0.1}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="text-balance text-5xl font-semibold leading-[1.04] text-foreground md:text-[68px] lg:text-[76px]"
          style={{
            fontFamily: "'DM Sans', -apple-system, sans-serif",
            letterSpacing: '-0.035em',
            textWrap: 'balance',
          }}
        >
          The RAG workspace that{' '}
          <span className="text-primary">
            thinks before it searches.
          </span>
        </motion.h1>

        {/* Sub-line */}
        <motion.p
          custom={0.22}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="mt-6 max-w-[540px] text-[15px] md:text-[16.5px] font-normal leading-[1.65] text-muted-foreground"
        >
          Agentic tool-routing decides between document retrieval, web search, deep research,
          and direct reasoning, each answer grounded with citations.
        </motion.p>

        {/* CTAs */}
        <motion.div
          custom={0.34}
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          <div>
            <button
              onClick={() => { navigate('/chat/new') }}
              className="group flex items-center gap-2 rounded-full bg-primary hover:bg-primary/90 active:scale-[0.98] disabled:opacity-50 px-6 py-2.5 text-sm font-semibold text-primary-foreground shadow-sm transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary motion-safe:hover:scale-[1.02] motion-reduce:transform-none"
            >
              <span>Launch workspace</span>
              <ArrowRight className="h-4 w-4 transition-transform duration-150 motion-reduce:transform-none group-hover:translate-x-0.5" aria-hidden="true" />
            </button>
          </div>
          <a
            href="https://github.com/ayush3739/RAG-QA"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-full border border-border bg-surface px-6 py-2.5 text-sm font-medium text-foreground transition-colors duration-150 hover:bg-surface-container active:scale-[0.98] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary motion-safe:hover:scale-[1.02] motion-reduce:transform-none"
          >
            <Github className="h-4 w-4" aria-hidden="true" />
            View on GitHub
          </a>
        </motion.div>

        {/* Trust line */}
        <motion.p
          custom={0.44}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          className="mt-5 text-[11px] text-muted-foreground"
        >
          Self-hosted, locally orchestrated, MIT licensed
        </motion.p>
      </div>

      {/* ── Product mockup ── */}
      <motion.div
        custom={0.52}
        variants={fadeUp}
        initial="hidden"
        animate="visible"
        className="relative z-10 mx-auto w-full max-w-[1100px] px-4 pb-20"
      >
        {/* Soft tight ambient bloom strictly behind top horizon rim */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-7 left-1/2 -translate-x-1/2 h-[90px] w-[75%] max-w-[750px] rounded-full bg-gradient-to-b from-blue-500/25 via-primary/15 to-transparent blur-[25px]"
        />

        {/* Crisp hairline horizon light on top edge */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-[1px] left-1/2 -translate-x-1/2 h-[1px] w-[60%] max-w-[600px] bg-gradient-to-r from-transparent via-blue-400/80 to-transparent"
        />

        {/* Browser chrome with clean border */}
        <div className="relative overflow-hidden rounded-[14px] border border-border bg-surface shadow-lg">
          {/* Title bar */}
          <div className="flex h-10 items-center border-b border-border bg-surface-container-low px-4">
            <div className="flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-full bg-danger" />
              <span className="h-3 w-3 rounded-full bg-warning" />
              <span className="h-3 w-3 rounded-full bg-success" />
            </div>
            <div className="mx-auto flex h-5 w-[200px] items-center justify-center gap-1.5 rounded-md border border-border bg-surface px-3">
              <span className="text-[10px] text-muted-foreground">app.KnowVex.ai</span>
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
