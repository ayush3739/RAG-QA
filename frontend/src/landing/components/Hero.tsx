import { useState, useEffect, useRef } from 'react'
import { navigate } from '../../lib/navigation'
import { motion, useReducedMotion } from 'framer-motion'
import { ShinyButton } from '../../components/ui/shiny-button'
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

/* ─── Sidebar nav items (matches actual app) ────────────── */
const NAV = [
  { id: 'dashboard', label: 'Dashboard',  icon: LayoutDashboard, count: null },
  { id: 'chats',     label: 'Chats',      icon: MessageSquare,   count: 3    },
  { id: 'documents', label: 'Documents',  icon: FolderOpen,      count: null },
  { id: 'settings',  label: 'Settings',   icon: Settings,        count: null },
]

/* ─── Interactive Demo Storyline Data ───────────────────── */
const Q1_PROMPT = 'Compare the chunking strategies across the uploaded research papers.'
const Q1_ASSISTANT_TEXT = 'Using hybrid RAG retrieval across your 3 documents. Found 7 relevant passages on chunking methodologies. Parent-child chunking outperforms fixed-size by 23% on recall…'
const Q1_WORDS = Q1_ASSISTANT_TEXT.split(' ')
const Q1_CITATIONS = [
  'Smith et al. 2023, §3.2',
  'Chen & Liu 2024, §4.1',
  'Zhu 2024, §2',
]

const Q2_PROMPT = "What's the latest on embedding models in 2025?"
const Q2_ASSISTANT_TEXT = 'Late-interaction and multimodal embedding architectures now lead MTEB benchmarks. Models featuring Matryoshka representation learning allow dynamic dimension truncation with under 1.5% loss in accuracy…'
const Q2_WORDS = Q2_ASSISTANT_TEXT.split(' ')
const Q2_CITATIONS = [
  'MTEB Leaderboard 2025',
  'arXiv:2501.0924',
]

const DOCUMENTS = [
  { name: 'chunking-benchmarks.pdf', size: '1.2 MB', pages: '14 pgs', hits: '4 passages' },
  { name: 'rag-eval-2024.pdf', size: '840 KB', pages: '8 pgs', hits: '2 passages' },
  { name: 'hybrid-search.pdf', size: '2.1 MB', pages: '22 pgs', hits: '1 passage' },
]

const SEARCH_SOURCES = [
  { name: 'arxiv.org', detail: 'cs.CL / cs.IR · 2501.0924', badge: 'Parsed' },
  { name: 'huggingface.co', detail: 'MTEB Benchmark 2025', badge: 'Live' },
]

/* ─── Pacing Timings (ms) ───────────────────────────────── */
const T_START_TYPE_Q1 = 400
const T_END_TYPE_Q1 = T_START_TYPE_Q1 + Q1_PROMPT.length * 30 // ~2440ms (30ms/char)
const T_CLICK_SEND_Q1 = T_END_TYPE_Q1 + 160                   // 2600ms
const T_ROUTING_Q1_END = T_CLICK_SEND_Q1 + 140                // 2740ms (<150ms classification pulse)
const T_ROUTED_Q1_END = T_ROUTING_Q1_END + 110               // 2850ms (tool badge pop)
const T_STREAM_Q1_END = T_ROUTED_Q1_END + Q1_WORDS.length * 25 // 3425ms (25ms/word)
const T_CIT_Q1_1 = 3450
const T_CIT_Q1_2 = 3600
const T_CIT_Q1_3 = 3750
const T_CIT_Q1_END = 3900

const T_START_TYPE_Q2 = 5200                                  // Pause ~1.3s for reading Q1
const T_END_TYPE_Q2 = T_START_TYPE_Q2 + Q2_PROMPT.length * 30 // 6520ms (30ms/char)
const T_CLICK_SEND_Q2 = T_END_TYPE_Q2 + 160                   // 6680ms
const T_ROUTING_Q2_END = T_CLICK_SEND_Q2 + 140                // 6820ms (<150ms classification pulse)
const T_ROUTED_Q2_END = T_ROUTING_Q2_END + 110               // 6930ms (tool badges pop)
const T_STREAM_Q2_END = T_ROUTED_Q2_END + Q2_WORDS.length * 25 // 7530ms (25ms/word)
const T_CIT_Q2_1 = 7550
const T_CIT_Q2_2 = 7700
const T_CIT_Q2_END = 7850

const T_HOLD_END = T_CIT_Q2_END + 5000                        // 12850ms (5000ms comfortable read hold)
const T_FADE_END = T_HOLD_END + 500                           // 13350ms (smooth fade & reset)
const CYCLE_DURATION = T_FADE_END

/* ─── State Snapshot Interface ──────────────────────────── */
interface DemoSnapshot {
  isResetFading: boolean
  inputText: string
  isSendActive: boolean
  showCursor: boolean
  showQ1Prompt: boolean
  q1Status: 'hidden' | 'routing' | 'routed' | 'streaming' | 'complete'
  q1WordCount: number
  q1CitationsCount: number
  showQ2Prompt: boolean
  q2Status: 'hidden' | 'routing' | 'routed' | 'streaming' | 'complete'
  q2WordCount: number
  q2CitationsCount: number
  inspectorMode: 'docs' | 'web'
  isDocsRetrieving: boolean
  showDocHits: boolean
  topBarDocsLabel: string
}

function getSnapshot(time: number, isReducedMotion: boolean): DemoSnapshot {
  if (isReducedMotion) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: Q1_CITATIONS.length,
      showQ2Prompt: true,
      q2Status: 'complete',
      q2WordCount: Q2_WORDS.length,
      q2CitationsCount: Q2_CITATIONS.length,
      inspectorMode: 'web',
      isDocsRetrieving: false,
      showDocHits: true,
      topBarDocsLabel: '3 Docs · Web',
    }
  }

  // 1. Initial idle
  if (time < T_START_TYPE_Q1) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: true,
      showQ1Prompt: false,
      q1Status: 'hidden',
      q1WordCount: 0,
      q1CitationsCount: 0,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 2. Typing Q1
  if (time < T_END_TYPE_Q1) {
    const chars = Math.min(
      Q1_PROMPT.length,
      Math.floor((time - T_START_TYPE_Q1) / 30)
    )
    return {
      isResetFading: false,
      inputText: Q1_PROMPT.slice(0, chars),
      isSendActive: false,
      showCursor: true,
      showQ1Prompt: false,
      q1Status: 'hidden',
      q1WordCount: 0,
      q1CitationsCount: 0,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 3. Send Q1 click
  if (time < T_CLICK_SEND_Q1) {
    return {
      isResetFading: false,
      inputText: Q1_PROMPT,
      isSendActive: true,
      showCursor: false,
      showQ1Prompt: false,
      q1Status: 'hidden',
      q1WordCount: 0,
      q1CitationsCount: 0,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 4. Q1 prompt submitted, routing decision
  if (time < T_ROUTING_Q1_END) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'routing',
      q1WordCount: 0,
      q1CitationsCount: 0,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: true,
      showDocHits: true,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 5. Q1 routed (tool badge appears)
  if (time < T_ROUTED_Q1_END) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'routed',
      q1WordCount: 0,
      q1CitationsCount: 0,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: true,
      showDocHits: true,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 6. Q1 streaming words
  if (time < T_STREAM_Q1_END) {
    const words = Math.min(
      Q1_WORDS.length,
      Math.floor((time - T_ROUTED_Q1_END) / 25) + 1
    )
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'streaming',
      q1WordCount: words,
      q1CitationsCount: 0,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: true,
      showDocHits: true,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 7. Q1 citations stagger in
  if (time < T_CIT_Q1_END) {
    let citCount = 0
    if (time >= T_CIT_Q1_3) citCount = 3
    else if (time >= T_CIT_Q1_2) citCount = 2
    else if (time >= T_CIT_Q1_1) citCount = 1

    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: citCount,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: false,
      showDocHits: true,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 8. Pause after Q1 (user reads Q1)
  if (time < T_START_TYPE_Q2) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: false,
      showDocHits: true,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 9. Typing Q2
  if (time < T_END_TYPE_Q2) {
    const chars = Math.min(
      Q2_PROMPT.length,
      Math.floor((time - T_START_TYPE_Q2) / 30)
    )
    return {
      isResetFading: false,
      inputText: Q2_PROMPT.slice(0, chars),
      isSendActive: false,
      showCursor: true,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: false,
      showDocHits: true,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 10. Send Q2 click
  if (time < T_CLICK_SEND_Q2) {
    return {
      isResetFading: false,
      inputText: Q2_PROMPT,
      isSendActive: true,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: false,
      q2Status: 'hidden',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'docs',
      isDocsRetrieving: false,
      showDocHits: true,
      topBarDocsLabel: '3 Docs',
    }
  }

  // 11. Q2 prompt submitted, routing decision
  if (time < T_ROUTING_Q2_END) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: true,
      q2Status: 'routing',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'web',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs · Web',
    }
  }

  // 12. Q2 routed (tool badges appear)
  if (time < T_ROUTED_Q2_END) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: true,
      q2Status: 'routed',
      q2WordCount: 0,
      q2CitationsCount: 0,
      inspectorMode: 'web',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs · Web',
    }
  }

  // 13. Q2 streaming words
  if (time < T_STREAM_Q2_END) {
    const words = Math.min(
      Q2_WORDS.length,
      Math.floor((time - T_ROUTED_Q2_END) / 25) + 1
    )
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: true,
      q2Status: 'streaming',
      q2WordCount: words,
      q2CitationsCount: 0,
      inspectorMode: 'web',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs · Web',
    }
  }

  // 14. Q2 citations stagger in
  if (time < T_CIT_Q2_END) {
    let citCount = 0
    if (time >= T_CIT_Q2_2) citCount = 2
    else if (time >= T_CIT_Q2_1) citCount = 1

    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: true,
      q2Status: 'complete',
      q2WordCount: Q2_WORDS.length,
      q2CitationsCount: citCount,
      inspectorMode: 'web',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs · Web',
    }
  }

  // 15. Hold complete dialogue (5 seconds)
  if (time < T_HOLD_END) {
    return {
      isResetFading: false,
      inputText: '',
      isSendActive: false,
      showCursor: false,
      showQ1Prompt: true,
      q1Status: 'complete',
      q1WordCount: Q1_WORDS.length,
      q1CitationsCount: 3,
      showQ2Prompt: true,
      q2Status: 'complete',
      q2WordCount: Q2_WORDS.length,
      q2CitationsCount: 2,
      inspectorMode: 'web',
      isDocsRetrieving: false,
      showDocHits: false,
      topBarDocsLabel: '3 Docs · Web',
    }
  }

  // 16. Fade out and reset transition
  return {
    isResetFading: true,
    inputText: '',
    isSendActive: false,
    showCursor: false,
    showQ1Prompt: true,
    q1Status: 'complete',
    q1WordCount: Q1_WORDS.length,
    q1CitationsCount: 3,
    showQ2Prompt: true,
    q2Status: 'complete',
    q2WordCount: Q2_WORDS.length,
    q2CitationsCount: 2,
    inspectorMode: 'web',
    isDocsRetrieving: false,
    showDocHits: false,
    topBarDocsLabel: '3 Docs · Web',
  }
}

function isSnapshotEqual(a: DemoSnapshot, b: DemoSnapshot): boolean {
  return (
    a.isResetFading === b.isResetFading &&
    a.inputText === b.inputText &&
    a.isSendActive === b.isSendActive &&
    a.showCursor === b.showCursor &&
    a.showQ1Prompt === b.showQ1Prompt &&
    a.q1Status === b.q1Status &&
    a.q1WordCount === b.q1WordCount &&
    a.q1CitationsCount === b.q1CitationsCount &&
    a.showQ2Prompt === b.showQ2Prompt &&
    a.q2Status === b.q2Status &&
    a.q2WordCount === b.q2WordCount &&
    a.q2CitationsCount === b.q2CitationsCount &&
    a.inspectorMode === b.inspectorMode &&
    a.isDocsRetrieving === b.isDocsRetrieving &&
    a.showDocHits === b.showDocHits &&
    a.topBarDocsLabel === b.topBarDocsLabel
  )
}

/* ─── The App Mockup ────────────────────────────────────── */
function AppMockup({ isHovered }: { isHovered: boolean }) {
  const reducedMotion = useReducedMotion() ?? false
  const [snapshot, setSnapshot] = useState<DemoSnapshot>(() => getSnapshot(0, reducedMotion))
  const prevSnapshotRef = useRef<DemoSnapshot>(snapshot)
  const isHoveredRef = useRef(isHovered)
  const chatScrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    isHoveredRef.current = isHovered
  }, [isHovered])

  // Native sequence loop with rAF
  useEffect(() => {
    if (reducedMotion) {
      const rmSnapshot = getSnapshot(0, true)
      prevSnapshotRef.current = rmSnapshot
      setSnapshot(rmSnapshot)
      return
    }

    let animId: number
    let lastTimestamp: number | null = null
    let localTime = 0

    const tick = (now: number) => {
      if (lastTimestamp === null || isHoveredRef.current) {
        lastTimestamp = now
      }
      const delta = Math.min(now - lastTimestamp, 100)
      lastTimestamp = now

      if (!isHoveredRef.current) {
        localTime = (localTime + delta) % CYCLE_DURATION
        const nextSnapshot = getSnapshot(localTime, false)
        if (!isSnapshotEqual(prevSnapshotRef.current, nextSnapshot)) {
          prevSnapshotRef.current = nextSnapshot
          setSnapshot(nextSnapshot)
        }
      }

      animId = requestAnimationFrame(tick)
    }

    animId = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(animId)
    }
  }, [reducedMotion])

  // Smooth auto-scroll behavior
  useEffect(() => {
    if (reducedMotion || isHovered) return

    const scroller = chatScrollRef.current
    if (!scroller) return

    if (!snapshot.showQ1Prompt || snapshot.isResetFading) {
      scroller.scrollTo({ top: 0, behavior: 'instant' })
      return
    }

    if (snapshot.showQ2Prompt) {
      if (snapshot.q2Status === 'streaming') {
        // Direct assignment during rapid streaming prevents smooth-scroll easing cancellations
        scroller.scrollTop = scroller.scrollHeight
      } else {
        scroller.scrollTo({
          top: scroller.scrollHeight,
          behavior: 'smooth',
        })
      }
    }
  }, [
    snapshot.showQ1Prompt,
    snapshot.showQ2Prompt,
    snapshot.q2Status,
    snapshot.q2WordCount,
    snapshot.q2CitationsCount,
    snapshot.isResetFading,
    isHovered,
    reducedMotion,
  ])

  return (
    <div className="flex h-full divide-x divide-border overflow-hidden select-none">
      {/* ── Left sidebar (matches actual app) ── */}
      <aside className="hidden w-[190px] shrink-0 flex-col bg-surface-container-low p-3 md:flex">
        {/* Brand */}
        <div className="mb-4 flex items-center gap-2 px-1">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-primary shadow-xs">
            <Database size={12} className="text-primary-foreground" aria-hidden="true" />
          </div>
          <span className="text-[12px] font-semibold tracking-tight text-foreground">KnowVex</span>
        </div>

        {/* New Chat button */}
        <button
          type="button"
          className="mb-3 flex items-center gap-2 rounded-xl border border-border bg-surface-container px-2.5 py-1.5 text-[11px] text-muted-foreground transition-colors duration-150 hover:text-foreground active:scale-[0.98] disabled:opacity-50"
        >
          <Plus size={11} aria-hidden="true" />
          New Chat
          <span className="ml-auto font-mono text-[8.5px] text-muted-foreground">⌘K</span>
        </button>

        {/* Nav items */}
        <nav className="flex flex-col gap-0.5">
          {NAV.map((item) => {
            const Icon = item.icon
            const isActive = item.id === 'chats'
            return (
              <div
                key={item.id}
                className={`flex items-center gap-2 rounded-lg px-2 py-1.5 text-[11px] transition-colors duration-150 ${
                  isActive
                    ? 'bg-surface-container-high text-foreground font-medium'
                    : 'text-muted-foreground'
                }`}
              >
                <Icon size={12} className={isActive ? 'text-primary' : ''} aria-hidden="true" />
                {item.label}
                {item.count != null && (
                  <span className="ml-auto flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-primary px-1 text-[8px] font-medium text-primary-foreground">
                    {item.count}
                  </span>
                )}
              </div>
            )
          })}
        </nav>

        {/* User chip at bottom */}
        <div className="mt-auto flex items-center gap-2 rounded-lg border border-border bg-surface-container px-2 py-1.5">
          <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary text-[9px] font-bold text-primary-foreground">
            U
          </div>
          <div className="min-w-0">
            <p className="truncate text-[10px] font-medium text-foreground">Researcher</p>
            <p className="truncate text-[8.5px] text-muted-foreground">research@knowvex.ai</p>
          </div>
        </div>
      </aside>

      {/* ── Main chat area ── */}
      <main
        className={`flex flex-1 flex-col overflow-hidden min-w-0 transition-opacity duration-500 ${
          snapshot.isResetFading ? 'opacity-0' : 'opacity-100'
        }`}
      >
        {/* Chat top bar */}
        <div className="flex h-10 shrink-0 items-center gap-2.5 border-b border-border px-4">
          <span className="text-[11.5px] font-medium text-foreground">
            {snapshot.inspectorMode === 'docs' ? 'Research Workspace' : 'Deep Research'}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-container px-2 py-0.5 text-[9px] text-muted-foreground">
            {snapshot.inspectorMode === 'docs' ? (
              <>
                <Database size={8} className="text-primary shrink-0" aria-hidden="true" />
                <span>3 Docs</span>
              </>
            ) : (
              <>
                <Globe size={8} className="text-secondary shrink-0" aria-hidden="true" />
                <span>{snapshot.topBarDocsLabel}</span>
              </>
            )}
          </span>
          <div className="ml-auto flex items-center gap-1.5">
            <span className="text-[9px] font-medium text-muted-foreground">Inspector</span>
            <PanelRight size={11} className="text-muted-foreground" aria-hidden="true" />
          </div>
        </div>

        {/* Messages list */}
        <div
          ref={chatScrollRef}
          className="flex flex-1 flex-col gap-3.5 overflow-y-auto px-4 py-3.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        >
          {/* Query 1: User Prompt */}
          {snapshot.showQ1Prompt && (
            <motion.div
              key="hero-q1-prompt"
              initial={reducedMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22, ease }}
              className="flex justify-end"
            >
              <div className="max-w-[82%] rounded-2xl rounded-tr-sm bg-primary px-3 py-2 text-[11px] leading-relaxed text-primary-foreground shadow-xs">
                {Q1_PROMPT}
              </div>
            </motion.div>
          )}

          {/* Query 1: Assistant Response */}
          {snapshot.q1Status !== 'hidden' && (
            <motion.div
              key="hero-q1-response"
              initial={reducedMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22, ease }}
              className="flex flex-col gap-1.5"
            >
              {/* Route status chip */}
              {snapshot.q1Status === 'routing' && (
                <div className="flex items-center gap-1.5">
                  <span className="relative flex h-2 w-2 items-center justify-center shrink-0">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-primary opacity-75 motion-safe:animate-ping" />
                    <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-primary" />
                  </span>
                  <span className="font-mono text-[9px] text-muted-foreground">Deciding route</span>
                </div>
              )}

              {/* Tool badge */}
              {snapshot.q1Status !== 'routing' && (
                <motion.div
                  initial={reducedMotion ? false : { opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.15 }}
                  className="flex flex-wrap gap-1.5"
                >
                  <span className="inline-flex items-center gap-1 rounded-full border border-primary/25 bg-primary/10 px-2 py-0.5 text-[8.5px] font-medium text-primary">
                    <Database size={8} aria-hidden="true" />
                    RAG Retrieval
                  </span>
                </motion.div>
              )}

              {/* Streamed or complete text */}
              {(snapshot.q1Status === 'streaming' || snapshot.q1Status === 'complete') && (
                <p className="text-[11px] leading-relaxed text-foreground">
                  {Q1_WORDS.slice(0, snapshot.q1WordCount).join(' ')}
                  {snapshot.q1Status === 'streaming' && (
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary motion-safe:animate-pulse ml-1 align-middle" />
                  )}
                </p>
              )}

              {/* Citations stagger */}
              {snapshot.q1CitationsCount > 0 && (
                <div className="mt-0.5 flex flex-wrap gap-1.5">
                  {Q1_CITATIONS.slice(0, snapshot.q1CitationsCount).map((c) => (
                    <motion.span
                      key={c}
                      initial={reducedMotion ? false : { opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.18 }}
                      className="inline-flex items-center gap-1 rounded-md border border-border bg-surface-container px-1.5 py-0.5 text-[8.5px] text-muted-foreground"
                    >
                      <CheckCircle2 size={8} className="text-success shrink-0" aria-hidden="true" />
                      {c}
                    </motion.span>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {/* Query 2: User Prompt */}
          {snapshot.showQ2Prompt && (
            <motion.div
              key="hero-q2-prompt"
              initial={reducedMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22, ease }}
              className="flex justify-end pt-1"
            >
              <div className="max-w-[82%] rounded-2xl rounded-tr-sm bg-primary px-3 py-2 text-[11px] leading-relaxed text-primary-foreground shadow-xs">
                {Q2_PROMPT}
              </div>
            </motion.div>
          )}

          {/* Query 2: Assistant Response */}
          {snapshot.q2Status !== 'hidden' && (
            <motion.div
              key="hero-q2-response"
              initial={reducedMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22, ease }}
              className="flex flex-col gap-1.5"
            >
              {/* Route status chip */}
              {snapshot.q2Status === 'routing' && (
                <div className="flex items-center gap-1.5">
                  <span className="relative flex h-2 w-2 items-center justify-center shrink-0">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75 motion-safe:animate-ping" />
                    <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-secondary" />
                  </span>
                  <span className="font-mono text-[9px] text-muted-foreground">Searching web</span>
                </div>
              )}

              {/* Tool badges */}
              {snapshot.q2Status !== 'routing' && (
                <motion.div
                  initial={reducedMotion ? false : { opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.15 }}
                  className="flex flex-wrap gap-1.5"
                >
                  <span className="inline-flex items-center gap-1 rounded-full border border-secondary/25 bg-secondary/10 px-2 py-0.5 text-[8.5px] font-medium text-secondary">
                    <Globe size={8} aria-hidden="true" />
                    Web Search
                  </span>
                  <span className="inline-flex items-center gap-1 rounded-full border border-success-subtle bg-success-subtle px-2 py-0.5 text-[8.5px] font-medium text-success">
                    <Zap size={8} aria-hidden="true" />
                    Deep Research
                  </span>
                </motion.div>
              )}

              {/* Streamed or complete text */}
              {(snapshot.q2Status === 'streaming' || snapshot.q2Status === 'complete') && (
                <p className="text-[11px] leading-relaxed text-foreground">
                  {Q2_WORDS.slice(0, snapshot.q2WordCount).join(' ')}
                  {snapshot.q2Status === 'streaming' && (
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-secondary motion-safe:animate-pulse ml-1 align-middle" />
                  )}
                </p>
              )}

              {/* Citations stagger */}
              {snapshot.q2CitationsCount > 0 && (
                <div className="mt-0.5 flex flex-wrap gap-1.5">
                  {Q2_CITATIONS.slice(0, snapshot.q2CitationsCount).map((c) => (
                    <motion.span
                      key={c}
                      initial={reducedMotion ? false : { opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.18 }}
                      className="inline-flex items-center gap-1 rounded-md border border-border bg-surface-container px-1.5 py-0.5 text-[8.5px] text-muted-foreground"
                    >
                      <Globe size={8} className="text-secondary shrink-0" aria-hidden="true" />
                      {c}
                    </motion.span>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Input bar */}
        <div className="shrink-0 border-t border-border px-3.5 py-2.5">
          <div className="flex items-center gap-2 rounded-2xl border border-border bg-surface-container px-3.5 py-2 transition-colors focus-within:border-primary/40">
            <div className="flex flex-1 items-center overflow-hidden min-w-0">
              {snapshot.inputText ? (
                <div className="flex items-center min-w-0 overflow-hidden">
                  <span className="truncate text-[11px] text-foreground font-normal">
                    {snapshot.inputText}
                  </span>
                  {snapshot.showCursor && (
                    <span className="inline-block h-3 w-[1.5px] shrink-0 bg-primary ml-0.5 animate-[cursorBlink_0.8s_infinite_step-end] align-middle" />
                  )}
                </div>
              ) : (
                <span className="flex items-center text-[11px] text-muted-foreground">
                  {snapshot.showCursor && (
                    <span className="inline-block h-3 w-[1.5px] shrink-0 bg-primary mr-1 animate-[cursorBlink_0.8s_infinite_step-end] align-middle" />
                  )}
                  Ask anything…
                </span>
              )}
            </div>
            {/* Mode badge */}
            <span className="shrink-0 rounded-full border border-border bg-surface-container-high px-1.5 py-0.5 text-[8.5px] font-medium text-muted-foreground">
              AUTO
            </span>
            <button
              type="button"
              className={`shrink-0 rounded-lg p-1 transition-all duration-100 ${
                snapshot.isSendActive
                  ? 'scale-90 bg-primary/80 text-primary-foreground'
                  : 'bg-primary text-primary-foreground active:scale-[0.98]'
              }`}
              aria-label="Send message"
            >
              <ArrowRight size={10} aria-hidden="true" />
            </button>
          </div>
        </div>
      </main>

      {/* ── Right inspector panel ── */}
      <aside
        className={`hidden w-[205px] shrink-0 flex-col bg-surface-container-low p-3 lg:flex min-w-0 transition-opacity duration-500 ${
          snapshot.isResetFading ? 'opacity-0' : 'opacity-100'
        }`}
      >
        <p className="mb-2 px-1 text-[9.5px] font-medium text-muted-foreground">Add sources</p>

        {/* Upload / Link buttons */}
        <div className="mb-3 grid grid-cols-2 gap-1.5">
          <button
            type="button"
            className="flex flex-col items-center gap-1 rounded-xl border border-border bg-surface-container py-2 text-[8.5px] text-muted-foreground transition-colors duration-150 hover:text-foreground active:scale-[0.98] disabled:opacity-50"
          >
            <Upload size={12} aria-hidden="true" />
            Upload File
          </button>
          <button
            type="button"
            className="flex flex-col items-center gap-1 rounded-xl border border-border bg-surface-container py-2 text-[8.5px] text-muted-foreground transition-colors duration-150 hover:text-foreground active:scale-[0.98] disabled:opacity-50"
          >
            <Link2 size={12} aria-hidden="true" />
            Link Existing
          </button>
        </div>

        {/* Dynamic Source Panel */}
        <motion.div
          key={snapshot.inspectorMode}
          initial={reducedMotion ? false : { opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="flex flex-1 flex-col overflow-hidden"
        >
          {snapshot.inspectorMode === 'docs' ? (
            <div className="flex flex-1 flex-col overflow-hidden">
              <div className="mb-2 flex items-center justify-between px-1">
                <p className="text-[9.5px] font-medium text-muted-foreground">Active documents (3)</p>
                {snapshot.isDocsRetrieving && (
                  <span className="flex items-center gap-1 font-mono text-[8px] text-success font-medium">
                    <span className="relative flex h-2 w-2 items-center justify-center shrink-0">
                      <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-75 motion-safe:animate-ping" />
                      <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-success" />
                    </span>
                    Retrieving
                  </span>
                )}
              </div>

              <div className="flex flex-col gap-1.5 overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                {DOCUMENTS.map((doc) => (
                  <div
                    key={doc.name}
                    className={`rounded-lg border px-2 py-1.5 transition-all duration-300 ${
                      snapshot.isDocsRetrieving
                        ? 'border-success/35 bg-success-subtle shadow-xs'
                        : 'border-border bg-surface-container'
                    }`}
                  >
                    <div className="flex items-center gap-1.5 min-w-0">
                      <FileText
                        size={11}
                        className={snapshot.isDocsRetrieving ? 'text-success shrink-0' : 'text-muted-foreground shrink-0'}
                        aria-hidden="true"
                      />
                      <span className="truncate font-mono text-[8.5px] font-medium text-foreground">
                        {doc.name}
                      </span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[8px] text-muted-foreground">
                      <span>{doc.size} · {doc.pages}</span>
                      {snapshot.showDocHits && (
                        <span className="font-mono text-success font-medium">{doc.hits}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-1 flex-col overflow-hidden">
              <div className="mb-2 flex items-center justify-between px-1">
                <p className="text-[9.5px] font-medium text-muted-foreground">Live search sources</p>
                <span className="flex items-center gap-1 font-mono text-[8px] text-secondary font-medium">
                  <span className="relative flex h-2 w-2 items-center justify-center shrink-0">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75 motion-safe:animate-pulse" />
                    <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-secondary" />
                  </span>
                  Active
                </span>
              </div>

              <div className="flex flex-col gap-1.5 overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                {SEARCH_SOURCES.map((source) => (
                  <div
                    key={source.name}
                    className="rounded-lg border border-secondary/30 bg-secondary/5 px-2 py-1.5 transition-colors duration-200"
                  >
                    <div className="flex items-center justify-between min-w-0">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <Globe size={11} className="text-secondary shrink-0" aria-hidden="true" />
                        <span className="truncate font-mono text-[8.5px] font-medium text-foreground">
                          {source.name}
                        </span>
                      </div>
                      <span className="rounded bg-secondary/10 px-1 py-0.5 font-mono text-[7.5px] text-secondary font-medium shrink-0 ml-1">
                        {source.badge}
                      </span>
                    </div>
                    <p className="mt-0.5 truncate text-[8px] text-muted-foreground">{source.detail}</p>
                  </div>
                ))}

                {/* Deep Research Node Card */}
                <div className="mt-1 rounded-lg border border-border bg-surface-container px-2 py-1.5">
                  <div className="flex items-center gap-1.5">
                    <Zap size={10} className="text-success shrink-0" aria-hidden="true" />
                    <span className="text-[9px] font-medium text-foreground">Deep Research Node</span>
                  </div>
                  <p className="mt-0.5 text-[8px] leading-tight text-muted-foreground">
                    Cross-referencing 12 benchmark variants with Matryoshka models.
                  </p>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </aside>
    </div>
  )
}

/* ─── Hero Section ──────────────────────────────────────── */
export default function Hero() {
  const [isHovered, setIsHovered] = useState(false)
  const reducedMotion = useReducedMotion() ?? false

  return (
    <section
      id="hero"
      className="relative flex min-h-dvh flex-col items-center bg-background overflow-hidden"
    >
      {/* ── Butter-smooth ambient hero spotlight ── */}
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
            <ShinyButton
              onClick={() => { navigate('/chat/new') }}
              size="default"
              variant="brand"
              animated={true}
            >
              <span>Launch workspace</span>
              <ArrowRight className="h-4 w-4 transition-transform duration-150 group-hover:translate-x-0.5" aria-hidden="true" />
            </ShinyButton>
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

      {/* ── Product mockup container ── */}
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

        {/* Browser chrome with interactive pause on hover */}
        <div
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          className="relative overflow-hidden rounded-[14px] border border-border bg-surface shadow-lg transition-shadow duration-200 hover:shadow-xl group/mockup"
        >
          {/* Title bar */}
          <div className="flex h-10 items-center border-b border-border bg-surface-container-low px-4 select-none">
            <div className="flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-full bg-danger" />
              <span className="h-3 w-3 rounded-full bg-warning" />
              <span className="h-3 w-3 rounded-full bg-success" />
            </div>
            <div className="mx-auto flex h-5 w-[170px] items-center justify-center gap-1.5 rounded-md border border-border bg-surface px-3">
              <span className="font-mono text-[9.5px] text-muted-foreground">app.knowvex.ai</span>
              {isHovered && !reducedMotion && (
                <span className="font-mono text-[8.5px] text-primary/80 font-medium">· paused</span>
              )}
            </div>
          </div>

          {/* App Demo Frame */}
          <div className="h-[480px] md:h-[540px]">
            <AppMockup isHovered={isHovered} />
          </div>
        </div>
      </motion.div>
    </section>
  )
}
