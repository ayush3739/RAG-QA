import { useRef } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { ArrowRight } from 'lucide-react'

export default function Hero() {
  const sectionRef = useRef<HTMLElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start start', 'end start'],
  })

  const panelScale = useTransform(scrollYProgress, [0, 1], [1, 0.95])
  const panelRotateX = useTransform(scrollYProgress, [0, 1], [4, 8])
  const panelOpacity = useTransform(scrollYProgress, [0.6, 1], [1, 0.6])

  const handleScroll = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault()
    const target = document.querySelector(href)
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section
      id="hero"
      ref={sectionRef}
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden linear-mesh-bg"
      style={{ backgroundColor: '#050505' }}
    >
      {/* Background orbs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        <div className="linear-orb top-0 left-1/2 h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/4 bg-[#7C5CFF] opacity-[0.08]" />
        <div className="linear-orb bottom-0 right-0 h-[400px] w-[400px] translate-x-1/4 translate-y-1/4 bg-[#3FD8C4] opacity-[0.04]" />
        <div className="linear-orb top-1/2 left-1/2 h-[300px] w-[300px] -translate-x-1/2 -translate-y-1/2 bg-white opacity-[0.02]" />
      </div>

      {/* Content */}
      <div className="relative z-10 mx-auto flex max-w-5xl flex-col items-center px-4 pb-24 pt-28 sm:px-6">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
          className="mb-10"
        >
          <span className="inline-flex items-center rounded-full border border-white/5 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-[#707070]">
            Open Source • Portfolio Project • Production-Inspired RAG System
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.23, 1, 0.32, 1], delay: 0.1 }}
          className="max-w-5xl text-center text-5xl font-light leading-[1.05] tracking-tight md:text-7xl lg:text-[88px]"
        >
          <span className="linear-gradient-text-subtle">The RAG system</span>
          <br />
          <span className="linear-gradient-text-subtle">that thinks before</span>
          <br />
          <span className="linear-gradient-text">it searches.</span>
        </motion.h1>

        {/* Sub-headline */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.23, 1, 0.32, 1], delay: 0.25 }}
          className="mt-6 max-w-2xl text-center text-base leading-relaxed text-[#707070] md:text-lg"
        >
          Agentic tool-routed RAG that decides between document retrieval,
          web search, research workflows, and direct reasoning—
          with citations, confidence scoring, and structured APIs.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.23, 1, 0.32, 1], delay: 0.4 }}
          className="mt-8 flex flex-col items-center gap-4 sm:flex-row"
        >
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="rounded-full bg-white px-6 py-2.5 text-sm font-medium text-black transition-colors duration-200 hover:bg-white/90 cursor-pointer"
          >
            Live Demo
          </button>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-white/10 bg-white/5 px-6 py-2.5 text-sm font-medium text-white transition-colors duration-200 hover:bg-white/10"
          >
            GitHub Repository
          </a>
        </motion.div>

        {/* Tech Stack Badges */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-8 flex flex-wrap justify-center gap-2 max-w-xl"
        >
          {['FastAPI', 'LangGraph', 'Postgres + pgvector', 'RAGAS', 'LangSmith', 'Docker'].map((tech) => (
            <span
              key={tech}
              className="px-3 py-1 text-[11px] font-mono text-[#7C5CFF] bg-[#7C5CFF]/[0.05] border border-[#7C5CFF]/15 rounded-full"
            >
              {tech}
            </span>
          ))}
        </motion.div>

        {/* Stats Strip */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="w-full max-w-4xl mt-16 grid grid-cols-2 md:grid-cols-5 gap-4 py-6 border-y border-white/5 text-center bg-white/[0.01] rounded-lg"
        >
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-white">10k+</p>
            <p className="text-[11px] font-mono text-[#707070] mt-1 uppercase">Chunks Indexed</p>
          </div>
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-[#3FD8C4]">&lt;1.5s</p>
            <p className="text-[11px] font-mono text-[#707070] mt-1 uppercase">Retrieval Latency</p>
          </div>
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-[#F5B942]">6</p>
            <p className="text-[11px] font-mono text-[#707070] mt-1 uppercase">Eval Metrics</p>
          </div>
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-[#7C5CFF]">4</p>
            <p className="text-[11px] font-mono text-[#707070] mt-1 uppercase">Tool Routes</p>
          </div>
          <div className="col-span-2 md:col-span-1">
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-white/80">2</p>
            <p className="text-[11px] font-mono text-[#707070] mt-1 uppercase">LLM Calls (Path)</p>
          </div>
        </motion.div>

        {/* Technical Highlights Strip */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mt-6 flex flex-wrap justify-center gap-x-6 gap-y-2 font-mono text-xs text-[#707070] border border-white/5 px-6 py-2.5 rounded-full bg-white/[0.02]"
        >
          {['Hybrid Search', 'Agentic Routing', 'Async Streaming', 'Multi-Document QA', 'Evaluation Suite', 'Observability'].map((item) => (
            <span key={item} className="flex items-center gap-1.5">
              <span className="text-[#3FD8C4]">✓</span> {item}
            </span>
          ))}
        </motion.div>

        {/* 3D Product Screenshot */}
        <motion.div
          ref={panelRef}
          style={{
            scale: panelScale,
            rotateX: panelRotateX,
            opacity: panelOpacity,
          }}
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.23, 1, 0.32, 1], delay: 0.5 }}
          className="relative mt-16 w-full max-w-4xl"
        >
          {/* Purple glow behind panel */}
          <div
            className="absolute left-1/2 top-1/2 h-[60%] w-[80%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#7C5CFF] opacity-[0.07]"
            style={{ filter: 'blur(100px)' }}
            aria-hidden="true"
          />

          {/* Product panel with 3D perspective */}
          <div
            className="linear-product-panel linear-reflection linear-shimmer"
            style={{ perspective: '1200px' }}
          >
            <img
              src="/dashboard.png"
              alt="DocuMind dashboard showing activity charts, document cards, and search interface"
              width={1200}
              height={720}
              className="w-full rounded-[15px]"
            />
          </div>
        </motion.div>
      </div>

      {/* Bottom fade */}
      <div
        className="pointer-events-none absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#050505] to-transparent"
        aria-hidden="true"
      />
    </section>
  )
}
