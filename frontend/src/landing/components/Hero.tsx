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
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden linear-mesh-bg bg-background"
    >
      {/* Background orbs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        <div className="linear-orb top-0 left-1/2 h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/4 bg-primary opacity-[0.08]" />
        <div className="linear-orb bottom-0 right-0 h-[400px] w-[400px] translate-x-1/4 translate-y-1/4 bg-secondary opacity-[0.04]" />
        <div className="linear-orb top-1/2 left-1/2 h-[300px] w-[300px] -translate-x-1/2 -translate-y-1/2 bg-white opacity-[0.02]" />
      </div>

      {/* Content */}
      <div className="relative z-10 mx-auto flex max-w-5xl flex-col items-center px-4 pb-24 pt-28 sm:px-6">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
          className="mb-10"
        >
          <span className="inline-flex items-center rounded-full border border-border bg-surface-container/50 px-4 py-1.5 text-sm font-medium text-muted-foreground">
            Open Source • Portfolio Project • Production-Inspired RAG System
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
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
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: [0.23, 1, 0.32, 1], delay: 0.25 }}
          className="mt-6 max-w-2xl text-center text-base leading-relaxed text-muted-foreground md:text-lg"
        >
          Agentic tool-routed RAG that decides between document retrieval,
          web search, research workflows, and direct reasoning—
          with citations, confidence scoring, and structured APIs.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: [0.23, 1, 0.32, 1], delay: 0.4 }}
          className="mt-8 flex flex-col items-center gap-4 sm:flex-row"
        >
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="group relative flex items-center justify-center gap-2 overflow-hidden rounded-full bg-white px-6 py-2.5 text-sm font-medium text-black shadow-[0_0_40px_-10px_rgba(255,255,255,0.3)] transition-all duration-300 hover:shadow-[0_0_60px_-15px_rgba(255,255,255,0.5)] hover:bg-[#FAFAFA] hover:-translate-y-0.5 cursor-pointer"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent -translate-x-full group-hover:animate-[linear-shimmer-move_1.5s_ease-in-out_infinite]" />
            <span className="relative z-10">Live Demo</span>
            <ArrowRight className="relative z-10 h-4 w-4 transition-transform duration-300 group-hover:translate-x-1" />
          </button>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-border bg-surface-container px-6 py-2.5 text-sm font-medium text-foreground transition-all duration-200 hover:bg-surface-container-high hover:border-primary/30"
          >
            GitHub Repository
          </a>
        </motion.div>

        {/* Tech Stack Badges */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-8 flex flex-wrap justify-center gap-2 max-w-xl"
        >
          {['FastAPI', 'LangGraph', 'Postgres + pgvector', 'RAGAS', 'LangSmith', 'Docker'].map((tech) => (
            <span
              key={tech}
              className="px-3 py-1 text-[11px] font-mono text-primary bg-primary/10 border border-primary/20 rounded-full"
            >
              {tech}
            </span>
          ))}
        </motion.div>

        {/* Stats Strip */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="w-full max-w-4xl mt-16 grid grid-cols-2 md:grid-cols-5 gap-4 py-6 border-y border-border text-center bg-surface-container-lowest rounded-lg"
        >
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-foreground">10k+</p>
            <p className="text-[11px] font-mono text-muted-foreground mt-1 uppercase">Chunks Indexed</p>
          </div>
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-secondary">&lt;1.5s</p>
            <p className="text-[11px] font-mono text-muted-foreground mt-1 uppercase">Retrieval Latency</p>
          </div>
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-amber-400">6</p>
            <p className="text-[11px] font-mono text-muted-foreground mt-1 uppercase">Eval Metrics</p>
          </div>
          <div>
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-primary">4</p>
            <p className="text-[11px] font-mono text-muted-foreground mt-1 uppercase">Tool Routes</p>
          </div>
          <div className="col-span-2 md:col-span-1">
            <p className="text-2xl md:text-3xl font-extralight tracking-tight text-foreground/80">2</p>
            <p className="text-[11px] font-mono text-muted-foreground mt-1 uppercase">LLM Calls (Path)</p>
          </div>
        </motion.div>

        {/* Technical Highlights Strip */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mt-6 flex flex-wrap justify-center gap-x-6 gap-y-2 font-mono text-xs text-muted-foreground border border-border px-6 py-2.5 rounded-full bg-surface-container"
        >
          {['Hybrid Search', 'Agentic Routing', 'Async Streaming', 'Multi-Document QA', 'Evaluation Suite', 'Observability'].map((item) => (
            <span key={item} className="flex items-center gap-1.5">
              <span className="text-secondary">✓</span> {item}
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
          className="relative mt-16 w-full max-w-4xl"
        >
          <motion.div
             initial={{ opacity: 0, y: 60 }}
             whileInView={{ opacity: 1, y: 0 }}
             viewport={{ once: true }}
             transition={{ duration: 1, ease: [0.23, 1, 0.32, 1], delay: 0.5 }}
          >
            {/* Purple glow behind panel */}
            <div
              className="absolute left-1/2 top-1/2 h-[60%] w-[80%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary opacity-[0.07]"
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
        </motion.div>
      </div>

      {/* Bottom fade */}
      <div
        className="pointer-events-none absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent"
        aria-hidden="true"
      />
    </section>
  )
}
