import { motion } from 'framer-motion'
import { ArrowRight, Github } from 'lucide-react'

export default function Hero() {
  const scrollTo = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault()
    document.querySelector(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.12,
        delayChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 16 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: [0.23, 1, 0.32, 1],
      },
    },
  }

  return (
    <section
      id="hero"
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-[#09090b]"
    >
      {/* Subtle grid texture */}
      <div
        className="pointer-events-none absolute inset-0"
        aria-hidden="true"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px)
          `,
          backgroundSize: '72px 72px',
        }}
      />
      {/* Faint indigo spot */}
      <div
        className="pointer-events-none absolute inset-0"
        aria-hidden="true"
        style={{
          background: 'radial-gradient(ellipse 60% 40% at 50% 0%, rgba(99,102,241,0.08) 0%, transparent 70%)',
        }}
      />

      {/* Content */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="relative z-10 mx-auto flex max-w-4xl flex-col items-center px-6 py-32 text-center"
      >
        {/* Badge */}
        <motion.div variants={itemVariants} className="mb-8 inline-flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.04] px-3.5 py-1 backdrop-blur-sm">
          <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse" />
          <span className="font-mono text-[11px] font-medium uppercase tracking-[0.1em] text-zinc-400">
            Open Source · RAG · Production-grade
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          variants={itemVariants}
          className="text-balance text-5xl font-medium leading-[1.06] tracking-[-0.035em] text-white md:text-7xl lg:text-[84px]"
          style={{ fontFamily: "'DM Sans', sans-serif" }}
        >
          The RAG system that{' '}
          <span className="text-zinc-400">thinks before it searches.</span>
        </motion.h1>

        {/* Subline */}
        <motion.p
          variants={itemVariants}
          className="mt-7 max-w-xl text-[17px] leading-relaxed text-zinc-500"
        >
          Agentic tool-routing that decides between document retrieval, web search,
          deep research, and direct reasoning — with citations and structured APIs.
        </motion.p>

        {/* CTA row */}
        <motion.div variants={itemVariants} className="mt-10 flex flex-wrap items-center justify-center gap-3">
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="group flex items-center gap-2 rounded-full bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white transition-all duration-200 hover:bg-indigo-500 hover:scale-[1.02] active:scale-[0.98] hover:shadow-lg hover:shadow-indigo-500/25"
          >
            Launch workspace
            <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
          </button>
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

        {/* Scroll hint */}
        <motion.a
          variants={itemVariants}
          href="#how-it-works"
          onClick={(e) => scrollTo(e, '#how-it-works')}
          className="mt-24 flex flex-col items-center gap-2 text-zinc-600 transition-colors duration-200 hover:text-zinc-300"
        >
          <span className="text-[11px] font-medium uppercase tracking-[0.12em]">How it works</span>
          <svg width="16" height="24" viewBox="0 0 16 24" fill="none" className="animate-bounce">
            <path d="M8 0v20M1 13l7 7 7-7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </motion.a>
      </motion.div>
    </section>
  )
}
