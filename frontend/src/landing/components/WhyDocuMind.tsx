import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'

export default function WhyDocuMind() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-80px' })
  const prefersReduced = useReducedMotion()
  const shouldAnimate = !prefersReduced
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const containerVariants = {
    hidden: {},
    visible: {
      transition: { staggerChildren: 0.1 },
    },
  }

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, ease: [0.23, 1, 0.32, 1] },
    },
  }

  return (
    <section
      id="why-exists"
      ref={sectionRef}
      className="relative py-20 md:py-24 overflow-hidden"
      style={{ backgroundColor: '#050505' }}
    >
      <div className="max-w-5xl mx-auto px-6">
        {/* Divider */}
        <div className="linear-gradient-line mb-12" />

        {/* Header */}
        <motion.div
          initial={mounted && shouldAnimate ? { opacity: 0, y: 16 } : false}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
          className="mb-12 text-center md:text-left"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Case Study
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Why DocuMind Exists
          </h2>
          <p className="mt-4 text-[#707070] max-w-xl text-base leading-relaxed">
            Evaluating the trade-offs of modern RAG pipelines and addressing standard architectural limitations.
          </p>
        </motion.div>

        {/* Two Column Layout: The Problem vs The Approach */}
        <motion.div
          variants={mounted && shouldAnimate ? containerVariants : undefined}
          initial={mounted && shouldAnimate ? 'hidden' : false}
          animate={isInView ? 'visible' : 'hidden'}
          className="grid grid-cols-1 md:grid-cols-2 gap-8"
        >
          {/* Column 1: The Problem */}
          <motion.div
            variants={shouldAnimate ? cardVariants : undefined}
            className="linear-gradient-border p-6 md:p-8 bg-white/[0.01]"
          >
            <h3 className="text-lg font-medium text-white mb-6 flex items-center gap-2">
              <span className="text-red-500 font-mono text-sm">❌</span> The Problem
            </h3>
            <p className="text-xs font-mono text-[#444] uppercase mb-4 tracking-wider">Most &quot;Chat with PDF&quot; systems</p>
            <ul className="space-y-4">
              <li className="flex items-start gap-3">
                <span className="text-red-500 mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">Retrieve on every query</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    Keyword-based and general knowledge questions waste database operations, adding unnecessary retrieval latency.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-red-500 mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">Hallucinate confidently</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    Standard systems output answers regardless of context quality, lacking fallbacks or groundedness validation.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-red-500 mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">Cannot reason across sources</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    Locked into single-pass vector database crawls without web search fallbacks or structured research synthesis.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-red-500 mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">No measurable quality</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    Iterating on code without automated validation pipelines leads to silent performance and retrieval regressions.
                  </p>
                </div>
              </li>
            </ul>
          </motion.div>

          {/* Column 2: The Approach */}
          <motion.div
            variants={shouldAnimate ? cardVariants : undefined}
            className="linear-gradient-border p-6 md:p-8 bg-white/[0.01]"
          >
            <h3 className="text-lg font-medium text-white mb-6 flex items-center gap-2">
              <span className="text-[#3FD8C4] font-mono text-sm">✅</span> The Approach
            </h3>
            <p className="text-xs font-mono text-[#7C5CFF]/75 uppercase mb-4 tracking-wider">DocuMind Router Architecture</p>
            <ul className="space-y-4">
              <li className="flex items-start gap-3">
                <span className="text-[#3FD8C4] mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">Routes to the right tool</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    LLM-based classification decides whether to answer directly, crawl indexed documents, search the web, or run research.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-[#3FD8C4] mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">Measures confidence</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    Evaluates retrieval scoring and falls back gracefully to secondary tools or declines to answer when uncertain.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-[#3FD8C4] mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">Supports multi-step research</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    Iteratively search, synthesize claims, and cross-reference document contexts alongside live web search paths.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-[#3FD8C4] mt-0.5 font-mono text-xs">·</span>
                <div>
                  <h4 className="text-sm font-medium text-[#EDEDED] leading-snug">Includes evaluation and guardrails</h4>
                  <p className="text-xs text-[#707070] mt-1 leading-relaxed">
                    Built-in RAGAS assessment metrics and input/output filters validate source grounding dynamically.
                  </p>
                </div>
              </li>
            </ul>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
