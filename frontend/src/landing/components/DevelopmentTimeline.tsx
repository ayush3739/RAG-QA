import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'

export default function DevelopmentTimeline() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-60px' })
  const prefersReduced = useReducedMotion()
  const shouldAnimate = !prefersReduced
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const phases = [
    { phase: 'Phase 1', title: 'RAG Core', desc: 'Parent-child chunking schemas, BM25 indexing integration, and cosine vector queries.' },
    { phase: 'Phase 2', title: 'FastAPI Backend', desc: 'Async gateway implementation, session routers, validation schemas, and streaming SSE.' },
    { phase: 'Phase 3', title: 'Tool Routing', desc: 'Classification model setup, routing state graph structures, and fallback thresholds.' },
    { phase: 'Phase 4', title: 'Research Workflows', desc: 'Decomposing queries, synthesis agents, web scraping, and evidence consolidation.' },
    { phase: 'Phase 5', title: 'Evaluation & Guardrails', desc: 'RAGAS benchmark validation suites, golden query set audits, and input injection filters.' }
  ]

  return (
    <section
      id="timeline"
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
          className="mb-16 text-center md:text-left"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Roadmap
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Development Timeline
          </h2>
          <p className="mt-4 text-[#707070] max-w-xl text-base leading-relaxed">
            Five sequential project phases executed to design, orchestrate, and validate the system architecture.
          </p>
        </motion.div>

        {/* Desktop Horizontal Line + Steps */}
        <div className="hidden md:block relative pb-8 mt-12">
          {/* Timeline Line */}
          <div className="absolute top-[3px] left-[10%] right-[10%] h-px bg-white/5" />
          <motion.div
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
            className="absolute top-[3px] left-[10%] right-[10%] h-px bg-gradient-to-r from-[#7C5CFF] via-[#3FD8C4] to-white origin-left"
          />

          <div className="grid grid-cols-5 gap-4 relative z-10">
            {phases.map((item, idx) => (
              <motion.div
                key={item.phase}
                initial={mounted && shouldAnimate ? { opacity: 0, y: 30 } : false}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: idx * 0.1, ease: [0.23, 1, 0.32, 1] }}
                className="text-center px-2 flex flex-col items-center"
              >
                {/* Node Dot */}
                <div className="w-2.5 h-2.5 rounded-full border-2 border-black bg-white shadow-[0_0_8px_rgba(255,255,255,0.4)] mb-6 z-20" />
                <span className="font-mono text-[10px] text-[#7C5CFF] tracking-wider uppercase font-semibold">{item.phase}</span>
                <h3 className="mt-2 text-sm font-medium text-white">{item.title}</h3>
                <p className="mt-2 text-xs leading-relaxed text-[#707070] max-w-[160px] mx-auto">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Mobile Vertical Timeline */}
        <div className="md:hidden relative pl-8 mt-8">
          <div className="absolute left-[3px] top-0 bottom-0 w-px bg-white/5" />
          <div className="space-y-8">
            {phases.map((item, idx) => (
              <motion.div
                key={item.phase}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: idx * 0.08 }}
                className="relative"
              >
                {/* Node Dot */}
                <div className="absolute -left-[32px] top-1 w-2.5 h-2.5 rounded-full border-2 border-black bg-white z-20" />
                <span className="font-mono text-[10px] text-[#7C5CFF] tracking-wider uppercase font-semibold">{item.phase}</span>
                <h3 className="text-sm font-medium text-white mt-1">{item.title}</h3>
                <p className="text-xs leading-relaxed text-[#707070] mt-1.5">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
