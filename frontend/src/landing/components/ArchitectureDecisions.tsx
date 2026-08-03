import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'
import { HelpCircle, Layers, Database, Cpu, Compass } from 'lucide-react'

export default function ArchitectureDecisions() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-60px' })
  const prefersReduced = useReducedMotion()
  const shouldAnimate = !prefersReduced
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const decisions = [
    {
      icon: Layers,
      title: 'Why LangGraph?',
      description: 'Enables stateful retrieval loops, dynamic router fallback thresholds, and conditional transitions that LangChain chains cannot support naturally.',
      tag: 'Orchestration'
    },
    {
      icon: Database,
      title: 'Why pgvector?',
      description: 'Keeps dense vector embeddings stored directly alongside relational document metadata. Eliminates synchronization delay and keeps the operational footprint small.',
      tag: 'Vector Storage'
    },
    {
      icon: Cpu,
      title: 'Why FastAPI?',
      description: 'Provides high-concurrency async runtimes, native streaming for Server-Sent Events, and automated OpenAPI validation out of the box.',
      tag: 'API Layer'
    },
    {
      icon: Compass,
      title: 'Why RAGAS?',
      description: 'Replaces qualitative, subjective "vibe-checks" with measurable mathematical metrics (Faithfulness, Precision, Recall) executed against golden test datasets.',
      tag: 'Evaluation'
    },
    {
      icon: HelpCircle,
      title: 'Why pgvector instead of Qdrant?',
      description: 'Migrated from Qdrant to pgvector to leverage a single database. Simplifies local deployment and operations while allowing direct SQL relational filters inside vector searches.',
      tag: 'Migration Tradeoff',
      highlight: true
    }
  ]

  return (
    <section
      id="decisions"
      ref={sectionRef}
      className="relative py-20 md:py-24 overflow-hidden"
      
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
          <span className="text-sm font-medium text-primary">
            Trade-offs
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Architecture Decisions
          </h2>
          <p className="mt-4 text-muted-foreground max-w-xl text-base leading-relaxed">
            Key engineering choices made to optimize reliability, latency, and codebase maintainability.
          </p>
        </motion.div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {decisions.map((dec, i) => {
            const Icon = dec.icon
            return (
              <motion.div
                key={dec.title}
                initial={mounted && shouldAnimate ? { opacity: 0, y: 20 } : false}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: i * 0.08, ease: [0.23, 1, 0.32, 1] }}
                className={`linear-gradient-border p-6 md:p-8 flex flex-col justify-between h-full bg-surface-container-lowest ${
                  dec.highlight ? 'md:col-span-2 lg:col-span-2 border-primary/20 bg-primary/5' : ''
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className={`p-2 rounded-lg border ${
                      dec.highlight ? 'border-primary/20 text-primary' : 'border-border text-muted-foreground'
                    } bg-surface-container`}>
                      <Icon className="w-4.5 h-4.5" strokeWidth={1.5} />
                    </span>
                    <span className="font-mono text-[9px] uppercase tracking-widest text-muted">
                      {dec.tag}
                    </span>
                  </div>
                  <h3 className="text-sm font-medium text-foreground mb-2">{dec.title}</h3>
                  <p className="text-xs leading-relaxed text-muted-foreground">{dec.description}</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
