import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'
import { ShieldAlert, Zap, Server, Brain } from 'lucide-react'

export default function ChallengesTradeoffs() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-60px' })
  const prefersReduced = useReducedMotion()
  const shouldAnimate = !prefersReduced
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const challenges = [
    {
      icon: Brain,
      title: 'Why not LangGraph everywhere?',
      description: 'Only the retriever subagent needs execution loops (Rerank -> Re-query -> Consolidate). Keeping the core router simple and linear minimizes latency overhead.',
      tag: 'Orchestration Complexity'
    },
    {
      icon: Zap,
      title: 'Why not a larger model?',
      description: 'Using small local models (like Llama 3 8B) for classification and routing handles tasks at a fraction of the token cost and context latency compared to GPT-4.',
      tag: 'Cost & Latency Tradeoff'
    },
    {
      icon: Server,
      title: 'Why not external vector DBs?',
      description: 'Choosing pgvector in Postgres keeps deployment simple. Eliminates metadata synchronization delay and keeps the local Docker network footprint small.',
      tag: 'Data Sync Overhead'
    },
    {
      icon: ShieldAlert,
      title: 'Why confidence is not enough',
      description: 'High model confidence score can still hallucinate under edge prompt injections. Implemented a groundedness evaluator to scan contexts before output.',
      tag: 'Hallucination Mitigation'
    }
  ]

  return (
    <section
      id="challenges"
      ref={sectionRef}
      className="relative py-20 md:py-24 linear-mesh-bg overflow-hidden"
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
            Design Tradeoffs
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Challenges &amp; Tradeoffs
          </h2>
          <p className="mt-4 text-[#707070] max-w-xl text-base leading-relaxed">
            Real engineering constraints require deliberate tradeoffs. Here is how I approached scalability and cost boundaries.
          </p>
        </motion.div>

        {/* 2-Column Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {challenges.map((chal, i) => {
            const Icon = chal.icon
            return (
              <motion.div
                key={chal.title}
                initial={mounted && shouldAnimate ? { opacity: 0, y: 20 } : false}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: i * 0.1, ease: [0.23, 1, 0.32, 1] }}
                className="linear-gradient-border p-6 md:p-8 flex flex-col justify-between h-full bg-white/[0.01]"
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="p-2 rounded-lg border border-white/5 text-[#707070] bg-white/[0.02]">
                      <Icon className="w-4.5 h-4.5" strokeWidth={1.5} />
                    </span>
                    <span className="font-mono text-[9px] uppercase tracking-widest text-[#444]">
                      {chal.tag}
                    </span>
                  </div>
                  <h3 className="text-sm font-medium text-white mb-2">{chal.title}</h3>
                  <p className="text-xs leading-relaxed text-[#707070]">{chal.description}</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
