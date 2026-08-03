import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'
import { Award, Compass, Key } from 'lucide-react'

export default function LessonsLearned() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-60px' })
  const prefersReduced = useReducedMotion()
  const shouldAnimate = !prefersReduced
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const lessons = [
    {
      icon: Award,
      title: 'Retrieval beats bigger models.',
      description: 'Prioritizing search strategies like RRF hybrid merging, parent-child chunking, and CrossEncoder rerankers yields far cleaner answers than simply upgrading parameters from GPT-3.5 to GPT-4.',
      tag: 'Data Quality'
    },
    {
      icon: Compass,
      title: 'Evaluation prevents regressions.',
      description: 'Without programmatic metrics (RAGAS), modifications to vector chunk dimensions, metadata parsing, or system prompt values cause silent, untraceable quality drops.',
      tag: 'Benchmarking'
    },
    {
      icon: Key,
      title: 'Autonomous agents need constraints.',
      description: 'Unbounded multi-agent loops tend to iterate endlessly, ballooning token costs and latency. Applying strict decision classification boundaries preserves deterministic system stability.',
      tag: 'Control Systems'
    }
  ]

  return (
    <section
      id="lessons"
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
            Takeaways
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Lessons Learned
          </h2>
          <p className="mt-4 text-[#707070] max-w-xl text-base leading-relaxed">
            Key engineering insights gained while designing, testing, and debugging stateful RAG systems.
          </p>
        </motion.div>

        {/* 3-Column Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {lessons.map((les, i) => {
            const Icon = les.icon
            return (
              <motion.div
                key={les.title}
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
                      {les.tag}
                    </span>
                  </div>
                  <h3 className="text-sm font-medium text-white mb-2 leading-snug">{les.title}</h3>
                  <p className="text-xs leading-relaxed text-[#707070]">{les.description}</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
