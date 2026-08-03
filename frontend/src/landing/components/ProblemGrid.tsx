import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'
import { problems } from '../data/problems'

const containerVariants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.06 },
  },
}

const rowVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] as [number, number, number, number] },
  },
}

export default function ProblemGrid() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-80px' })
  const prefersReducedMotion = useReducedMotion()
  const shouldAnimate = !prefersReducedMotion
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  return (
    <section
      id="problem"
      ref={sectionRef}
      className="relative py-20 md:py-24 overflow-hidden"
      style={{ backgroundColor: '#050505' }}
    >
      <div className="max-w-4xl mx-auto px-6">
        <div className="linear-gradient-line mb-12" />

        <motion.div
          initial={mounted && shouldAnimate ? { opacity: 0, y: 16 } : false}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
          className="mb-10"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Motivation
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Addressing standard RAG limits
          </h2>
          <p className="mt-4 text-[#707070] max-w-xl text-base leading-relaxed">
            Traditional architectures often face systemic constraints that I wanted to resolve in this implementation.
          </p>
        </motion.div>

        <motion.div
          variants={mounted && shouldAnimate ? containerVariants : undefined}
          initial={mounted && shouldAnimate ? 'hidden' : false}
          animate={isInView ? 'visible' : (mounted && shouldAnimate ? 'hidden' : false)}
          className="max-w-3xl"
        >
          {problems.map((problem) => {
            const Icon = problem.icon
            return (
              <motion.div
                key={problem.title}
                variants={shouldAnimate ? rowVariants : undefined}
                className="flex items-start gap-4 py-5 border-b border-white/5 last:border-b-0"
              >
                <Icon className="w-4 h-4 text-[#7C5CFF] opacity-60 mt-0.5 flex-shrink-0" strokeWidth={1.5} />
                <div>
                  <h3 className="text-sm font-medium text-[#EDEDED] leading-snug">{problem.title}</h3>
                  <p className="text-sm text-[#707070] mt-1 leading-relaxed">{problem.impact}</p>
                </div>
              </motion.div>
            )
          })}
        </motion.div>
      </div>
    </section>
  )
}
