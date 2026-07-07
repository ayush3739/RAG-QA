import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'

export default function ComparisonTable() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-60px' })
  const prefersReduced = useReducedMotion()
  const shouldAnimate = !prefersReduced
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const rows = [
    { feature: 'Dynamic Tool Routing', traditional: '❌ Retrieve-on-every-query', documind: '✅ Single-pass intent router' },
    { feature: 'Multi-Step Research', traditional: '❌ Single database query search', documind: '✅ Stateful research subagent loops' },
    { feature: 'Confidence Thresholds', traditional: '❌ Outputs hallucinations blindly', documind: '✅ Measures score; fallback alerts' },
    { feature: 'Automated Evaluation', traditional: '❌ Human feedback vibe checks', documind: '✅ Programmatic RAGAS testing' },
    { feature: 'Structured REST Access', traditional: '❌ Locked inside a browser client', documind: '✅ Structured stream JSON endpoints' }
  ]

  return (
    <section
      id="comparison"
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
            Comparison
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            What Makes It Different
          </h2>
          <p className="mt-4 text-[#707070] max-w-xl text-base leading-relaxed">
            Comparing standard Chat-with-PDF implementations side-by-side with DocuMind’s agentic pipeline design.
          </p>
        </motion.div>

        {/* Table Container */}
        <motion.div
          initial={mounted && shouldAnimate ? { opacity: 0, y: 24 } : false}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="linear-gradient-border overflow-hidden bg-white/[0.01]"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left font-sans text-xs sm:text-sm border-collapse">
              <thead>
                <tr className="border-b border-white/5 bg-white/[0.02]">
                  <th className="py-4 px-6 font-mono text-[10px] uppercase text-[#444] tracking-wider w-1/3">Feature</th>
                  <th className="py-4 px-6 font-mono text-[10px] uppercase text-[#444] tracking-wider w-1/3">Typical Chat-with-PDF</th>
                  <th className="py-4 px-6 font-mono text-[10px] uppercase text-[#7C5CFF] tracking-wider w-1/3">DocuMind RAG</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {rows.map((row) => (
                  <tr key={row.feature} className="hover:bg-white/[0.01] transition-colors duration-150">
                    <td className="py-4 px-6 font-medium text-white">{row.feature}</td>
                    <td className="py-4 px-6 text-[#707070]">{row.traditional}</td>
                    <td className="py-4 px-6 text-[#EDEDED] font-medium">{row.documind}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
