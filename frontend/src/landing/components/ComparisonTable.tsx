import { motion } from 'framer-motion'

const rows = [
  { feature: 'Dynamic Tool Routing', traditional: 'Retrieve-on-every-query', documind: 'Single-pass intent router' },
  { feature: 'Multi-Step Research', traditional: 'Single database query search', documind: 'Stateful research subagent loops' },
  { feature: 'Confidence Thresholds', traditional: 'Outputs hallucinations blindly', documind: 'Measures score; fallback alerts' },
  { feature: 'Automated Evaluation', traditional: 'Human feedback vibe checks', documind: 'Programmatic RAGAS testing' },
  { feature: 'Structured REST Access', traditional: 'Locked inside a browser client', documind: 'Structured stream JSON endpoints' }
]

export default function ComparisonTable() {
  return (
    <section id="comparison" className="border-t border-white/[0.05] bg-[#09090b] py-28">
      <div className="mx-auto max-w-5xl px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-zinc-600">
            Case Study · 07
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            What Makes It Different
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            Comparing standard Chat-with-PDF implementations side-by-side with DocuMind’s agentic pipeline design.
          </p>
        </motion.div>

        {/* Table Container */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="overflow-hidden rounded-2xl border border-white/[0.05] bg-white/[0.02]"
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-white/[0.05] bg-[#060608]">
                  <th className="w-1/3 px-6 py-4 font-mono text-[10px] uppercase tracking-wider text-zinc-500">Feature</th>
                  <th className="w-1/3 px-6 py-4 font-mono text-[10px] uppercase tracking-wider text-zinc-500">Typical Chat-with-PDF</th>
                  <th className="w-1/3 px-6 py-4 font-mono text-[10px] uppercase tracking-wider text-indigo-400">DocuMind RAG</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.05]">
                {rows.map((row, idx) => (
                  <motion.tr
                    key={row.feature}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: idx * 0.05 }}
                    className="transition-colors duration-150 hover:bg-white/[0.03]"
                  >
                    <td className="px-6 py-4 font-medium text-white">{row.feature}</td>
                    <td className="px-6 py-4 text-zinc-500">
                      <span className="mr-2 text-red-500/50">✕</span> {row.traditional}
                    </td>
                    <td className="px-6 py-4 font-medium text-zinc-300">
                      <span className="mr-2 text-emerald-500/50">✓</span> {row.documind}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
