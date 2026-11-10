import { motion } from 'framer-motion'

const rows = [
  { feature: 'Dynamic Tool Routing', traditional: 'Retrieve-on-every-query', KnowVex: 'Single-pass intent router' },
  { feature: 'Multi-Step Research', traditional: 'Single database query search', KnowVex: 'Stateful research subagent loops' },
  { feature: 'Confidence Thresholds', traditional: 'Outputs hallucinations blindly', KnowVex: 'Measures score; fallback alerts' },
  { feature: 'Automated Evaluation', traditional: 'Human feedback vibe checks', KnowVex: 'Programmatic RAGAS testing' },
  { feature: 'Structured REST Access', traditional: 'Locked inside a browser client', KnowVex: 'Structured stream JSON endpoints' }
]

export default function ComparisonTable() {
  return (
    <section id="comparison" className="border-t border-border bg-background py-28">
      <div className="mx-auto max-w-5xl px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
            Case Study · 07
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-foreground md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            What Makes It Different
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-muted-foreground">
            Comparing standard Chat-with-PDF implementations side-by-side with KnowVex’s agentic pipeline design.
          </p>
        </motion.div>

        {/* Table Container */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="overflow-hidden rounded-2xl border border-border bg-card/60 dark:bg-white/[0.02] shadow-sm"
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-border bg-surface-container-low dark:bg-[#060608]">
                  <th className="w-1/3 px-6 py-4 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Feature</th>
                  <th className="w-1/3 px-6 py-4 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Typical Chat-with-PDF</th>
                  <th className="w-1/3 px-6 py-4 font-mono text-[10px] uppercase tracking-wider text-primary font-semibold">KnowVex RAG</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {rows.map((row, idx) => (
                  <motion.tr
                    key={row.feature}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: idx * 0.05 }}
                    className="transition-colors duration-150 hover:bg-muted/40 dark:hover:bg-white/[0.03]"
                  >
                    <td className="px-6 py-4 font-medium text-foreground">{row.feature}</td>
                    <td className="px-6 py-4 text-muted-foreground">
                      <span className="mr-2 text-red-500 font-bold">✕</span> {row.traditional}
                    </td>
                    <td className="px-6 py-4 font-medium text-foreground/90">
                      <span className="mr-2 text-emerald-500 font-bold">✓</span> {row.KnowVex}
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
