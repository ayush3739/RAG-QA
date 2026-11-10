import { motion } from 'framer-motion'

const metrics = [
  {
    value: '<150ms',
    label: 'Routing latency',
    sub: 'Single LLM call to classify and dispatch',
  },
  {
    value: '4 tools',
    label: 'Integrated search modes',
    sub: 'Direct · Documents · Web · Deep Research',
  },
  {
    value: '100%',
    label: 'Source-cited responses',
    sub: 'Every answer traces back to a source',
  },
]

export default function MetricsGrid() {
  return (
    <section id="metrics" className="border-y border-border bg-background py-20">
      <div className="mx-auto max-w-6xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
          className="grid grid-cols-1 gap-0 divide-y divide-border md:grid-cols-3 md:divide-x md:divide-y-0"
        >
          {metrics.map((m, i) => (
            <motion.div
              key={m.value}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.12 }}
              className="group px-0 py-10 transition-colors duration-150 md:px-12 md:py-4 first:pl-0 last:pr-0"
            >
              <p
                className="mb-2 text-5xl font-medium tracking-[-0.04em] text-foreground transition-transform duration-150 motion-reduce:transform-none group-hover:translate-x-1 md:text-6xl"
                style={{ fontFamily: "'DM Sans', sans-serif" }}
              >
                {m.value}
              </p>
              <p className="mb-1 text-[15px] font-medium text-foreground">
                {m.label}
              </p>
              <p className="text-[13px] text-muted-foreground">
                {m.sub}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
