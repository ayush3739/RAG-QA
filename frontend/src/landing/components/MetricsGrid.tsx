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
      <div className="mx-auto max-w-7xl px-6 w-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] as const }}
          className="grid w-full grid-cols-1 gap-0 divide-y divide-border md:grid-cols-3 md:divide-x md:divide-y-0"
        >
          {metrics.map((m) => (
            <div
              key={m.value}
              className="group px-0 py-8 transition-colors duration-150 md:px-8 lg:px-12 md:py-4 w-full flex flex-col justify-center md:first:pl-0 md:last:pr-0"
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
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
