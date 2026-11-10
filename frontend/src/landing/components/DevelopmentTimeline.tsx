import { motion } from 'framer-motion'

const phases = [
  { phase: 'Phase 1', title: 'RAG Core', desc: 'Parent-child chunking schemas, BM25 indexing integration, and cosine vector queries.' },
  { phase: 'Phase 2', title: 'FastAPI Backend', desc: 'Async gateway implementation, session routers, validation schemas, and streaming SSE.' },
  { phase: 'Phase 3', title: 'Tool Routing', desc: 'Classification model setup, routing state graph structures, and fallback thresholds.' },
  { phase: 'Phase 4', title: 'Research Workflows', desc: 'Decomposing queries, synthesis agents, web scraping, and evidence consolidation.' },
  { phase: 'Phase 5', title: 'Evaluation & Guardrails', desc: 'RAGAS benchmark validation suites, golden query set audits, and input injection filters.' }
]

export default function DevelopmentTimeline() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.12 },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] },
    },
  }

  return (
    <section id="timeline" className="border-t border-border bg-background py-28">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
            Case Study · 09
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-foreground md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Development Timeline
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-muted-foreground">
            Five sequential project phases executed to design, orchestrate, and validate the system architecture.
          </p>
        </motion.div>

        {/* Timeline Desktop */}
        <div className="hidden md:block">
          <div className="relative pt-4">
            {/* Horizontal Line */}
            <div className="absolute top-[21px] left-[10px] right-[10px] h-px bg-border" />
            <motion.div
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.0, ease: 'easeOut', delay: 0.2 }}
              className="absolute top-[21px] left-[10px] right-[10px] h-px origin-left bg-primary/60"
            />

            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: '-50px' }}
              className="relative grid grid-cols-5 gap-4"
            >
              {phases.map((item) => (
                <motion.div
                  key={item.phase}
                  variants={itemVariants}
                  className="group px-2 transition-transform duration-200 hover:-translate-y-1"
                >
                  <div
                    className="mb-6 h-[9px] w-[9px] rounded-full bg-primary ring-4 ring-background transition-all duration-300 group-hover:scale-125 group-hover:ring-primary/20"
                  />
                  <span className="font-mono text-[10px] uppercase tracking-wider text-primary font-medium">
                    {item.phase}
                  </span>
                  <h3 className="mt-3 text-[15px] font-medium text-foreground transition-colors duration-200 group-hover:text-primary">{item.title}</h3>
                  <p className="mt-2 text-[13px] leading-relaxed text-muted-foreground transition-colors duration-200 group-hover:text-foreground/80">{item.desc}</p>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>

        {/* Timeline Mobile */}
        <div className="relative ml-2 border-l border-border md:hidden">
          <motion.div
            initial={{ scaleY: 0 }}
            whileInView={{ scaleY: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1.0, ease: 'easeOut' }}
            className="absolute bottom-0 left-[-1px] top-0 h-full w-px origin-top bg-primary/60"
          />
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="space-y-10 pl-6"
          >
            {phases.map((item) => (
              <motion.div key={item.phase} variants={itemVariants} className="relative">
                <div
                  className="absolute -left-[29px] top-1.5 h-[9px] w-[9px] rounded-full bg-primary ring-4 ring-background"
                />
                <span className="font-mono text-[10px] uppercase tracking-wider text-primary font-medium">
                  {item.phase}
                </span>
                <h3 className="mt-2 text-[15px] font-medium text-foreground">{item.title}</h3>
                <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">{item.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  )
}
