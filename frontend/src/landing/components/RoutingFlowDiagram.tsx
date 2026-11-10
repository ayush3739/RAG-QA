import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'

const steps = [
  {
    number: '01',
    title: 'User Query',
    description: 'Any question, factual, research, or document-specific.',
  },
  {
    number: '02',
    title: 'Router Agent',
    description: 'One function-calling pass classifies intent and selects the optimal tool.',
    accent: true,
  },
  {
    number: '03',
    title: 'Execute',
    description: 'Runs document search, web retrieval, deep research, or direct LLM reasoning.',
  },
  {
    number: '04',
    title: 'Structured Answer',
    description: 'Returns a cited response with confidence score and execution trace.',
  },
]

export default function RoutingFlowDiagram() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.12 },
    },
  }

  const stepVariants = {
    hidden: { opacity: 0, y: 16 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] },
    },
  }

  return (
    <section id="how-it-works" className="bg-background py-28 border-t border-border">
      <div className="mx-auto max-w-6xl px-6">

        {/* Section heading */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-foreground md:text-5xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            One decision. Four tools.
          </h2>
          <p className="mt-4 max-w-md text-[15px] leading-relaxed text-muted-foreground">
            The router agent dispatches each query to exactly the right tool, with no manual configuration needed.
          </p>
        </motion.div>

        {/* 4 Cards with clean layout */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-4 md:grid-cols-4"
        >
          {steps.map((step, i) => (
            <motion.div
              key={step.number}
              variants={stepVariants}
              className="group relative flex flex-col justify-between rounded-2xl border border-border bg-surface p-6 transition-all duration-300 ease-out hover:-translate-y-1.5 hover:border-primary/60 hover:bg-surface-container hover:shadow-2xl hover:shadow-primary/10 dark:hover:shadow-black/80"
            >
              <div>
                {/* Top header row */}
                <div className="mb-6 flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold text-muted-foreground">
                    {step.number}
                  </span>

                  {i < steps.length - 1 && (
                    <ArrowRight className="hidden h-4 w-4 text-muted-foreground transition-transform duration-150 motion-reduce:transform-none group-hover:translate-x-1 md:block" aria-hidden="true" />
                  )}
                </div>

                <h3 className="mb-2 text-[15px] font-semibold text-foreground">
                  {step.title}
                </h3>
                <p className="text-[13px] leading-relaxed text-muted-foreground">
                  {step.description}
                </p>
              </div>

              {/* Tool Tags */}
              {step.accent && (
                <div className="mt-6 flex flex-wrap gap-1.5 pt-2">
                  {['Direct', 'Docs', 'Web', 'Research'].map((label) => (
                    <span
                      key={label}
                      className="rounded-full border border-border bg-surface-container px-2 py-0.5 text-[10px] font-medium text-muted-foreground"
                    >
                      {label}
                    </span>
                  ))}
                </div>
              )}
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
