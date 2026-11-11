import { motion } from 'framer-motion'
import { Search, ShieldCheck, CheckSquare, Eye, type LucideIcon } from 'lucide-react'

interface BentoCell {
  id: string
  icon: LucideIcon
  title: string
  description: string
  mdColSpan: string
}

const bentoCells: BentoCell[] = [
  {
    id: 'hybrid-search',
    icon: Search,
    title: 'Hybrid Search',
    description: 'Combines lexical BM25 matching and high-dimensional cosine vector similarity. Blends results using Reciprocal Rank Fusion (RRF), then filters the top candidate list with a CrossEncoder model for context ranking.',
    mdColSpan: 'md:col-span-2',
  },
  {
    id: 'trust-layer',
    icon: ShieldCheck,
    title: 'Trust Layer',
    description: 'Attaches metadata citation markers (document name, page index, source section) to every LLM claim. Validates responses via confidence scores to decline out-of-scope queries.',
    mdColSpan: 'md:col-span-1',
  },
  {
    id: 'evaluation',
    icon: CheckSquare,
    title: 'Evaluation Suite',
    description: 'Integrates automated evaluations (RAGAS framework) directly into the test suite. Assesses groundedness and response accuracy against predefined golden QA evaluation sets.',
    mdColSpan: 'md:col-span-1',
  },
  {
    id: 'observability',
    icon: Eye,
    title: 'Observability',
    description: 'Implements comprehensive tracing using LangSmith logging. Tracks subagent decision histories, intermediate query reformulations, tool execution latencies, and raw prompt values.',
    mdColSpan: 'md:col-span-2',
  },
]

export default function FeatureHighlights() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] as const },
    },
  }

  return (
    <section id="features" className="border-t border-border bg-background py-28">
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
            Case Study · 03
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-foreground md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Deep Engine Features
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-muted-foreground">
            The underlying machinery that ensures high-quality retrieval and strict guardrails before the response reaches the user.
          </p>
        </motion.div>

        {/* Bento Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-4 md:grid-cols-3"
        >
          {bentoCells.map((cell) => {
            const Icon = cell.icon
            return (
              <motion.div
                key={cell.id}
                variants={itemVariants}
                className={`group flex flex-col justify-between rounded-2xl border border-border bg-card/60 dark:bg-white/[0.02] p-8 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-card dark:hover:bg-white/[0.04] hover:shadow-xl hover:shadow-primary/5 dark:hover:shadow-black/50 ${cell.mdColSpan}`}
              >
                <div>
                  <div className="mb-6 flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary transition-all duration-300 group-hover:scale-110 group-hover:bg-primary/20">
                    <Icon className="h-5 w-5" strokeWidth={1.5} />
                  </div>
                  <h3 className="mb-3 text-[15px] font-medium text-foreground transition-colors duration-200 group-hover:text-primary">{cell.title}</h3>
                  <p className="text-[13px] leading-relaxed text-muted-foreground transition-colors duration-200 group-hover:text-foreground/80">{cell.description}</p>
                </div>
              </motion.div>
            )
          })}
        </motion.div>

        {/* JSON Schema Block */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="group mt-4 rounded-2xl border border-primary/20 bg-primary/[0.03] p-8 transition-all duration-300 hover:border-primary/40 md:p-10"
        >
          <h3 className="mb-6 text-[15px] font-medium text-foreground transition-colors duration-200 group-hover:text-primary">
            Guaranteed Structured Output
          </h3>
          <div className="overflow-x-auto rounded-lg border border-border bg-surface-container-low dark:bg-[#060608] p-5 font-mono text-[11px] leading-relaxed text-muted-foreground transition-colors duration-300 group-hover:border-primary/30">
            <span className="text-emerald-500 font-semibold">POST</span> /api/v1/research<br /><br />
            {'{'}<br />
            <span className="text-muted-foreground/70">  // Strict Pydantic response model validation</span><br />
            {'  '}<span className="text-primary font-medium">&quot;answer&quot;</span>: <span className="text-emerald-500">&quot;The 2024 climate report indicates...&quot;</span>,<br />
            {'  '}<span className="text-primary font-medium">&quot;citations&quot;</span>: [<br />
            {'    '}{'{'} <span className="text-primary font-medium">&quot;doc_id&quot;</span>: <span className="text-emerald-500">&quot;1a2b&quot;</span>, <span className="text-primary font-medium">&quot;chunk&quot;</span>: <span className="text-amber-500">4</span> {'}'}<br />
            {'  '}],<br />
            {'  '}<span className="text-primary font-medium">&quot;confidence_score&quot;</span>: <span className="text-amber-500">0.92</span><br />
            {'}'}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
