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
      transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] },
    },
  }

  return (
    <section id="features" className="border-t border-white/[0.05] bg-[#09090b] py-28">
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
            Case Study · 03
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Deep Engine Features
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
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
                className={`group flex flex-col justify-between rounded-2xl border border-white/[0.05] bg-white/[0.02] p-8 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50 ${cell.mdColSpan}`}
              >
                <div>
                  <div className="mb-6 flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.04] text-zinc-400 transition-all duration-300 group-hover:scale-110 group-hover:bg-indigo-500/10 group-hover:text-indigo-300">
                    <Icon className="h-5 w-5" strokeWidth={1.5} />
                  </div>
                  <h3 className="mb-3 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-indigo-200">{cell.title}</h3>
                  <p className="text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">{cell.description}</p>
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
          className="group mt-4 rounded-2xl border border-indigo-500/10 bg-indigo-500/[0.02] p-8 transition-all duration-300 hover:border-indigo-500/20 hover:bg-indigo-500/[0.03] md:p-10"
        >
          <h3 className="mb-6 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-indigo-300">
            Guaranteed Structured Output
          </h3>
          <div className="overflow-x-auto rounded-lg border border-white/[0.05] bg-[#060608] p-5 font-mono text-[11px] leading-relaxed text-zinc-400 transition-colors duration-300 group-hover:border-indigo-500/20">
            <span className="text-emerald-400">POST</span> /api/v1/research<br /><br />
            {'{'}<br />
            <span className="text-zinc-500">  // Strict Pydantic response model validation</span><br />
            {'  '}<span className="text-indigo-400">&quot;answer&quot;</span>: <span className="text-emerald-400">&quot;The 2024 climate report indicates...&quot;</span>,<br />
            {'  '}<span className="text-indigo-400">&quot;citations&quot;</span>: [<br />
            {'    '}{'{'} <span className="text-indigo-400">&quot;doc_id&quot;</span>: <span className="text-emerald-400">&quot;1a2b&quot;</span>, <span className="text-indigo-400">&quot;chunk&quot;</span>: <span className="text-amber-400">4</span> {'}'}<br />
            {'  '}],<br />
            {'  '}<span className="text-indigo-400">&quot;confidence_score&quot;</span>: <span className="text-amber-400">0.92</span><br />
            {'}'}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
