import { motion } from 'framer-motion'
import { HelpCircle, Layers, Database, Cpu, Compass } from 'lucide-react'

const decisions = [
  {
    icon: Layers,
    title: 'Why LangGraph?',
    description: 'Enables stateful retrieval loops, dynamic router fallback thresholds, and conditional transitions that LangChain chains cannot support naturally.',
    tag: 'Orchestration',
    colSpan: 'md:col-span-1'
  },
  {
    icon: Database,
    title: 'Why pgvector?',
    description: 'Keeps dense vector embeddings stored directly alongside relational document metadata. Eliminates synchronization delay and keeps the operational footprint small.',
    tag: 'Vector Storage',
    colSpan: 'md:col-span-1'
  },
  {
    icon: Cpu,
    title: 'Why FastAPI?',
    description: 'Provides high-concurrency async runtimes, native streaming for Server-Sent Events, and automated OpenAPI validation out of the box.',
    tag: 'API Layer',
    colSpan: 'md:col-span-1'
  },
  {
    icon: Compass,
    title: 'Why RAGAS?',
    description: 'Replaces qualitative, subjective "vibe-checks" with measurable mathematical metrics (Faithfulness, Precision, Recall) executed against golden test datasets.',
    tag: 'Evaluation',
    colSpan: 'md:col-span-1'
  },
  {
    icon: HelpCircle,
    title: 'Why pgvector instead of Qdrant?',
    description: 'Migrated from Qdrant to pgvector to leverage a single database. Simplifies local deployment and operations while allowing direct SQL relational filters inside vector searches.',
    tag: 'Migration Tradeoff',
    colSpan: 'md:col-span-2',
    highlight: true
  }
]

export default function ArchitectureDecisions() {
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
    <section id="decisions" className="border-t border-white/[0.05] bg-[#09090b] py-28">
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
            Case Study · 04
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Architecture Decisions
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            Key engineering choices made to optimize reliability, latency, and codebase maintainability.
          </p>
        </motion.div>

        {/* Bento Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3"
        >
          {decisions.map((dec) => {
            const Icon = dec.icon
            return (
              <motion.div
                key={dec.title}
                variants={itemVariants}
                className={`group flex flex-col justify-between rounded-2xl border border-white/[0.05] bg-white/[0.02] p-8 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50 ${dec.colSpan}`}
              >
                <div>
                  <div className="mb-6 flex items-center justify-between">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.04] text-zinc-400 transition-all duration-300 group-hover:scale-110 group-hover:bg-indigo-500/10 group-hover:text-indigo-300">
                      <Icon className="h-5 w-5" strokeWidth={1.5} />
                    </span>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                      {dec.tag}
                    </span>
                  </div>
                  <h3 className="mb-3 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-indigo-200">{dec.title}</h3>
                  <p className="text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">{dec.description}</p>
                </div>
              </motion.div>
            )
          })}
        </motion.div>
      </div>
    </section>
  )
}
