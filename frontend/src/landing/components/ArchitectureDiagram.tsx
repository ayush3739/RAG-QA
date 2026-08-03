import { motion } from 'framer-motion'

interface ArchLayer {
  step: string;
  name: string;
  description: string;
  chips: string[];
  highlight?: boolean;
}

const layers: ArchLayer[] = [
  {
    step: '01',
    name: 'Frontend Client',
    description: 'React / Vite SPA with Zustand client state management.',
    chips: ['React', 'Vite', 'Zustand', 'Framer Motion'],
  },
  {
    step: '02',
    name: 'FastAPI Gateway',
    description: 'Async API gateway handling requests, ingestion, and SSE event streaming.',
    chips: ['FastAPI', 'Pydantic', 'Uvicorn', 'SSE'],
  },
  {
    step: '03',
    name: 'Router Agent',
    description: 'Dynamic decision layer mapping queries to specialized execution paths.',
    chips: ['LangGraph', 'Tool-Routing', 'Function-Calling'],
    highlight: true,
  },
  {
    step: '04',
    name: 'Retriever Subagent',
    description: 'Executes stateful retrieval, reranking, and confidence threshold validations.',
    chips: ['LangGraph', 'CrossEncoder', 'BM25 Rerank'],
  },
  {
    step: '05',
    name: 'Postgres + pgvector',
    description: 'Unified storage for relational collection metadata and high-dimensional document vectors.',
    chips: ['pgvector', 'PostgreSQL', 'SQLAlchemy'],
  },
  {
    step: '06',
    name: 'LLM Providers',
    description: 'Local and cloud inference endpoints for routing, synthesis, and final answering.',
    chips: ['Ollama', 'OpenAI API', 'Llama 3'],
  },
];

export default function ArchitectureDiagram() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 16 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] },
    },
  }

  return (
    <section id="architecture" className="border-t border-white/[0.05] bg-[#09090b] py-28">
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
            Case Study · 02
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            System Architecture
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            A compact overview of the tech stack — from the user interface down to vector storage and inference engines.
          </p>
        </motion.div>

        {/* Compact 2-Column / 3-Column Bento Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3"
        >
          {layers.map((layer) => (
            <motion.div
              key={layer.name}
              variants={itemVariants}
              className="group flex flex-col justify-between rounded-2xl border border-white/[0.05] bg-white/[0.02] p-6 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50"
            >
              <div>
                <div className="mb-4 flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                    {layer.step}
                  </span>
                  {layer.highlight && (
                    <span className="rounded-full border border-white/[0.08] bg-white/[0.04] px-2 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider text-zinc-400 transition-colors duration-200 group-hover:border-white/20 group-hover:text-zinc-300">
                      Core Agent
                    </span>
                  )}
                </div>

                <h3 className="mb-2 text-[15px] font-semibold text-white transition-colors duration-200 group-hover:text-indigo-200">
                  {layer.name}
                </h3>
                <p className="text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                  {layer.description}
                </p>
              </div>

              {/* Chips */}
              <div className="mt-6 flex flex-wrap gap-1.5 pt-2">
                {layer.chips.map((chip) => (
                  <span
                    key={chip}
                    className="rounded-md border border-white/[0.05] bg-white/[0.03] px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-zinc-400 transition-colors duration-200 group-hover:border-white/10 group-hover:text-zinc-300"
                  >
                    {chip}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
