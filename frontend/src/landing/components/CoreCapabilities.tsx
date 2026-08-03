import { motion } from 'framer-motion'
import { MessageSquare, Search, Terminal } from 'lucide-react'

const capabilities = [
  {
    icon: MessageSquare,
    title: 'Chat with Documents',
    description: 'Upload complex files, process them into hierarchical chunks, and ask questions directly. Seamless integration allows immediate search across indexed documents.',
    badge: 'Interactive Client',
  },
  {
    icon: Search,
    title: 'Research Mode',
    description: 'Synthesize knowledge. Decompose query targets, dynamically execute web lookups when documents lack answers, and compile cited reports.',
    badge: 'Stateful Agentic Loops',
  },
  {
    icon: Terminal,
    title: 'RAG API First',
    description: 'Programmatic REST endpoints return clean structured JSON with sources, confidence score, and intermediate node execution trace logs.',
    badge: 'Developer REST Interface',
    codeSnippet: true
  }
]

export default function CoreCapabilities() {
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
    <section id="capabilities" className="border-t border-white/[0.05] bg-[#09090b] py-28">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-zinc-600">
            Platform Specs
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Core Capabilities
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            Three core pathways built to experiment with dynamic retrieval, async task execution, and structured APIs.
          </p>
        </motion.div>

        {/* 3-Column Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-6 lg:grid-cols-3"
        >
          {capabilities.map((cap) => {
            const Icon = cap.icon
            return (
              <motion.div
                key={cap.title}
                variants={itemVariants}
                className="group flex flex-col justify-between rounded-2xl border border-white/[0.05] bg-white/[0.02] p-8 transition-all duration-300 hover:-translate-y-1.5 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50"
              >
                <div>
                  <div className="mb-8 flex items-center justify-between">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 transition-all duration-300 group-hover:scale-110 group-hover:bg-indigo-500/20 group-hover:text-indigo-300">
                      <Icon className="h-5 w-5" strokeWidth={1.5} />
                    </span>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                      {cap.badge}
                    </span>
                  </div>
                  <h3 className="mb-3 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-indigo-200">{cap.title}</h3>
                  <p className="mb-6 text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">{cap.description}</p>
                </div>

                {cap.codeSnippet ? (
                  <div className="flex h-32 flex-col justify-center rounded-lg border border-white/[0.05] bg-[#060608] p-4 font-mono text-[10px] leading-relaxed text-zinc-500 overflow-x-auto transition-colors duration-300 group-hover:border-white/10 group-hover:bg-[#07070a]">
                    <p><span className="text-emerald-400">POST</span> /api/v1/research</p>
                    <p className="mt-2 text-zinc-400">{'{'}</p>
                    <p className="pl-4">&quot;query&quot;: &quot;Q3 Revenue impact&quot;,</p>
                    <p className="pl-4">&quot;web_fallback&quot;: true</p>
                    <p className="text-zinc-400">{'}'}</p>
                  </div>
                ) : (
                  <div className="h-32 rounded-lg border border-white/[0.05] bg-[#060608]/50 transition-colors duration-300 group-hover:border-white/10" />
                )}
              </motion.div>
            )
          })}
        </motion.div>
      </div>
    </section>
  )
}
