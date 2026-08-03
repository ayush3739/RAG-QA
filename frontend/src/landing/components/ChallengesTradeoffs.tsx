import { motion } from 'framer-motion'
import { ShieldAlert, Zap, Server, Brain } from 'lucide-react'

const challenges = [
  {
    icon: Brain,
    title: 'Why not LangGraph everywhere?',
    description: 'Only the retriever subagent needs execution loops (Rerank -> Re-query -> Consolidate). Keeping the core router simple and linear minimizes latency overhead.',
    tag: 'Orchestration Complexity'
  },
  {
    icon: Zap,
    title: 'Why not a larger model?',
    description: 'Using small local models (like Llama 3 8B) for classification and routing handles tasks at a fraction of the token cost and context latency compared to GPT-4.',
    tag: 'Cost & Latency Tradeoff'
  },
  {
    icon: Server,
    title: 'Why not external vector DBs?',
    description: 'Choosing pgvector in Postgres keeps deployment simple. Eliminates metadata synchronization delay and keeps the local Docker network footprint small.',
    tag: 'Data Sync Overhead'
  },
  {
    icon: ShieldAlert,
    title: 'Why confidence is not enough',
    description: 'High model confidence score can still hallucinate under edge prompt injections. Implemented a groundedness evaluator to scan contexts before output.',
    tag: 'Hallucination Mitigation'
  }
]

export default function ChallengesTradeoffs() {
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
    <section id="challenges" className="border-t border-white/[0.05] bg-[#09090b] py-28">
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
            Case Study · 05
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Challenges &amp; Tradeoffs
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            Real engineering constraints require deliberate tradeoffs. Here is how I approached scalability and cost boundaries.
          </p>
        </motion.div>

        {/* 2-Column Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-6 md:grid-cols-2"
        >
          {challenges.map((chal) => {
            const Icon = chal.icon
            return (
              <motion.div
                key={chal.title}
                variants={itemVariants}
                className="group flex flex-col justify-between rounded-2xl border border-white/[0.05] bg-white/[0.02] p-8 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50"
              >
                <div>
                  <div className="mb-6 flex items-center justify-between">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.04] text-zinc-400 transition-all duration-300 group-hover:scale-110 group-hover:bg-indigo-500/10 group-hover:text-indigo-300">
                      <Icon className="h-5 w-5" strokeWidth={1.5} />
                    </span>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                      {chal.tag}
                    </span>
                  </div>
                  <h3 className="mb-3 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-indigo-200">{chal.title}</h3>
                  <p className="text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">{chal.description}</p>
                </div>
              </motion.div>
            )
          })}
        </motion.div>
      </div>
    </section>
  )
}
