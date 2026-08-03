import { motion } from 'framer-motion'
import { Award, Compass, Key } from 'lucide-react'

const lessons = [
  {
    icon: Award,
    title: 'Retrieval beats bigger models.',
    description: 'Prioritizing search strategies like RRF hybrid merging, parent-child chunking, and CrossEncoder rerankers yields far cleaner answers than simply upgrading parameters from GPT-3.5 to GPT-4.',
    tag: 'Data Quality'
  },
  {
    icon: Compass,
    title: 'Evaluation prevents regressions.',
    description: 'Without programmatic metrics (RAGAS), modifications to vector chunk dimensions, metadata parsing, or system prompt values cause silent, untraceable quality drops.',
    tag: 'Benchmarking'
  },
  {
    icon: Key,
    title: 'Autonomous agents need constraints.',
    description: 'Unbounded multi-agent loops tend to iterate endlessly, ballooning token costs and latency. Applying strict decision classification boundaries preserves deterministic system stability.',
    tag: 'Control Systems'
  }
]

export default function LessonsLearned() {
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
    <section id="lessons" className="border-t border-white/[0.05] bg-[#09090b] py-28">
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
            Case Study · 08
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Lessons Learned
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            Key engineering insights gained while designing, testing, and debugging stateful RAG systems.
          </p>
        </motion.div>

        {/* 3-Column Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-6 md:grid-cols-3"
        >
          {lessons.map((les) => {
            const Icon = les.icon
            return (
              <motion.div
                key={les.title}
                variants={itemVariants}
                className="group flex flex-col justify-between rounded-2xl border border-white/[0.05] bg-white/[0.02] p-8 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1.5 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50"
              >
                <div>
                  <div className="mb-6 flex items-center justify-between">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.04] text-zinc-400 transition-all duration-300 group-hover:scale-110 group-hover:bg-indigo-500/10 group-hover:text-indigo-300">
                      <Icon className="h-5 w-5" strokeWidth={1.5} />
                    </span>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                      {les.tag}
                    </span>
                  </div>
                  <h3 className="mb-3 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-indigo-200">{les.title}</h3>
                  <p className="text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">{les.description}</p>
                </div>
              </motion.div>
            )
          })}
        </motion.div>
      </div>
    </section>
  )
}
