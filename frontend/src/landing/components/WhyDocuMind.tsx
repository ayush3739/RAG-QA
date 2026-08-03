import { motion } from 'framer-motion'

export default function WhyDocuMind() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.15 },
    },
  }

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] },
    },
  }

  return (
    <section id="why-exists" className="border-t border-white/[0.05] bg-[#09090b] py-28">
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
            Case Study · 01
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Why DocuMind Exists
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            Evaluating the trade-offs of modern RAG pipelines and addressing standard architectural limitations.
          </p>
        </motion.div>

        {/* Two Column Layout: The Problem vs The Approach */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 gap-8 md:grid-cols-2"
        >
          {/* Column 1: The Problem */}
          <motion.div
            variants={cardVariants}
            className="group rounded-2xl border border-white/[0.05] bg-white/[0.02] p-8 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50"
          >
            <h3 className="mb-8 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-red-300">
              The Problem
            </h3>
            <p className="mb-6 font-mono text-[10px] uppercase tracking-wider text-zinc-600">
              Most &quot;Chat with PDF&quot; systems
            </p>
            <ul className="space-y-6">
              <li className="flex items-start gap-4">
                <span className="font-mono text-xs text-red-400">01</span>
                <div>
                  <h4 className="text-[13px] font-medium text-zinc-300">Retrieve on every query</h4>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                    Keyword-based and general knowledge questions waste database operations, adding unnecessary retrieval latency.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-4">
                <span className="font-mono text-xs text-red-400">02</span>
                <div>
                  <h4 className="text-[13px] font-medium text-zinc-300">Hallucinate confidently</h4>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                    Standard systems output answers regardless of context quality, lacking fallbacks or groundedness validation.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-4">
                <span className="font-mono text-xs text-red-400">03</span>
                <div>
                  <h4 className="text-[13px] font-medium text-zinc-300">Cannot reason across sources</h4>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                    Locked into single-pass vector database crawls without web search fallbacks or structured research synthesis.
                  </p>
                </div>
              </li>
            </ul>
          </motion.div>

          {/* Column 2: The Approach */}
          <motion.div
            variants={cardVariants}
            className="group rounded-2xl border border-white/[0.05] bg-white/[0.02] p-8 backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.03] hover:shadow-xl hover:shadow-black/50"
          >
            <h3 className="mb-8 text-[15px] font-medium text-white transition-colors duration-200 group-hover:text-indigo-200">
              The Approach
            </h3>
            <p className="mb-6 font-mono text-[10px] uppercase tracking-wider text-zinc-600">
              Agentic RAG Architecture
            </p>
            <ul className="space-y-6">
              <li className="flex items-start gap-4">
                <span className="font-mono text-xs text-indigo-400">01</span>
                <div>
                  <h4 className="text-[13px] font-medium text-zinc-300">Intelligent Routing</h4>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                    Queries bypass the vector DB entirely if the LLM detects it can answer directly or needs web search.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-4">
                <span className="font-mono text-xs text-indigo-400">02</span>
                <div>
                  <h4 className="text-[13px] font-medium text-zinc-300">Multi-Hop Reasoning</h4>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                    Complex research queries spawn a multi-step workflow, extracting context from both vector store and web.
                  </p>
                </div>
              </li>
              <li className="flex items-start gap-4">
                <span className="font-mono text-xs text-indigo-400">03</span>
                <div>
                  <h4 className="text-[13px] font-medium text-zinc-300">Strict Citations</h4>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-500 transition-colors duration-200 group-hover:text-zinc-400">
                    Responses are forced into structured JSON formats that require source citation and confidence scores.
                  </p>
                </div>
              </li>
            </ul>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
