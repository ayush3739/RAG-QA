import { motion } from 'framer-motion';

export default function HonestScoping() {
  return (
    <section id="roadmap" className="relative py-16" style={{ backgroundColor: '#050505' }}>
      <div className="mx-auto max-w-3xl px-6">
        {/* Divider */}
        <div className="linear-gradient-line mb-10" />

        {/* Heading */}
        <motion.h2
          className="text-lg font-medium text-[#EDEDED]"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5 }}
        >
          Honest scoping
        </motion.h2>

        {/* What it is */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-30px' }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mt-6"
        >
          <p className="font-mono text-xs text-[#7C5CFF]">What it is:</p>
          <p className="mt-2 text-sm leading-relaxed text-[#707070]">
            A single-pass tool router with deterministic escalation. Measured quality via RAGAS CI. RAG-specific guardrails at every layer.
          </p>
        </motion.div>

        {/* What it isn't */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-30px' }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-6"
        >
          <p className="font-mono text-xs text-[#444]">What it isn&apos;t:</p>
          <p className="mt-2 text-sm leading-relaxed text-[#444]">
            Not an autonomous multi-step agent. No retry loops, no self-correction, no multi-goal planning. That&apos;s a separate project.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
