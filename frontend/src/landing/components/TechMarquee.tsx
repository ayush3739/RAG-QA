import { motion } from 'framer-motion';

const techNames = [
  'FastAPI',
  'LangChain',
  'Qdrant',
  'Ollama',
  'Tavily',
  'CrossEncoder',
  'BM25',
  'RAGAS',
  'SQLite',
  'Pydantic',
  'Python',
  'Docker',
];

export default function TechMarquee() {
  return (
    <section id="tech-stack" className="relative py-16" style={{ backgroundColor: '#050505' }}>
      <div className="max-w-3xl mx-auto px-6">
        {/* Divider */}
        <div className="linear-gradient-line mb-10" />

        {/* Label */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.6 }}
          className="mb-10"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Built With
          </span>
        </motion.div>

        {/* Tech names row */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex flex-wrap justify-center gap-x-3 gap-y-2 font-mono text-xs text-[#444]"
        >
          {techNames.map((name, idx) => (
            <span
              key={name}
              className="hover:text-[#707070] transition-colors duration-200 cursor-default"
            >
              {name}
              {idx < techNames.length - 1 && <span className="ml-3">·</span>}
            </span>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
