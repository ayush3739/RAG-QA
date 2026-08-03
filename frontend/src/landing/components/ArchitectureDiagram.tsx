import { useRef, useState, useEffect } from 'react';
import {
  motion,
  useScroll,
  useTransform,
  useReducedMotion,
} from 'framer-motion';

interface ArchLayer {
  name: string;
  description: string;
  chips: string[];
  highlight?: boolean;
}

const layers: ArchLayer[] = [
  {
    name: 'Frontend Client',
    description: 'React / Vite SPA with Zustand client state management',
    chips: ['React', 'Vite', 'Zustand', 'Framer Motion'],
  },
  {
    name: 'FastAPI Gateway',
    description: 'Async API gateway handling requests, ingestion, and SSE event streaming',
    chips: ['FastAPI', 'Pydantic', 'Uvicorn', 'SSE'],
  },
  {
    name: 'Router Agent',
    description: 'Dynamic decision layer mapping queries to specialized execution paths',
    chips: ['LangGraph', 'Tool-Routing', 'Function-Calling'],
    highlight: true,
  },
  {
    name: 'Retriever Subagent',
    description: 'Executes stateful retrieval, reranking, and confidence threshold validations',
    chips: ['LangGraph', 'CrossEncoder', 'BM25 Rerank'],
  },
  {
    name: 'Postgres + pgvector',
    description: 'Unified storage for relational collection metadata and high-dimensional document vectors',
    chips: ['pgvector', 'PostgreSQL', 'SQLAlchemy'],
  },
  {
    name: 'LLM Providers',
    description: 'Local and cloud inference endpoints for routing, synthesis, and final answering',
    chips: ['Ollama', 'OpenAI API', 'Llama 3'],
  },
];

/* Staggered reveal variants (runs on first viewport entry) */
const containerVariants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08 },
  },
};

const rowVariants = {
  hidden: { opacity: 0, x: -16 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

export default function ArchitectureDiagram() {
  const sectionRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start 0.78', 'end 0.3'],
  });

  /* ── Connecting line: draws top→bottom with scroll ── */
  const lineScale = useTransform(scrollYProgress, [0, 1], [0, 1]);

  /* ── Data flow particle Y position ── */
  const particleY = useTransform(scrollYProgress, [0, 1], ['0%', '100%']);

  /* ── Per-layer glow: each activates as particle passes through ──
      6 layers → centres at 0, 0.2, 0.4, 0.6, 0.8, 1.0 of scroll range */
  const lg0 = useTransform(scrollYProgress, [-0.10, -0.01, 0.08, 0.16], [0, 1, 1, 0]);
  const lg1 = useTransform(scrollYProgress, [0.10, 0.19, 0.28, 0.36], [0, 1, 1, 0]);
  const lg2 = useTransform(scrollYProgress, [0.30, 0.39, 0.48, 0.56], [0, 1, 1, 0]);
  const lg3 = useTransform(scrollYProgress, [0.50, 0.59, 0.68, 0.76], [0, 1, 1, 0]);
  const lg4 = useTransform(scrollYProgress, [0.70, 0.79, 0.88, 0.96], [0, 1, 1, 0]);
  const lg5 = useTransform(scrollYProgress, [0.90, 0.99, 1.08, 1.16], [0, 1, 1, 0]);

  const layerGlows = [lg0, lg1, lg2, lg3, lg4, lg5];

  /* Particle opacity (fade out at extremes so it doesn't stick) */
  const particleOp = useTransform(
    scrollYProgress,
    [-0.02, 0.04, 0.96, 1.02],
    [0, 1, 1, 0],
  );

  return (
    <section ref={sectionRef} id="architecture" className="relative py-20 md:py-24" style={{ backgroundColor: '#050505' }}>
      <div className="max-w-6xl mx-auto px-6">
        {/* Divider */}
        <div className="linear-gradient-line mb-12" />

        {/* Header */}
        <motion.div
          initial={mounted && !reducedMotion ? { opacity: 0, y: 20 } : false}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.6 }}
          className="mb-10"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Architecture
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Every layer is intentional
          </h2>
        </motion.div>

        {/* Architecture stack */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-60px' }}
          className="relative"
        >
          {/* Desktop left connecting line (draws with scroll) */}
          {mounted && (
            <motion.div
              className="absolute left-6 top-0 bottom-0 w-px origin-top hidden md:block"
              style={{
                scaleY: reducedMotion ? 1 : lineScale,
                background:
                  'linear-gradient(to bottom, rgba(124, 92, 255, 0.35), rgba(63, 216, 196, 0.15))',
                boxShadow: '0 0 8px rgba(124, 92, 255, 0.2)',
              }}
            />
          )}

          {/* ── Mobile left connecting line ── */}
          {mounted && (
            <motion.div
              className="absolute left-2 top-0 bottom-0 w-px origin-top md:hidden"
              style={{
                scaleY: reducedMotion ? 1 : lineScale,
                background:
                  'linear-gradient(to bottom, rgba(124, 92, 255, 0.35), rgba(63, 216, 196, 0.15))',
                boxShadow: '0 0 6px rgba(124, 92, 255, 0.15)',
              }}
            />
          )}

          {/* ── Data flow particle (desktop) ── */}
          {mounted && (
            <motion.div
              className="absolute left-6 -translate-x-1/2 -translate-y-1/2 hidden md:block pointer-events-none z-20"
              style={{
                top: reducedMotion ? '0%' : particleY,
                opacity: reducedMotion ? 0 : particleOp,
              }}
            >
              <div className="relative">
                <div className="absolute w-6 h-6 -left-[5px] -top-[5px] rounded-full bg-[#7C5CFF]/20 blur-md" />
                <div
                  className="relative w-2 h-2 rounded-full bg-[#7C5CFF]"
                  style={{
                    boxShadow:
                      '0 0 8px rgba(124, 92, 255, 0.9), 0 0 16px rgba(124, 92, 255, 0.4)',
                  }}
                />
              </div>
            </motion.div>
          )}

          {/* ── Data flow particle (mobile) ── */}
          {mounted && (
            <motion.div
              className="absolute left-2 -translate-x-1/2 -translate-y-1/2 md:hidden pointer-events-none z-20"
              style={{
                top: reducedMotion ? '0%' : particleY,
                opacity: reducedMotion ? 0 : particleOp,
              }}
            >
              <div className="relative">
                <div className="absolute w-5 h-5 -left-[4px] -top-[4px] rounded-full bg-[#7C5CFF]/20 blur-md" />
                <div
                  className="relative w-2 h-2 rounded-full bg-[#7C5CFF]"
                  style={{
                    boxShadow: '0 0 8px rgba(124, 92, 255, 0.9)',
                  }}
                />
              </div>
            </motion.div>
          )}

          {/* ── Layers ── */}
          {layers.map((layer, idx) => (
            <div key={layer.name}>
              {/* Thin gradient connector between layers */}
              {idx > 0 && (
                <div className="h-px bg-gradient-to-r from-transparent via-white/5 to-transparent" />
              )}

              <motion.div variants={rowVariants} className="relative">
                {/* Scroll-linked activation glow */}
                <motion.div
                  className="absolute -inset-[1px] rounded-2xl pointer-events-none z-20"
                  style={{
                    opacity: mounted ? (reducedMotion
                      ? layer.highlight
                        ? 0.4
                        : 0
                      : layerGlows[idx]) : 0,
                    boxShadow: layer.highlight
                      ? '0 0 25px rgba(124, 92, 255, 0.3), 0 0 50px rgba(124, 92, 255, 0.1), inset 0 1px 0 rgba(124, 92, 255, 0.08)'
                      : '0 0 20px rgba(124, 92, 255, 0.2), 0 0 40px rgba(124, 92, 255, 0.06)',
                  }}
                >
                  {/* Pulsing inner glow */}
                  {!reducedMotion && (
                    <motion.div
                      className="absolute inset-0 rounded-2xl"
                      animate={{
                        boxShadow: [
                          '0 0 15px rgba(124, 92, 255, 0.2)',
                          '0 0 30px rgba(124, 92, 255, 0.4), 0 0 60px rgba(124, 92, 255, 0.12)',
                          '0 0 15px rgba(124, 92, 255, 0.2)',
                        ],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: 'easeInOut',
                      }}
                    />
                  )}
                </motion.div>

                {/* Layer card */}
                <div
                  className={`linear-gradient-border p-5 md:p-6 flex flex-col md:flex-row md:items-center md:justify-between gap-3 md:gap-6 ${
                    layer.highlight ? 'bg-[#7C5CFF]/[0.03]' : ''
                  }`}
                >
                  <div>
                    <h3 className="font-mono text-sm text-[#EDEDED]">
                      {layer.name}
                    </h3>
                    <p className="text-xs text-[#707070] mt-0.5">
                      {layer.description}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {layer.chips.map((chip) => (
                      <span
                        key={chip}
                        className="font-mono text-[10px] text-[#444] bg-white/[0.03] rounded px-2 py-0.5 border border-white/5"
                      >
                        {chip}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
