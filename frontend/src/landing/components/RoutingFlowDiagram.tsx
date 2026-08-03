import { useRef, useState, useEffect } from 'react';
import {
  motion,
  useScroll,
  useTransform,
  useReducedMotion,
} from 'framer-motion';

const steps = [
  {
    number: 'STEP 01',
    title: 'User Query',
    description: 'Any query — simple factual verification, context QA, or complex synthesis.',
    highlight: false,
  },
  {
    number: 'STEP 02',
    title: 'Router Agent',
    description: 'Dynamic classifier determines the optimal route in one pass, minimizing latency.',
    highlight: true,
  },
  {
    number: 'STEP 03',
    title: 'Execute Routes',
    description: 'Dispatches to direct LLM reasoning, document search, web search, or stateful research.',
    highlight: false,
  },
  {
    number: 'STEP 04',
    title: 'Structured Answer',
    description: 'Returns source citation keys, confidence validation score, and execution traces.',
    highlight: false,
  },
];

const branches = [
  { label: 'Direct', color: '#F5B942', endX: 15 },
  { label: 'Docs', color: '#7C5CFF', endX: 85 },
  { label: 'Web', color: '#3FD8C4', endX: 155 },
  { label: 'Research', color: '#A78BFA', endX: 225 },
];

export default function RoutingFlowDiagram() {
  const sectionRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true) }, []);

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start 0.82', 'end 0.35'],
  });

  const progressWidth = useTransform(scrollYProgress, [0, 1], ['0%', '100%']);
  const p1Left = useTransform(scrollYProgress, [0, 0.33], ['12.5%', '37.5%']);
  const p2Left = useTransform(scrollYProgress, [0.33, 0.66], ['37.5%', '62.5%']);
  const p3Left = useTransform(scrollYProgress, [0.66, 1.0], ['62.5%', '87.5%']);
  const p1Op = useTransform(scrollYProgress, [0.0, 0.04, 0.30, 0.33], [0, 1, 1, 0]);
  const p2Op = useTransform(scrollYProgress, [0.33, 0.37, 0.63, 0.66], [0, 1, 1, 0]);
  const p3Op = useTransform(scrollYProgress, [0.66, 0.70, 0.97, 1.0], [0, 1, 1, 0]);
  const glow0 = useTransform(scrollYProgress, [-0.05, 0.04, 0.22, 0.30], [0, 0.7, 0.7, 0]);
  const glow1 = useTransform(scrollYProgress, [0.20, 0.30, 0.48, 0.58], [0, 1, 1, 0]);
  const glow2 = useTransform(scrollYProgress, [0.48, 0.58, 0.78, 0.86], [0, 0.7, 0.7, 0]);
  const glow3 = useTransform(scrollYProgress, [0.78, 0.88, 1.0, 1.05], [0, 0.7, 0.7, 0]);
  const c0Op = useTransform(scrollYProgress, [-0.05, 0.05], [0.12, 1]);
  const c1Op = useTransform(scrollYProgress, [0.20, 0.30], [0.12, 1]);
  const c2Op = useTransform(scrollYProgress, [0.48, 0.58], [0.12, 1]);
  const c3Op = useTransform(scrollYProgress, [0.78, 0.88], [0.12, 1]);
  const c0Y = useTransform(scrollYProgress, [-0.05, 0.05], [20, 0]);
  const c1Y = useTransform(scrollYProgress, [0.20, 0.30], [20, 0]);
  const c2Y = useTransform(scrollYProgress, [0.48, 0.58], [20, 0]);
  const c3Y = useTransform(scrollYProgress, [0.78, 0.88], [20, 0]);
  const branchOp = useTransform(scrollYProgress, [0.30, 0.36, 0.44, 0.52], [0, 1, 1, 0]);
  const branchDraw = useTransform(scrollYProgress, [0.30, 0.42], [0, 1]);
  const mobileProgH = useTransform(scrollYProgress, [0, 1], ['0%', '100%']);
  const mobileParticleTop = useTransform(scrollYProgress, [0, 1], ['0%', '100%']);

  const glows = [glow0, glow1, glow2, glow3];
  const cOps = [c0Op, c1Op, c2Op, c3Op];
  const cYs = [c0Y, c1Y, c2Y, c3Y];

  const particles = [
    { left: reducedMotion ? '25%' : p1Left, opacity: reducedMotion ? 0 : p1Op },
    { left: reducedMotion ? '50%' : p2Left, opacity: reducedMotion ? 0 : p2Op },
    { left: reducedMotion ? '75%' : p3Left, opacity: reducedMotion ? 0 : p3Op },
  ];

  return (
    <section ref={sectionRef} id="how-it-works" className="relative py-20 md:py-24" style={{ backgroundColor: '#050505' }}>
      <div className="max-w-6xl mx-auto px-6">
        <div className="linear-gradient-line mb-12" />

        <motion.div
          initial={mounted && !reducedMotion ? { opacity: 0, y: 20 } : false}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.6 }}
          className="mb-10"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">How It Works</span>
          <h2 className="mt-4 text-4xl md:text-6xl font-light tracking-tight linear-gradient-text-subtle">
            One decision. The right tool.
          </h2>
          <p className="mt-3 text-[#707070] max-w-xl leading-relaxed text-sm md:text-base">
            A single function-calling pass routes every query to the optimal tool.
          </p>
        </motion.div>

        {/* Desktop Flow */}
        <div className="hidden md:block relative pb-12">
          <div className="absolute top-[44px] left-[12.5%] right-[12.5%] h-px bg-white/[0.04] rounded-full" />
          {mounted && (
            <motion.div
              className="absolute top-[43px] left-[12.5%] h-[2px] origin-left rounded-full"
              style={{
                width: reducedMotion ? '100%' : progressWidth,
                background: 'linear-gradient(90deg, #7C5CFF 0%, #A78BFA 50%, #3FD8C4 100%)',
                boxShadow: '0 0 10px rgba(124, 92, 255, 0.5), 0 0 20px rgba(124, 92, 255, 0.2)',
              }}
            />
          )}

          {mounted && particles.map((p, i) => (
            <motion.div key={i} className="absolute top-[40px] -translate-x-1/2 pointer-events-none z-20" style={p}>
              <div className="relative">
                <div className="absolute w-5 h-5 -left-[3px] -top-[3px] rounded-full bg-[#7C5CFF]/25 blur-sm" />
                <div className="relative w-2.5 h-2.5 rounded-full bg-[#7C5CFF]"
                  style={{ boxShadow: '0 0 8px rgba(124, 92, 255, 0.9), 0 0 16px rgba(124, 92, 255, 0.4)' }} />
              </div>
            </motion.div>
          ))}

          <div className="grid grid-cols-4 gap-6 relative z-10">
            {steps.map((step, i) => (
              <motion.div
                key={step.number}
                className="relative"
                style={{
                  opacity: mounted ? (reducedMotion ? 1 : cOps[i]) : 1,
                  y: mounted ? (reducedMotion ? 0 : cYs[i]) : 0,
                }}
              >
                <motion.div
                  className="absolute -inset-[1px] rounded-2xl pointer-events-none z-20"
                  style={{
                    opacity: mounted ? (reducedMotion ? (step.highlight ? 0.5 : 0) : glows[i]) : 0,
                    boxShadow: step.highlight
                      ? '0 0 30px rgba(124, 92, 255, 0.35), 0 0 60px rgba(124, 92, 255, 0.12), inset 0 1px 0 rgba(124, 92, 255, 0.1)'
                      : '0 0 20px rgba(124, 92, 255, 0.25), 0 0 40px rgba(124, 92, 255, 0.08)',
                  }}
                >
                  {!reducedMotion && (
                    <motion.div
                      className="absolute inset-0 rounded-2xl"
                      animate={{ boxShadow: ['0 0 20px rgba(124, 92, 255, 0.3)', '0 0 35px rgba(124, 92, 255, 0.5), 0 0 70px rgba(124, 92, 255, 0.15)', '0 0 20px rgba(124, 92, 255, 0.3)'] }}
                      transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  )}
                </motion.div>

                <div className={`linear-gradient-border p-5 h-full relative transition-transform duration-500 ${step.highlight ? 'md:scale-105' : ''}`}>
                  {step.highlight && (
                    <div className="absolute inset-0 bg-[#7C5CFF] opacity-[0.03] rounded-2xl pointer-events-none" />
                  )}
                  <span className="font-mono text-[10px] text-[#7C5CFF] tracking-widest uppercase">{step.number}</span>
                  <h3 className="mt-2 text-base font-medium text-[#EDEDED]">{step.title}</h3>
                  <p className="mt-1.5 text-xs text-[#707070] leading-relaxed">{step.description}</p>
                </div>

                {step.highlight && (
                  <motion.div
                    className="absolute top-full left-1/2 -translate-x-1/2 mt-3 pointer-events-none z-30"
                    style={{ opacity: reducedMotion ? 0 : (mounted ? branchOp : 0) }}
                  >
                    <svg width="240" height="58" viewBox="0 0 240 58" fill="none" className="overflow-visible">
                      {branches.map((b) => (
                        <g key={b.label}>
                          <motion.path
                            d={`M120,0 Q${(120 + b.endX) / 2},18 ${b.endX},44`}
                            pathLength={1}
                            stroke={b.color}
                            strokeWidth="1"
                            fill="none"
                            opacity="0.5"
                            style={{ pathLength: reducedMotion ? 1 : branchDraw }}
                          />
                          <circle cx={b.endX} cy={44} r="3" fill={b.color} opacity="0.35" />
                          <text x={b.endX} y="56" textAnchor="middle" fill="#444" fontSize="8" fontFamily="monospace">
                            {b.label}
                          </text>
                        </g>
                      ))}
                    </svg>
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Mobile Flow */}
        <div className="md:hidden relative pl-10">
          <div className="absolute left-3 top-0 bottom-0 w-px bg-white/[0.04]" />
          {mounted && (
            <motion.div
              className="absolute left-[11px] top-0 w-[2px] origin-top rounded-full"
              style={{
                height: reducedMotion ? '100%' : mobileProgH,
                background: 'linear-gradient(180deg, #7C5CFF 0%, #A78BFA 50%, #3FD8C4 100%)',
                boxShadow: '0 0 10px rgba(124, 92, 255, 0.5)',
              }}
            />
          )}
          {mounted && (
            <motion.div
              className="absolute left-[11px] -translate-x-1/2 -translate-y-1/2 pointer-events-none z-20"
              style={{ top: reducedMotion ? '0%' : mobileParticleTop }}
            >
              <div className="relative">
                <div className="absolute w-5 h-5 -left-[3px] -top-[3px] rounded-full bg-[#7C5CFF]/25 blur-sm" />
                <div className="relative w-2.5 h-2.5 rounded-full bg-[#7C5CFF]"
                  style={{ boxShadow: '0 0 8px rgba(124, 92, 255, 0.9), 0 0 16px rgba(124, 92, 255, 0.4)' }} />
              </div>
            </motion.div>
          )}

          {steps.map((step, i) => (
            <motion.div
              key={step.number}
              className="relative mb-4 last:mb-0"
              style={{
                opacity: mounted ? (reducedMotion ? 1 : cOps[i]) : 1,
                y: mounted ? (reducedMotion ? 0 : cYs[i]) : 0,
              }}
            >
              <div className="absolute -left-[25px] top-5 w-2 h-2 rounded-full bg-[#0A0A0A] border border-white/10" />
              <motion.div
                className="absolute -inset-[1px] rounded-2xl pointer-events-none z-20"
                style={{
                  opacity: mounted ? (reducedMotion ? 0 : glows[i]) : 0,
                  boxShadow: step.highlight
                    ? '0 0 25px rgba(124, 92, 255, 0.3), 0 0 50px rgba(124, 92, 255, 0.1)'
                    : '0 0 20px rgba(124, 92, 255, 0.25), 0 0 40px rgba(124, 92, 255, 0.08)',
                }}
              />
              <div className="linear-gradient-border p-5 relative">
                {step.highlight && (
                  <div className="absolute inset-0 bg-[#7C5CFF] opacity-[0.03] rounded-2xl pointer-events-none" />
                )}
                <span className="font-mono text-[10px] text-[#7C5CFF] tracking-widest uppercase">{step.number}</span>
                <h3 className="mt-2 text-base font-medium text-[#EDEDED]">{step.title}</h3>
                <p className="mt-1.5 text-xs text-[#707070] leading-relaxed">{step.description}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
