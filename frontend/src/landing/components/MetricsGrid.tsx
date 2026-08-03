import { useRef, useState } from 'react';
import { motion, useInView, useReducedMotion } from 'framer-motion';
import { metrics } from '../data/metrics';
import { useCountUp } from '../hooks/useCountUp';

function MetricItem({
  metric,
  index,
}: {
  metric: (typeof metrics)[number];
  index: number;
}) {
  const { count, ref: countRef } = useCountUp(metric.value, 2000, true);
  const [isHovered, setIsHovered] = useState(false);
  const prefersReduced = useReducedMotion();
  const itemRef = useRef<HTMLDivElement>(null);

  const displayVal = (val: number) => {
    return (val / 100).toFixed(2);
  };

  // For reduced motion, show everything immediately without animation
  if (prefersReduced) {
    return (
      <div className="py-8 md:py-12">
        <div className="flex items-baseline gap-1">
          <span className="text-6xl md:text-8xl font-extralight tracking-tighter linear-gradient-text-accent">
            {displayVal(metric.value)}
          </span>
        </div>
        <p className="mt-3 text-sm font-medium text-[#EDEDED]">
          {metric.label}
        </p>
        <p className="mt-1 text-xs text-[#444]">{metric.description}</p>
        <div className="mt-4 h-[2px] w-full rounded-full bg-[#1A1A1A]">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#7C5CFF] to-[#3FD8C4]"
            style={{ width: `${metric.value}%` }}
          />
        </div>
      </div>
    );
  }

  return (
    <motion.div
      ref={itemRef}
      initial={{ opacity: 0, y: 40, rotate: 2 }}
      whileInView={{ opacity: 1, y: 0, rotate: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{
        type: 'spring',
        stiffness: 100,
        damping: 20,
        delay: index * 0.1,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="py-8 md:py-12"
    >
      {/* Dramatic number reveal: scale + blur → clear */}
      <div ref={countRef} className="flex items-baseline gap-1">
        <motion.span
          initial={{ scale: 0.8, filter: 'blur(10px)' }}
          whileInView={{ scale: 1, filter: 'blur(0px)' }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{
            type: 'spring',
            stiffness: 100,
            damping: 15,
          }}
          whileHover={{ scale: 1.05 }}
          className="text-6xl md:text-8xl font-extralight tracking-tighter linear-gradient-text-accent origin-left"
        >
          {displayVal(count)}
        </motion.span>
      </div>

      <p className="mt-3 text-sm font-medium text-[#EDEDED]">{metric.label}</p>
      <p className="mt-1 text-xs text-[#444]">{metric.description}</p>

      {/* Animated progress bar */}
      <div className="mt-4 h-[2px] w-full rounded-full bg-[#1A1A1A]">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${metric.value}%` }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{
            type: 'spring',
            stiffness: 60,
            damping: 20,
            delay: index * 0.1 + 0.3,
          }}
          className="h-full rounded-full bg-gradient-to-r from-[#7C5CFF] to-[#3FD8C4]"
          style={{
            boxShadow: isHovered
              ? '0 0 12px rgba(124, 92, 255, 0.5), 0 0 4px rgba(63, 216, 196, 0.3)'
              : 'none',
            transition: 'box-shadow 0.3s ease',
          }}
        />
      </div>
    </motion.div>
  );
}

export default function MetricsGrid() {
  const prefersReduced = useReducedMotion();

  return (
    <section id="metrics" className="relative py-20 md:py-24" style={{ backgroundColor: '#050505' }}>
      <div className="max-w-5xl mx-auto px-6">
        {/* Divider */}
        <div className="linear-gradient-line mb-12" />

        {/* Header */}
        <motion.div
          initial={prefersReduced ? false : { opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.6 }}
          className="mb-10"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Evaluation
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text">
            Quality Benchmarks
          </h2>
        </motion.div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {metrics.map((metric, i) => (
            <MetricItem key={metric.label} metric={metric} index={i} />
          ))}
        </div>

        {/* Target caption */}
        <p className="mt-8 text-center text-xs font-mono text-[#444]">
          * Target values calculated dynamically from the RAGAS offline evaluation suite golden dataset.
        </p>
      </div>
    </section>
  );
}
