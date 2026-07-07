import { useRef, useState, useEffect } from 'react'
import {
  motion,
  useInView,
  useReducedMotion,
  useSpring,
  AnimatePresence,
} from 'framer-motion'
import {
  Search,
  ShieldCheck,
  CheckSquare,
  Eye,
  type LucideIcon,
} from 'lucide-react'

interface BentoCell {
  id: string
  icon: LucideIcon
  title: string
  description: string
  hiddenDetail: string
  mdColSpan: string
  mdRowSpan: string
}

const bentoCells: BentoCell[] = [
  {
    id: 'hybrid-search',
    icon: Search,
    title: 'Hybrid Search',
    description:
      'Combines lexical BM25 matching and high-dimensional cosine vector similarity. Blends results using Reciprocal Rank Fusion (RRF), then filters the top candidate list with a CrossEncoder model for context ranking.',
    hiddenDetail: 'Fused search preserves both keyword precision and latent semantic similarity.',
    mdColSpan: 'md:col-span-2',
    mdRowSpan: 'md:row-span-1',
  },
  {
    id: 'trust-layer',
    icon: ShieldCheck,
    title: 'Trust Layer',
    description:
      'Attaches metadata citation markers (document name, page index, source section) to every LLM claim. Validates responses via confidence scores to decline out-of-scope queries.',
    hiddenDetail: 'Prevents confidently wrong LLM answers by enforcing a strict confidence threshold.',
    mdColSpan: 'md:col-span-1',
    mdRowSpan: 'md:row-span-1',
  },
  {
    id: 'evaluation',
    icon: CheckSquare,
    title: 'Evaluation Suite',
    description:
      'Integrates automated evaluations (RAGAS framework) directly into the test suite. Assesses groundedness and response accuracy against predefined golden QA evaluation sets.',
    hiddenDetail: 'Automated test suites catch regression loops during vector size changes.',
    mdColSpan: 'md:col-span-1',
    mdRowSpan: 'md:row-span-1',
  },
  {
    id: 'observability',
    icon: Eye,
    title: 'Observability',
    description:
      'Implements comprehensive tracing using LangSmith logging. Tracks subagent decision histories, intermediate query reformulations, tool execution latencies, and raw prompt values.',
    hiddenDetail: 'Provides full visibility into router paths and latency bottlenecks.',
    mdColSpan: 'md:col-span-2',
    mdRowSpan: 'md:row-span-1',
  },
]

function ApiCodeBlock() {
  return (
    <div className="linear-code p-5 mt-4 text-xs sm:text-[13px] leading-relaxed overflow-x-auto">
      <pre className="text-[#707070]">
        <span className="text-[#3FD8C4]">POST</span>{' '}
        <span className="text-[#888]">/api/v1/research</span>
        {'\n'}
        {'{'}
        {'\n'}
        {'  '}
        <span className="text-[#7C5CFF]">&quot;topic&quot;</span>
        <span className="text-[#555]">: </span>
        <span className="text-[#3FD8C4]">&quot;Climate findings 2024&quot;</span>
        <span className="text-[#555]">,</span>
        {'\n'}
        {'  '}
        <span className="text-[#7C5CFF]">&quot;collection&quot;</span>
        <span className="text-[#555]">: </span>
        <span className="text-[#3FD8C4]">&quot;climate_report&quot;</span>
        <span className="text-[#555]">,</span>
        {'\n'}
        {'  '}
        <span className="text-[#7C5CFF]">&quot;include_web&quot;</span>
        <span className="text-[#555]">: </span>
        <span className="text-[#F5B942]">true</span>
        <span className="text-[#555]">,</span>
        {'\n'}
        {'  '}
        <span className="text-[#7C5CFF]">&quot;confidence&quot;</span>
        <span className="text-[#555]">: </span>
        <span className="text-[#F5B942]">0.87</span>
        {'\n'}
        {'}'}
      </pre>
    </div>
  )
}

const gridVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.08,
    },
  },
}

const cellVariants = {
  hidden: {
    opacity: 0,
    y: 20,
    filter: 'blur(4px)',
  },
  visible: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: {
      duration: 0.6,
      ease: [0.23, 1, 0.32, 1] as [number, number, number, number],
    },
  },
}

interface BentoCellComponentProps {
  cell: BentoCell
  shouldAnimate: boolean
}

function BentoCellComponent({ cell, shouldAnimate }: BentoCellComponentProps) {
  const [isHovered, setIsHovered] = useState(false)
  const borderGlowOpacity = useSpring(0, { stiffness: 300, damping: 30 })

  useEffect(() => {
    borderGlowOpacity.set(isHovered ? 0.8 : 0)
  }, [isHovered, borderGlowOpacity])

  const Icon = cell.icon

  return (
    <motion.div
      variants={shouldAnimate ? cellVariants : undefined}
      className={`relative overflow-hidden ${cell.mdColSpan} ${cell.mdRowSpan} ${
        cell.mdRowSpan.includes('row-span-2') ? 'p-8' : 'p-6'
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={
        shouldAnimate
          ? {
              scale: 1.02,
              zIndex: 10,
              transition: { type: 'spring', stiffness: 400, damping: 25 },
            }
          : undefined
      }
    >
      {/* Animated purple gradient border glow */}
      <motion.div
        className="absolute inset-0 pointer-events-none z-10"
        style={{
          opacity: borderGlowOpacity,
          background:
            'linear-gradient(135deg, rgba(124,92,255,0.4) 0%, rgba(63,216,196,0.15) 100%)',
          WebkitMask:
            'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
          WebkitMaskComposite: 'xor',
          maskComposite: 'exclude',
          padding: '1px',
          borderRadius: 'inherit',
        }}
        aria-hidden
      />

      {/* Icon — subtle pulse on hover */}
      <motion.div
        whileHover={
          shouldAnimate
            ? {
                scale: 1.15,
                rotate: 5,
                transition: { type: 'spring', stiffness: 400, damping: 20 },
              }
            : undefined
        }
        className="inline-flex"
      >
        <Icon
          className="w-5 h-5 text-[#7C5CFF] opacity-60"
          strokeWidth={1.5}
        />
      </motion.div>

      <h3 className="mt-4 text-lg font-medium text-[#EDEDED]">
        {cell.title}
      </h3>

      <p className="mt-2 text-sm text-[#707070] leading-relaxed">
        {cell.description}
      </p>

      {/* Hidden detail — revealed on hover */}
      <AnimatePresence>
        {isHovered && cell.hiddenDetail && (
          <motion.p
            key={`${cell.id}-detail`}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
            className="text-xs text-[#707070] leading-relaxed mt-3"
          >
            {cell.hiddenDetail}
          </motion.p>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function FeatureHighlights() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-60px' })
  const prefersReducedMotion = useReducedMotion()
  const shouldAnimate = !prefersReducedMotion
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <section
      id="features"
      ref={sectionRef}
      className="relative py-20 md:py-24 linear-mesh-bg overflow-hidden"
      style={{ backgroundColor: '#050505' }}
    >
      {/* Ambient orb — subtle breathing pulse */}
      <motion.div
        className="linear-orb w-[400px] h-[400px] bg-[#3FD8C4]/[0.05] bottom-[-150px] right-[-100px]"
        animate={
          shouldAnimate
            ? { opacity: [0.5, 0.8, 0.5], scale: [1, 1.05, 1] }
            : undefined
        }
        transition={{
          duration: 6,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        style={{ position: 'absolute' }}
      />

      <div className="relative max-w-6xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={mounted && shouldAnimate ? { opacity: 0, y: 16 } : false}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
          className="mb-10"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Production Features
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Engineered for Reliability
          </h2>
        </motion.div>

        {/* Bento Grid */}
        <motion.div
          variants={mounted && shouldAnimate ? gridVariants : undefined}
          initial={mounted && shouldAnimate ? 'hidden' : false}
          animate={isInView ? 'visible' : (mounted && shouldAnimate ? 'hidden' : false)}
          className="linear-bento linear-shimmer grid grid-cols-1 md:grid-cols-3"
        >
          {bentoCells.map((cell) => (
            <BentoCellComponent
              key={cell.id}
              cell={cell}
              shouldAnimate={shouldAnimate}
            />
          ))}
        </motion.div>
      </div>
    </section>
  )
}
