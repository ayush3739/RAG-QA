import { useRef, useState, useEffect } from 'react'
import { motion, useInView, useReducedMotion } from 'framer-motion'
import { MessageSquare, Search, Terminal } from 'lucide-react'

export default function CoreCapabilities() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-60px' })
  const prefersReduced = useReducedMotion()
  const shouldAnimate = !prefersReduced
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  const capabilities = [
    {
      icon: MessageSquare,
      title: 'Chat with Documents',
      description: 'Upload complex files, process them into hierarchical chunks, and ask questions directly. Seamless integration allows immediate search across indexed documents.',
      badge: 'Interactive Client',
      screenshot: '/chat.png',
      alt: 'Document chat panel'
    },
    {
      icon: Search,
      title: 'Research Mode',
      description: 'Synthesize knowledge. Decompose query targets, dynamically execute web lookups when documents lack answers, and compile cited reports.',
      badge: 'Stateful Agentic Loops',
      screenshot: '/dashboard.png',
      alt: 'Research execution panel'
    },
    {
      icon: Terminal,
      title: 'RAG API First',
      description: 'Programmatic REST endpoints return clean structured JSON with sources, confidence score, and intermediate node execution trace logs.',
      badge: 'Developer REST Interface',
      codeSnippet: true
    }
  ]

  return (
    <section
      id="capabilities"
      ref={sectionRef}
      className="relative py-20 md:py-24 linear-mesh-bg overflow-hidden"
      style={{ backgroundColor: '#050505' }}
    >
      {/* Background decoration */}
      <div
        className="linear-orb w-[500px] h-[500px] bg-[#7C5CFF]/[0.05]"
        style={{ position: 'absolute', top: '20%', left: '50%', transform: 'translateX(-50%)' }}
      />

      <div className="relative max-w-6xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={mounted && shouldAnimate ? { opacity: 0, y: 16 } : false}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
          className="mb-16 text-center md:text-left"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            Platform Specs
          </span>
          <h2 className="mt-4 text-3xl md:text-5xl font-light tracking-tight linear-gradient-text-subtle">
            Core Capabilities
          </h2>
          <p className="mt-4 text-[#707070] max-w-xl text-base leading-relaxed">
            Three core pathways built to experiment with dynamic retrieval, async task execution, and structured APIs.
          </p>
        </motion.div>

        {/* 3-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {capabilities.map((cap, i) => {
            const Icon = cap.icon
            return (
              <motion.div
                key={cap.title}
                initial={mounted && shouldAnimate ? { opacity: 0, y: 30 } : false}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: i * 0.1, ease: [0.23, 1, 0.32, 1] }}
                className="linear-gradient-border p-6 md:p-8 flex flex-col justify-between h-full bg-white/[0.01]"
              >
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <span className="p-2.5 rounded-xl border border-white/5 bg-white/[0.03] text-[#7C5CFF]">
                      <Icon className="w-5 h-5" strokeWidth={1.5} />
                    </span>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-[#444]">
                      {cap.badge}
                    </span>
                  </div>

                  <h3 className="text-lg font-medium text-white mb-3">{cap.title}</h3>
                  <p className="text-xs leading-relaxed text-[#707070] mb-6">{cap.description}</p>
                </div>

                {cap.codeSnippet ? (
                  <div className="linear-code p-4 text-[10px] leading-relaxed overflow-x-auto text-[#707070] font-mono h-40 flex flex-col justify-center">
                    <p><span className="text-[#3FD8C4]">POST</span> /api/v1/research</p>
                    <p className="text-[#333]">{'{'}</p>
                    <p className="pl-3"><span className="text-[#7C5CFF]">&quot;topic&quot;</span>: <span className="text-[#3FD8C4]">&quot;climate_change&quot;</span>,</p>
                    <p className="pl-3"><span className="text-[#7C5CFF]">&quot;confidence_threshold&quot;</span>: <span className="text-[#F5B942]">0.8</span></p>
                    <p className="text-[#333]">{'},'}</p>
                    <p className="text-[#444]">// returns sources, confidence, trace</p>
                  </div>
                ) : (
                  <div className="relative rounded-lg overflow-hidden border border-white/5 bg-black/40 h-40">
                    <img
                      src={cap.screenshot}
                      alt={cap.alt}
                      className="absolute inset-0 w-full h-full object-cover object-top opacity-65 hover:opacity-85 transition-opacity duration-300"
                    />
                  </div>
                )}
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
