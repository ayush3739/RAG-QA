import { useRef, useState, useEffect, useCallback } from 'react'
import {
  motion,
  useInView,
  useReducedMotion,
  useMotionValue,
  useTransform,
  useSpring,
  useMotionTemplate,
} from 'framer-motion'

interface ProductPanelProps {
  src: string
  alt: string
  label: string
  tiltDirection: 'left' | 'right'
  isInView: boolean
  shouldAnimate: boolean
  index: number
  mounted: boolean
}

function ProductPanel({
  src,
  alt,
  label,
  tiltDirection,
  isInView,
  shouldAnimate,
  index,
  mounted,
}: ProductPanelProps) {
  const [isHoverCapable, setIsHoverCapable] = useState(false)

  useEffect(() => {
    setIsHoverCapable(window.matchMedia('(hover: hover)').matches)
  }, [])

  const mouseX = useMotionValue(0.5)
  const mouseY = useMotionValue(0.5)
  const glareX = useMotionValue(0)
  const glareY = useMotionValue(0)
  const glareOpacity = useMotionValue(0)
  const springGlareOpacity = useSpring(glareOpacity, { stiffness: 300, damping: 30 })

  const baseRotateX = 4
  const baseRotateY = tiltDirection === 'left' ? -2 : 2

  const rotateX = useTransform(mouseY, [0, 1], [baseRotateX + 5, baseRotateX - 5])
  const rotateY = useTransform(mouseX, [0, 1], [baseRotateY - 5, baseRotateY + 5])

  const springRotateX = useSpring(rotateX, { stiffness: 200, damping: 25 })
  const springRotateY = useSpring(rotateY, { stiffness: 200, damping: 25 })

  const glareBg = useMotionTemplate`radial-gradient(600px circle at ${glareX}px ${glareY}px, rgba(124,92,255,0.07), transparent 40%)`

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isHoverCapable) return
      const rect = e.currentTarget.getBoundingClientRect()
      const nx = (e.clientX - rect.left) / rect.width
      const ny = (e.clientY - rect.top) / rect.height
      mouseX.set(nx)
      mouseY.set(ny)
      glareX.set(e.clientX - rect.left)
      glareY.set(e.clientY - rect.top)
      glareOpacity.set(1)
    },
    [isHoverCapable, mouseX, mouseY, glareX, glareY, glareOpacity],
  )

  const handleMouseLeave = useCallback(() => {
    mouseX.set(0.5)
    mouseY.set(0.5)
    glareOpacity.set(0)
  }, [mouseX, mouseY, glareOpacity])

  const entranceDelay = 0.15 * index + 0.15

  return (
    <motion.div
      initial={mounted && shouldAnimate ? { opacity: 0, scale: 0.92 } : false}
      animate={isInView ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.8, delay: entranceDelay, ease: [0.23, 1, 0.32, 1] }}
      whileHover={shouldAnimate ? { y: -8, transition: { duration: 0.35, ease: 'easeOut' } } : undefined}
    >
      <div style={{ perspective: '1200px' }}>
        <motion.div
          style={{
            rotateX: isHoverCapable && shouldAnimate ? springRotateX : undefined,
            rotateY: isHoverCapable && shouldAnimate ? springRotateY : undefined,
            transformStyle: 'preserve-3d',
          }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <div
            className={`aspect-[4/3] ${
              tiltDirection === 'right'
                ? 'linear-product-panel linear-product-panel-tilt-right'
                : 'linear-product-panel'
            }`}
          >
            <img
              src={src}
              alt={alt}
              className="object-cover rounded-[15px] w-full h-full"
            />
            {isHoverCapable && (
              <motion.div
                style={{ background: glareBg, opacity: springGlareOpacity }}
                className="pointer-events-none absolute inset-0 rounded-[15px] z-10"
              />
            )}
          </div>
        </motion.div>
      </div>

      <motion.span
        initial={mounted && shouldAnimate ? { opacity: 0, y: 8 } : false}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, delay: entranceDelay + 0.35, ease: [0.23, 1, 0.32, 1] }}
        className="block font-mono text-xs text-[#444] mt-4 ml-1"
      >
        {label}
      </motion.span>
    </motion.div>
  )
}

export default function ProductShowcase() {
  const sectionRef = useRef<HTMLElement>(null)
  const isInView = useInView(sectionRef, { once: true, margin: '-80px' })
  const prefersReducedMotion = useReducedMotion()
  const shouldAnimate = !prefersReducedMotion
  const [mounted, setMounted] = useState(false)

  useEffect(() => { setMounted(true) }, [])

  return (
    <section
      id="product"
      ref={sectionRef}
      className="relative py-20 md:py-24 linear-mesh-bg overflow-hidden"
      style={{ backgroundColor: '#050505' }}
    >
      <div
        className="linear-orb w-[500px] h-[500px] bg-[#7C5CFF]/[0.07]"
        style={{ position: 'absolute', top: '-200px', left: '50%', transform: 'translateX(-50%)' }}
      />

      <div className="relative max-w-6xl mx-auto px-6">
        <motion.div
          initial={mounted && shouldAnimate ? { opacity: 0, y: 20 } : false}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
          className="mb-10"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">Showcase</span>
          <h2 className="mt-4 text-4xl md:text-6xl font-light tracking-tight linear-gradient-text-subtle">
            Explore the Application
          </h2>
          <p className="mt-5 text-[#707070] text-base max-w-lg leading-relaxed">
            A fully functional sandbox project — from document uploads to routed reasoning and research reports.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ProductPanel
            src="/dashboard.png"
            alt="DocuMind Dashboard showing activity charts and document collections"
            label="Dashboard"
            tiltDirection="left"
            isInView={isInView}
            shouldAnimate={shouldAnimate}
            index={0}
            mounted={mounted}
          />
          <ProductPanel
            src="/chat.png"
            alt="DocuMind Research Chat interface with confidence scores and citations"
            label="Research Chat"
            tiltDirection="right"
            isInView={isInView}
            shouldAnimate={shouldAnimate}
            index={1}
            mounted={mounted}
          />
        </div>
      </div>
    </section>
  )
}
