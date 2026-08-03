import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import RoutingFlowDiagram from './components/RoutingFlowDiagram'
import MetricsGrid from './components/MetricsGrid'
import ProductShowcase from './components/ProductShowcase'
import Footer from './components/Footer'

// Case study sections — shown only when expanded
import WhyDocuMind from './components/WhyDocuMind'
import CoreCapabilities from './components/CoreCapabilities'
import ArchitectureDiagram from './components/ArchitectureDiagram'
import FeatureHighlights from './components/FeatureHighlights'
import ArchitectureDecisions from './components/ArchitectureDecisions'
import ChallengesTradeoffs from './components/ChallengesTradeoffs'
import ApiShowcase from './components/ApiShowcase'
import ComparisonTable from './components/ComparisonTable'
import LessonsLearned from './components/LessonsLearned'
import DevelopmentTimeline from './components/DevelopmentTimeline'

export default function LandingPage() {
  const [showCaseStudy, setShowCaseStudy] = useState(false)

  return (
    <div className="min-h-screen flex flex-col bg-[#09090b] text-white antialiased">
      <Navbar />
      <main className="flex-1">
        {/* ── Above the fold: 4 clean sections ── */}
        <Hero />
        <RoutingFlowDiagram />
        <MetricsGrid />
        <ProductShowcase />

        {/* ── Case study toggle ── */}
        <div className="border-y border-white/[0.05] bg-[#09090b] py-14">
          <div className="mx-auto flex max-w-6xl flex-col items-center gap-4 px-6 text-center">
            <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-zinc-600">
              Engineering deep-dive
            </p>
            <h3
              className="text-xl font-medium tracking-[-0.02em] text-white"
              style={{ fontFamily: "'DM Sans', sans-serif" }}
            >
              Read the full case study
            </h3>
            <p className="max-w-md text-[14px] text-zinc-600">
              Architecture decisions, routing logic, LLM provider abstraction, lessons learned, and everything in between.
            </p>
            <button
              onClick={() => setShowCaseStudy((v) => !v)}
              className="group mt-2 flex items-center gap-2 rounded-full border border-white/[0.08] px-5 py-2 text-[13px] font-medium text-zinc-400 transition-all duration-150 hover:border-white/[0.16] hover:text-white"
            >
              {showCaseStudy ? 'Hide case study' : 'Read the case study'}
              <ChevronDown
                className={`h-4 w-4 transition-transform duration-300 ${showCaseStudy ? 'rotate-180' : ''}`}
              />
            </button>
          </div>
        </div>

        {/* ── Case study content ── */}
        {showCaseStudy && (
          <div className="bg-[#09090b]">
            <WhyDocuMind />
            <CoreCapabilities />
            <ArchitectureDiagram />
            <FeatureHighlights />
            <ArchitectureDecisions />
            <ChallengesTradeoffs />
            <ApiShowcase />
            <ComparisonTable />
            <LessonsLearned />
            <DevelopmentTimeline />
          </div>
        )}
      </main>
      <Footer />
    </div>
  )
}
