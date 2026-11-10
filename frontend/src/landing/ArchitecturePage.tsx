import { useEffect } from 'react'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import WhyKnowVex from './components/WhyKnowVex'
import CoreCapabilities from './components/CoreCapabilities'
import ArchitectureDiagram from './components/ArchitectureDiagram'
import FeatureHighlights from './components/FeatureHighlights'
import ArchitectureDecisions from './components/ArchitectureDecisions'
import ChallengesTradeoffs from './components/ChallengesTradeoffs'
import ApiShowcase from './components/ApiShowcase'
import ComparisonTable from './components/ComparisonTable'
import LessonsLearned from './components/LessonsLearned'
import DevelopmentTimeline from './components/DevelopmentTimeline'

export default function ArchitecturePage() {
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  return (
    <div className="relative min-h-screen flex flex-col bg-background text-foreground antialiased selection:bg-primary/30 selection:text-foreground overflow-hidden">
      {/* Fixed Ambient Glows for the whole page */}
      <div className="pointer-events-none fixed -left-[20%] top-[20%] h-[800px] w-[800px] rounded-full bg-blue-500/10 dark:bg-blue-500/15 blur-[150px] z-0" aria-hidden="true" />
      <div className="pointer-events-none fixed -right-[10%] top-[60%] h-[600px] w-[600px] rounded-full bg-purple-500/10 dark:bg-purple-500/15 blur-[120px] z-0" aria-hidden="true" />
      
      <div className="relative z-10 flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1 pt-24">
          <div className="mx-auto max-w-4xl px-6 md:px-12 mb-16 mt-8 text-center">
            <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-primary mb-4">
              Engineering Deep-Dive
            </p>
            <h1 
              className="text-4xl md:text-5xl font-bold tracking-tight text-foreground mb-6"
              style={{ fontFamily: "'DM Sans', sans-serif" }}
            >
              The KnowVex Architecture
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto leading-relaxed">
              An in-depth look at the decisions, trade-offs, and technologies powering a low-latency, agentic RAG platform.
            </p>
          </div>
          
          <div>
            <WhyKnowVex />
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
        </main>
        <Footer />
      </div>
    </div>
  )
}
