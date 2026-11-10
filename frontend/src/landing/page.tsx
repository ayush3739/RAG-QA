import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import RoutingFlowDiagram from './components/RoutingFlowDiagram'
import MetricsGrid from './components/MetricsGrid'
import ProductShowcase from './components/ProductShowcase'
import FAQSection from './components/FAQSection'
import Footer from './components/Footer'
import TerminalDemo from './components/TerminalDemo'
import LogoMarquee from './components/LogoMarquee'
// import Testimonials from './components/Testimonials'
import PricingSection from './components/PricingSection'

export default function LandingPage() {
  return (
    <div className="relative min-h-screen flex flex-col bg-background text-foreground antialiased selection:bg-primary/30 selection:text-foreground overflow-hidden">
      {/* Ambient Blue Halo Glows */}
      <div className="pointer-events-none fixed -left-[20%] top-[20%] h-[800px] w-[800px] rounded-full bg-blue-500/10 dark:bg-blue-500/15 blur-[150px] z-0" aria-hidden="true" />
      <div className="pointer-events-none fixed -right-[10%] top-[60%] h-[600px] w-[600px] rounded-full bg-purple-500/10 dark:bg-purple-500/15 blur-[120px] z-0" aria-hidden="true" />

      <div className="relative z-10 flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1">
          {/* Above the fold sections */}
          <Hero />
          <TerminalDemo />
          <LogoMarquee />
          <RoutingFlowDiagram />
          <MetricsGrid />
          <ProductShowcase />

          {/* Pricing */}
          <PricingSection />

          {/* FAQ */}
          <FAQSection />
        </main>
        <Footer />
      </div>
    </div>
  )
}
