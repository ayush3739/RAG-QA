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
import Testimonials from './components/Testimonials'
import PricingSection from './components/PricingSection'

export default function LandingPage() {

  return (
    <div className="relative min-h-screen flex flex-col bg-[#09090b] text-white antialiased overflow-hidden">
      {/* Fixed Ambient Glows for the whole page */}
      <div className="pointer-events-none fixed -left-[20%] top-[20%] h-[800px] w-[800px] rounded-full bg-blue-500/5 blur-[150px] z-0" />
      <div className="pointer-events-none fixed -right-[10%] top-[60%] h-[600px] w-[600px] rounded-full bg-purple-500/5 blur-[120px] z-0" />
      
      <div className="relative z-10 flex flex-col min-h-screen">
        <Navbar />
      <main className="flex-1">
        {/* ── Above the fold: 4 clean sections ── */}
        <Hero />
        <TerminalDemo />
        <LogoMarquee />
        <RoutingFlowDiagram />
        <MetricsGrid />
        <ProductShowcase />
        {/* ── Testimonials ── */}
        <Testimonials />

        {/* ── Pricing ── */}
        <PricingSection />

        {/* ── FAQ ── */}
        <FAQSection />
      </main>
      <Footer />
      </div>
    </div>
  )
}
