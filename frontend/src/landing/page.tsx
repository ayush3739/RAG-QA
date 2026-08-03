import Navbar from './components/Navbar'
import Hero from './components/Hero'
import WhyDocuMind from './components/WhyDocuMind'
import CoreCapabilities from './components/CoreCapabilities'
import ArchitectureDiagram from './components/ArchitectureDiagram'
import RoutingFlowDiagram from './components/RoutingFlowDiagram'
import FeatureHighlights from './components/FeatureHighlights'
import ArchitectureDecisions from './components/ArchitectureDecisions'
import ChallengesTradeoffs from './components/ChallengesTradeoffs'
import MetricsGrid from './components/MetricsGrid'
import ApiShowcase from './components/ApiShowcase'
import ComparisonTable from './components/ComparisonTable'
import LessonsLearned from './components/LessonsLearned'
import DevelopmentTimeline from './components/DevelopmentTimeline'
import RequestAccessForm from './components/RequestAccessForm'
import Footer from './components/Footer'

export default function LandingPage() {
  return (
    <div className="dark min-h-screen flex flex-col bg-background text-foreground font-sans antialiased">
      <Navbar />
      <main className="flex-1">
        <Hero />
        <WhyDocuMind />
        <CoreCapabilities />
        <ArchitectureDiagram />
        <RoutingFlowDiagram />
        <FeatureHighlights />
        <ArchitectureDecisions />
        <ChallengesTradeoffs />
        <MetricsGrid />
        <ApiShowcase />
        <ComparisonTable />
        <LessonsLearned />
        <DevelopmentTimeline />
        <RequestAccessForm />
      </main>
      <Footer />
    </div>
  )
}
