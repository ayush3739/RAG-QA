import { Github, ArrowRight, Twitter } from 'lucide-react'
import { navigate } from '../../lib/navigation'
import PixelLogo from './PixelLogo'
import { ShinyButton } from '../../components/ui/shiny-button'

export default function Footer() {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    if (href.startsWith('/')) {
      e.preventDefault()
      navigate(href)
      return
    }

    if (window.location.pathname !== '/') {
      e.preventDefault()
      navigate('/' + href)
      return
    }

    if (href.startsWith('#')) {
      e.preventDefault()
      document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <footer className="relative border-t border-border bg-background pt-20">
      <div className="mx-auto max-w-7xl px-6 md:px-12">
        {/* Top Section */}
        <div className="flex flex-col md:flex-row justify-between items-start gap-12 pb-16">
          {/* Brand & Tagline */}
          <div className="flex flex-col items-start gap-4 max-w-sm">
            <PixelLogo text="KnowVex" className="w-44" />
            <p className="text-sm text-muted-foreground leading-relaxed">
              Upload a document, ask a question. Agentic tool-routing decides the rest with grounded citations.
            </p>
            <div className="mt-2">
              <ShinyButton
                onClick={() => { navigate('/chat/new') }}
                size="default"
                variant="brand"
                animated={true}
              >
                <span>Launch workspace</span>
                <ArrowRight className="h-4 w-4 transition-transform duration-150 group-hover:translate-x-0.5" aria-hidden="true" />
              </ShinyButton>
            </div>
          </div>

          {/* Clean Links Grid */}
          <div className="grid grid-cols-2 gap-12 sm:gap-20 relative z-10">
            {/* System / Navigation */}
            <div className="flex flex-col gap-4">
              <h3 className="text-xs font-semibold text-foreground">
                Explore
              </h3>
              <ul className="flex flex-col gap-3">
                <li>
                  <a
                    href="#how-it-works"
                    onClick={(e) => handleNavClick(e, '#how-it-works')}
                    className="text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    How it works
                  </a>
                </li>
                <li>
                  <a
                    href="#metrics"
                    onClick={(e) => handleNavClick(e, '#metrics')}
                    className="text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    System metrics
                  </a>
                </li>
                <li>
                  <a
                    href="/architecture"
                    onClick={(e) => handleNavClick(e, '/architecture')}
                    className="text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    Architecture deep-dive
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    onClick={(e) => handleNavClick(e, '/architecture')}
                    className="text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    Blog and Articles
                  </a>
                </li>
                <li>
                  <a
                    href="/chat/new"
                    onClick={(e) => handleNavClick(e, '/chat/new')}
                    className="text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    Chat workspace
                  </a>
                </li>
              </ul>
            </div>

            {/* Connect / Open Source */}
            <div className="flex flex-col gap-4">
              <h3 className="text-xs font-semibold text-foreground">
                Connect
              </h3>
              <ul className="flex flex-col gap-3">
                <li>
                  <a
                    href="https://github.com/ayush3739/RAG-QA"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    <Github className="h-4 w-4" aria-hidden="true" /> GitHub
                  </a>
                </li>
                <li>
                  <a
                    href="https://twitter.com/ayush3739"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    <Twitter className="h-4 w-4" aria-hidden="true" /> Twitter / X
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom copyright bar */}
      <div className="relative z-10 border-t border-border bg-surface-container-low">
        <div className="mx-auto flex max-w-7xl items-center justify-center px-6 py-6">
          <p className="text-xs text-muted-foreground">
            KnowVex: Ayush Maurya ({new Date().getFullYear()})
          </p>
        </div>
      </div>
    </footer>
  )
}
