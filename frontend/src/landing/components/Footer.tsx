import { Github, ArrowRight, Twitter } from 'lucide-react'
import PixelLogo from './PixelLogo'

export default function Footer() {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    if (href.startsWith('#/')) {
      window.location.hash = href
      return
    }

    if (window.location.hash.startsWith('#/architecture')) {
      window.location.hash = href
      return
    }

    if (href.startsWith('#')) {
      e.preventDefault()
      document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <footer className="relative border-t border-white/[0.05] bg-[#050508] pt-20 overflow-hidden">
      {/* Ambient background glows */}
      <div className="pointer-events-none absolute top-0 left-1/4 h-[350px] w-[350px] rounded-full bg-blue-500/5 blur-[120px]" />
      <div className="pointer-events-none absolute bottom-0 right-1/4 h-[400px] w-[400px] rounded-full bg-purple-500/5 blur-[150px]" />

      <div className="mx-auto max-w-7xl px-6 md:px-12">
        {/* Top Section */}
        <div className="flex flex-col md:flex-row justify-between items-start gap-12 pb-16">
          {/* Brand & Tagline */}
          <div className="flex flex-col items-start gap-4 max-w-sm">
            <PixelLogo text="DOCUMIND" className="w-44" />
            <p className="text-[14px] text-zinc-400 leading-relaxed">
              Upload a document, ask a question. Agentic tool-routing decides the rest with grounded citations.
            </p>
            <div className="group relative rounded-full p-[1px] shadow-[0_0_20px_rgba(99,102,241,0.2)] transition-all duration-300 hover:shadow-[0_0_30px_rgba(99,102,241,0.35)] hover:scale-[1.02] active:scale-[0.98] mt-2">
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400/80 via-purple-400/80 to-blue-400/80 opacity-80 transition-opacity duration-300 group-hover:opacity-100" />
              <button
                type="button"
                onClick={() => { window.location.hash = '#/chat/new' }}
                className="relative flex items-center gap-2 rounded-full bg-[#050508]/90 px-5 py-2 text-sm font-medium backdrop-blur-md transition-colors hover:bg-[#050508]/70 cursor-pointer"
              >
                <span className="bg-gradient-to-r from-blue-100 to-purple-200 bg-clip-text text-transparent">
                  Launch workspace
                </span>
                <ArrowRight className="h-3.5 w-3.5 text-purple-200 transition-transform duration-200 group-hover:translate-x-0.5" />
              </button>
            </div>
          </div>

          {/* Clean Links Grid */}
          <div className="grid grid-cols-2 gap-12 sm:gap-20 relative z-10">
            {/* System / Navigation */}
            <div className="flex flex-col gap-4">
              <h3 className="text-[12px] font-mono uppercase tracking-wider text-zinc-300 font-semibold">
                Explore
              </h3>
              <ul className="flex flex-col gap-3">
                <li>
                  <a
                    href="#how-it-works"
                    onClick={(e) => handleNavClick(e, '#how-it-works')}
                    className="text-[14px] text-zinc-400 transition-colors hover:text-white"
                  >
                    How it works
                  </a>
                </li>
                <li>
                  <a
                    href="#metrics"
                    onClick={(e) => handleNavClick(e, '#metrics')}
                    className="text-[14px] text-zinc-400 transition-colors hover:text-white"
                  >
                    System metrics
                  </a>
                </li>
                <li>
                  <a
                    href="#/architecture"
                    onClick={(e) => handleNavClick(e, '#/architecture')}
                    className="text-[14px] text-zinc-400 transition-colors hover:text-white"
                  >
                    Architecture deep-dive
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    onClick={(e) => handleNavClick(e, '#/architecture')}
                    className="text-[14px] text-zinc-400 transition-colors hover:text-white"
                  >
                    Blog & Articles
                  </a>
                </li>
                <li>
                  <a
                    href="#/chat/new"
                    onClick={(e) => handleNavClick(e, '#/chat/new')}
                    className="text-[14px] text-zinc-400 transition-colors hover:text-white"
                  >
                    Chat workspace
                  </a>
                </li>
              </ul>
            </div>

            {/* Connect / Open Source */}
            <div className="flex flex-col gap-4">
              <h3 className="text-[12px] font-mono uppercase tracking-wider text-zinc-300 font-semibold">
                Connect
              </h3>
              <ul className="flex flex-col gap-3">
                <li>
                  <a
                    href="https://github.com/ayush3739/RAG-QA"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-[14px] text-zinc-400 transition-colors hover:text-white"
                  >
                    <Github className="h-4 w-4" /> GitHub
                  </a>
                </li>
                <li>
                  <a
                    href="https://twitter.com/ayush3739"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-[14px] text-zinc-400 transition-colors hover:text-white"
                  >
                    <Twitter className="h-4 w-4" /> Twitter / X
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom copyright bar */}
      <div className="relative z-10 border-t border-white/[0.04] bg-[#030305]/50 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-center px-6 py-6">
          <p className="font-mono text-[12px] text-zinc-500">
            DocuMind — Ayush Maurya · {new Date().getFullYear()}
          </p>
        </div>
      </div>
    </footer>
  )
}
