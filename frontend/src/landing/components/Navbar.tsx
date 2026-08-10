import { useState, useEffect } from 'react'
import { Github, ArrowRight } from 'lucide-react'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    // If it's a structural route (like #/architecture)
    if (href.startsWith('#/')) {
      window.location.hash = href
      return
    }

    // If we are currently on the architecture page, and clicking an anchor link,
    // we need to navigate back to the landing page and jump there natively.
    if (window.location.hash.startsWith('#/architecture')) {
      window.location.hash = href;
      return;
    }

    // Otherwise, we are on the landing page, so smooth scroll
    e.preventDefault()
    document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <nav
      id="navbar"
      role="navigation"
      aria-label="Main navigation"
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'border-b border-white/[0.04] bg-[#060608]/40 backdrop-blur-2xl supports-[backdrop-filter]:bg-[#060608]/20'
          : 'bg-transparent'
      }`}
    >
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">

        {/* Logo */}
        <a
          href="#hero"
          onClick={(e) => handleClick(e, '#hero')}
          className="flex items-center gap-2 transition-opacity hover:opacity-70"
        >
          <span
            className="font-mono text-sm font-medium tracking-tight text-white"
            style={{ fontFamily: "'DM Sans', sans-serif", letterSpacing: '-0.02em' }}
          >
            DocuMind
          </span>
        </a>

        {/* Center nav links — desktop */}
        <div className="hidden items-center gap-8 md:flex">
          {[
            { label: 'How it works', href: '#how-it-works' },
            { label: 'Metrics', href: '#metrics' },
            { label: 'Architecture', href: '#/architecture' },
          ].map((link) => (
            <a
              key={link.href}
              href={link.href}
              onClick={(e) => handleClick(e, link.href)}
              className="text-[13px] font-medium text-zinc-500 transition-colors duration-150 hover:text-white"
            >
              {link.label}
            </a>
          ))}
          <a
            href="https://github.com/ayush3739/RAG-QA"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-[13px] font-medium text-zinc-500 transition-colors duration-150 hover:text-white"
          >
            <Github className="h-3.5 w-3.5" />
            GitHub
          </a>
        </div>

        {/* CTA */}
        {/* CTA */}
        <div className="group hidden md:flex relative rounded-full p-[1px] shadow-[0_0_15px_rgba(99,102,241,0.2)] transition-all duration-300 hover:shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:scale-[1.02] active:scale-[0.98]">
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400/80 via-purple-400/80 to-blue-400/80 opacity-80 transition-opacity duration-300 group-hover:opacity-100" />
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="relative flex items-center gap-1.5 rounded-full bg-[#060608]/90 px-4 py-1.5 text-xs font-medium backdrop-blur-md transition-colors hover:bg-[#060608]/70"
          >
            <span className="bg-gradient-to-r from-blue-100 to-purple-200 bg-clip-text text-transparent">
              Launch app
            </span>
            <ArrowRight className="h-3 w-3 text-purple-200 transition-transform duration-150 group-hover:translate-x-0.5" />
          </button>
        </div>

        {/* Mobile CTA */}
        <div className="group flex md:hidden relative rounded-full p-[1px] shadow-[0_0_15px_rgba(99,102,241,0.2)] transition-all duration-300 hover:shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:scale-[1.02] active:scale-[0.98]">
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400/80 via-purple-400/80 to-blue-400/80 opacity-80 transition-opacity duration-300 group-hover:opacity-100" />
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="relative flex items-center gap-1 rounded-full bg-[#060608]/90 px-3 py-1.5 text-xs font-medium backdrop-blur-md transition-colors hover:bg-[#060608]/70"
          >
            <span className="bg-gradient-to-r from-blue-100 to-purple-200 bg-clip-text text-transparent">
              Launch
            </span>
          </button>
        </div>
      </div>
    </nav>
  )
}
