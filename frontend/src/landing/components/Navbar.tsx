import { useState, useEffect } from 'react'
import { Github, ArrowRight } from 'lucide-react'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const scrollTo = (e: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    e.preventDefault()
    document.querySelector(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <nav
      id="navbar"
      role="navigation"
      aria-label="Main navigation"
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'border-b border-white/[0.06] bg-[#09090b]/90 backdrop-blur-xl'
          : 'bg-transparent'
      }`}
    >
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">

        {/* Logo */}
        <a
          href="#hero"
          onClick={(e) => scrollTo(e, '#hero')}
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
          ].map((link) => (
            <a
              key={link.href}
              href={link.href}
              onClick={(e) => scrollTo(e, link.href)}
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
        <button
          onClick={() => { window.location.hash = '#/chat/new' }}
          className="group hidden items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-1.5 text-xs font-semibold text-white transition-all duration-150 hover:bg-indigo-500 md:flex"
        >
          Launch app
          <ArrowRight className="h-3 w-3 transition-transform duration-150 group-hover:translate-x-0.5" />
        </button>

        {/* Mobile CTA */}
        <button
          onClick={() => { window.location.hash = '#/chat/new' }}
          className="flex items-center gap-1 rounded-full bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white md:hidden"
        >
          Launch
        </button>
      </div>
    </nav>
  )
}
