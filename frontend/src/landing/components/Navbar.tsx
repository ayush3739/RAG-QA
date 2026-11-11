import { useState, useEffect } from 'react'
import { Github, ArrowRight, Sun, Moon } from 'lucide-react'
import { navigate } from '../../lib/navigation'
import { useStore } from '../../store/useStore'
import { ShinyButton } from '../../components/ui/shiny-button'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const theme = useStore((s) => s.theme)
  const setTheme = useStore((s) => s.setTheme)

  const isArchitecture = typeof window !== 'undefined' && 
    (window.location.pathname === '/architecture' || window.location.hash.includes('architecture'))

  const toggleTheme = () => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark'
    setTheme(nextTheme)
  }

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault()

    if (href === '/' || href === '#hero') {
      if (window.location.pathname !== '/') {
        navigate('/')
      }
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }

    if (href === '/architecture') {
      if (window.location.pathname !== '/architecture') {
        navigate('/architecture')
      }
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }

    if (href.startsWith('#')) {
      if (window.location.pathname !== '/') {
        navigate('/')
        setTimeout(() => {
          document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
        }, 150)
      } else {
        document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
      }
      return
    }

    navigate(href)
  }

  const navLinks = [
    ...(isArchitecture ? [{ label: 'Home', href: '/' }] : []),
    { label: 'How it works', href: '#how-it-works' },
    { label: 'Metrics', href: '#metrics' },
    { label: 'Architecture', href: '/architecture' },
  ]

  return (
    <nav
      id="navbar"
      role="navigation"
      aria-label="Main navigation"
      className={`fixed top-0 left-0 right-0 z-50 transition-colors duration-150 border-b ${
        scrolled || isArchitecture
          ? 'border-border bg-background/90 backdrop-blur-md'
          : 'border-transparent bg-transparent'
      }`}
    >
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">

        {/* Logo */}
        <a
          href="/"
          onClick={(e) => handleClick(e, '/')}
          className="flex items-center gap-2 transition-opacity hover:opacity-80"
        >
          <span
            className="font-sans text-sm font-semibold tracking-tight text-foreground"
          >
            KnowVex
          </span>
        </a>

        {/* Center nav links — desktop */}
        <div className="hidden items-center gap-8 md:flex">
          {navLinks.map((link) => {
            const isActive = link.href === '/architecture' ? isArchitecture : false
            return (
              <a
                key={link.label}
                href={link.href}
                onClick={(e) => handleClick(e, link.href)}
                className={`text-[13px] transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary ${
                  isActive
                    ? 'font-semibold text-primary'
                    : 'font-medium text-muted-foreground hover:text-foreground'
                }`}
              >
                {link.label}
              </a>
            )
          })}
          <a
            href="https://github.com/ayush3739/RAG-QA"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-[13px] font-medium text-muted-foreground transition-colors duration-150 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
          >
            <Github className="h-3.5 w-3.5" aria-hidden="true" />
            GitHub
          </a>
        </div>

        {/* Right Actions: Theme Toggle & Launch CTA */}
        <div className="hidden md:flex items-center gap-3">
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="flex h-8 w-8 items-center justify-center rounded-full border border-border bg-surface text-muted-foreground transition-colors duration-150 hover:bg-surface-container hover:text-foreground active:scale-[0.98] disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary motion-safe:hover:scale-105 motion-reduce:transform-none"
          >
            {theme === 'dark' ? (
              <Sun className="h-4 w-4 text-warning" aria-hidden="true" />
            ) : (
              <Moon className="h-4 w-4 text-primary" aria-hidden="true" />
            )}
          </button>

          <ShinyButton
            size="sm"
            animated={false}
            onClick={() => { navigate('/chat/new') }}
          >
            <span>Launch app</span>
            <ArrowRight className="h-3 w-3 transition-transform duration-150 group-hover:translate-x-0.5" aria-hidden="true" />
          </ShinyButton>
        </div>

        {/* Mobile Actions: Theme Toggle & Launch */}
        <div className="flex md:hidden items-center gap-2">
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-border bg-surface text-muted-foreground transition-colors duration-150 hover:bg-surface-container hover:text-foreground active:scale-[0.98] disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            {theme === 'dark' ? (
              <Sun className="h-3.5 w-3.5 text-warning" aria-hidden="true" />
            ) : (
              <Moon className="h-3.5 w-3.5 text-primary" aria-hidden="true" />
            )}
          </button>

          <ShinyButton
            size="sm"
            animated={false}
            onClick={() => { navigate('/chat/new') }}
            className="!px-3 !py-1"
          >
            <span>Launch</span>
          </ShinyButton>
        </div>
      </div>
    </nav>
  )
}
