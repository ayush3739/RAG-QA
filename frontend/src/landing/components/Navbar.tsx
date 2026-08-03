import { useState, useEffect } from 'react'
import {
  Sheet,
  SheetTrigger,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '../../components/ui/sheet'
import { Menu } from 'lucide-react'

const NAV_LINKS = [
  { label: 'Product', href: '#hero' },
  { label: 'Features', href: '#features' },
  { label: 'API', href: '#api' },
  { label: 'Changelog', href: '#roadmap' },
] as const

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault()
    setMobileOpen(false)
    const target = document.querySelector(href)
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <nav
      id="navbar"
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled ? 'linear-glass' : 'bg-transparent'
      }`}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
        {/* Logo */}
        <a
          href="#hero"
          onClick={(e) => handleNavClick(e, '#hero')}
          className="font-mono text-base font-medium tracking-tight transition-opacity hover:opacity-80"
        >
          <span className="text-white">Docu</span>
          <span className="linear-gradient-text-accent">Mind</span>
        </a>

        {/* Desktop nav links */}
        <div className="hidden items-center gap-6 md:flex">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              onClick={(e) => handleNavClick(e, link.href)}
              className="text-xs font-medium uppercase tracking-widest text-[#707070] transition-colors duration-200 hover:text-white"
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* Desktop right side */}
        <div className="hidden items-center gap-4 md:flex">
          <a
            href="#api"
            onClick={(e) => handleNavClick(e, '#api')}
            className="text-xs text-[#707070] transition-colors duration-200 hover:text-white"
          >
            API Specs
          </a>
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-medium text-white transition-colors duration-200 hover:bg-white/10"
          >
            Launch Workspace
          </button>
        </div>

        {/* Mobile hamburger */}
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetTrigger asChild>
            <button
              className="flex items-center justify-center rounded-md p-2 text-[#707070] transition-colors hover:text-white md:hidden"
              aria-label="Open menu"
            >
              <Menu className="size-5" />
            </button>
          </SheetTrigger>
          <SheetContent
            side="right"
            className="w-[300px] border-[#1A1A1A] bg-[#050505] p-0"
          >
            <SheetHeader className="border-b border-[#1A1A1A] px-6 py-5">
              <SheetTitle className="font-mono text-base text-white">
                <span className="text-white">Docu</span>
                <span className="linear-gradient-text-accent">Mind</span>
              </SheetTitle>
            </SheetHeader>
            <div className="flex flex-col gap-1 px-4 py-6">
              {NAV_LINKS.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={(e) => handleNavClick(e, link.href)}
                  className="rounded-md px-3 py-2.5 text-sm font-medium text-[#707070] transition-colors hover:bg-white/[0.03] hover:text-white"
                >
                  {link.label}
                </a>
              ))}
            </div>
            <div className="border-t border-[#1A1A1A] px-4 py-5">
              <a
                href="#api"
                onClick={(e) => handleNavClick(e, '#api')}
                className="mb-3 block rounded-md px-3 py-2.5 text-sm text-[#707070] transition-colors hover:text-white"
              >
                API Specs
              </a>
              <button
                onClick={() => { setMobileOpen(false); window.location.hash = '#/chat/new' }}
                className="w-full rounded-full border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-white/10"
              >
                Launch Workspace
              </button>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </nav>
  )
}
