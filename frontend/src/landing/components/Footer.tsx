import { Github, ArrowRight, Twitter } from 'lucide-react'
import PixelLogo from './PixelLogo'

const footerLinks = [
  {
    title: 'Product',
    links: [
      { name: 'Features', href: '#' },
      { name: 'Security', href: '#' },
      { name: 'API Reference', href: '#' },
      { name: 'Pricing', href: '#' },
    ],
  },
  {
    title: 'Resources',
    links: [
      { name: 'Documentation', href: '#' },
      { name: 'Blog', href: '#' },
      { name: 'Changelog', href: '#' },
      { name: 'Community', href: '#' },
    ],
  },
  {
    title: 'Company',
    links: [
      { name: 'About', href: '#' },
      { name: 'Contact', href: '#' },
      { name: 'Privacy Policy', href: '#' },
      { name: 'Terms of Service', href: '#' },
    ],
  },
]

export default function Footer() {
  return (
    <footer className="relative border-t border-white/[0.05] bg-[#050508] pt-24 overflow-hidden">
      {/* Ambient background glows for footer */}
      <div className="pointer-events-none absolute top-0 left-1/4 h-[400px] w-[400px] rounded-full bg-blue-500/5 blur-[120px]" />
      <div className="pointer-events-none absolute bottom-0 right-1/4 h-[500px] w-[500px] rounded-full bg-purple-500/5 blur-[150px]" />
      <div className="mx-auto max-w-7xl px-6 md:px-12">
        {/* Top Section: Links and CTA */}
        <div className="flex flex-col md:flex-row justify-between gap-16 md:gap-8 pb-20">
          
          {/* Brand & Description */}
          <div className="flex flex-col items-start gap-6 max-w-xs">
            <PixelLogo text="DOCUMIND" className="w-48" />
            <p className="text-[14px] text-zinc-500 leading-relaxed">
              Upload a document, ask a question. Agentic tool-routing decides the rest. No setup. No API keys.
            </p>
            <div className="group relative rounded-full p-[1px] shadow-[0_0_20px_rgba(99,102,241,0.2)] transition-all duration-300 hover:shadow-[0_0_30px_rgba(99,102,241,0.35)] hover:scale-[1.02] active:scale-[0.98] mt-4">
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400/80 via-purple-400/80 to-blue-400/80 opacity-80 transition-opacity duration-300 group-hover:opacity-100" />
              <button
                onClick={() => { window.location.hash = '#/chat/new' }}
                className="relative flex items-center gap-2 rounded-full bg-[#050508]/90 px-6 py-2.5 text-sm font-medium backdrop-blur-md transition-colors hover:bg-[#050508]/70"
              >
                <span className="bg-gradient-to-r from-blue-100 to-purple-200 bg-clip-text text-transparent">
                  Launch workspace
                </span>
                <ArrowRight className="h-4 w-4 text-purple-200 transition-transform duration-200 group-hover:translate-x-0.5" />
              </button>
            </div>
          </div>

          {/* Links Columns */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-12 md:gap-16 relative z-10">
            {/* Product */}
            <div className="flex flex-col gap-6">
              <h3 className="text-[13px] font-bold uppercase tracking-wider text-white">Product</h3>
              <ul className="flex flex-col gap-4">
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Features</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Security</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">API Reference</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Pricing</a></li>
              </ul>
            </div>

            {/* Socials */}
            <div className="flex flex-col gap-6">
              <h3 className="text-[13px] font-bold uppercase tracking-wider text-white">Socials</h3>
              <ul className="flex flex-col gap-4">
                <li>
                  <a href="https://twitter.com/ayush3739" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-[14px] text-zinc-400 transition-colors hover:text-white">
                    <Twitter className="h-4 w-4" /> Twitter
                  </a>
                </li>
                <li>
                  <a href="https://github.com/ayush3739/RAG-QA" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-[14px] text-zinc-400 transition-colors hover:text-white">
                    <Github className="h-4 w-4" /> GitHub
                  </a>
                </li>
              </ul>
            </div>

            {/* Resources */}
            <div className="flex flex-col gap-6">
              <h3 className="text-[13px] font-bold uppercase tracking-wider text-white">Resources</h3>
              <ul className="flex flex-col gap-4">
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Documentation</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Blog</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Changelog</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Community</a></li>
              </ul>
            </div>

            {/* Company */}
            <div className="flex flex-col gap-6">
              <h3 className="text-[13px] font-bold uppercase tracking-wider text-white">Company</h3>
              <ul className="flex flex-col gap-4">
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">About</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Contact</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Privacy Policy</a></li>
                <li><a href="#" className="text-[14px] text-zinc-400 transition-colors hover:text-white">Terms of Service</a></li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="relative z-10 border-t border-white/[0.04] bg-[#030305]/50 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-center px-6 py-6">
          <p className="font-mono text-[12px] text-zinc-600">
            DocuMind — Ayush Maurya · {new Date().getFullYear()}
          </p>
        </div>
      </div>
    </footer>
  )
}
