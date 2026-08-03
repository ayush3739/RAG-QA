import { Github, ArrowRight } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-white/[0.05] bg-[#09090b]">
      {/* CTA band */}
      <div className="mx-auto max-w-6xl px-6 py-20">
        <div className="flex flex-col items-center gap-6 text-center">
          <h2
            className="text-3xl font-medium tracking-[-0.03em] text-white md:text-5xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Try it now — it&apos;s free.
          </h2>
          <p className="max-w-sm text-[15px] text-zinc-500">
            Upload a document, ask a question. No setup. No API keys.
          </p>
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="group flex items-center gap-2 rounded-full bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white transition-all duration-150 hover:bg-indigo-500 hover:shadow-lg hover:shadow-indigo-500/20"
          >
            Launch workspace
            <ArrowRight className="h-4 w-4 transition-transform duration-150 group-hover:translate-x-0.5" />
          </button>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="border-t border-white/[0.04]">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <p className="font-mono text-[12px] text-zinc-600">
            DocuMind — Ayush Maurya · {new Date().getFullYear()}
          </p>
          <a
            href="https://github.com/ayush3739/RAG-QA"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-[12px] text-zinc-600 transition-colors hover:text-white"
          >
            <Github className="h-3.5 w-3.5" />
            GitHub
          </a>
        </div>
      </div>
    </footer>
  )
}
