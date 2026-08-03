export default function Footer() {
  return (
    <footer className="mt-auto border-t border-white/5" style={{ backgroundColor: '#050505' }}>
      <div className="mx-auto max-w-6xl px-6 py-12">
        {/* Top row */}
        <div className="flex items-start justify-between">
          <span className="font-mono text-sm text-[#444]">DocuMind</span>
          <span className="font-mono text-[11px] text-[#333]">&copy; 2024</span>
        </div>

        {/* Bottom row */}
        <div className="mt-8 flex items-center justify-between">
          <nav className="flex items-center gap-1 text-xs" aria-label="Footer navigation">
            <a
              href="#"
              className="text-[#444] transition-colors hover:text-[#707070]"
            >
              GitHub
            </a>
            <span className="text-[#333]">&middot;</span>
            <a
              href="#"
              className="text-[#444] transition-colors hover:text-[#707070]"
            >
              Documentation
            </a>
            <span className="text-[#333]">&middot;</span>
            <a
              href="#"
              className="text-[#444] transition-colors hover:text-[#707070]"
            >
              Changelog
            </a>
          </nav>
          <span className="font-mono text-[11px] text-[#333]">
            Built with FastAPI, LangChain, and Qdrant.
          </span>
        </div>
      </div>
    </footer>
  );
}
