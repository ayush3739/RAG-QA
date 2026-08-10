import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, Plus } from 'lucide-react'

const ease = [0.23, 1, 0.32, 1] as const

const FAQS = [
  {
    q: 'What makes DocuMind different from a plain RAG chatbot?',
    a: 'Most RAG systems blindly retrieve from documents regardless of the question. DocuMind uses an agentic router that first classifies your query — then picks the right tool: document retrieval via hybrid BM25 + vector search, live web search, deep multi-step research, or pure LLM reasoning. Every answer includes grounded citations so you always know where the information came from.',
  },
  {
    q: 'What document types are supported?',
    a: 'PDF, DOCX, TXT, and Markdown files are supported out of the box. Documents are chunked using a parent-child hierarchical strategy — small chunks are indexed for precision retrieval while full parent paragraphs are sent to the LLM for context.',
  },
  {
    q: 'How does the hybrid search work?',
    a: 'Each query runs in parallel through BM25 (exact keyword matching) and a semantic vector index stored in PostgreSQL with pgvector. The results are merged using Reciprocal Rank Fusion (RRF) — giving you the best of both lexical and semantic retrieval without needing a separate search service.',
  },
  {
    q: 'Is my data stored securely?',
    a: 'Yes. All document embeddings and metadata live in your own PostgreSQL instance. Auth uses JWT access tokens stored only in memory (never localStorage) and refresh tokens in HttpOnly, SameSite=Lax cookies — preventing XSS-based token theft. OAuth via GitHub and Google is fully supported.',
  },
  {
    q: 'Can I self-host DocuMind?',
    a: 'Absolutely — DocuMind is fully open source under the MIT license. You need Python 3.11+, Node 18+, and a PostgreSQL 15+ database with the pgvector extension. Full setup instructions are in the README on GitHub.',
  },
  {
    q: 'Which LLM providers are supported?',
    a: 'The backend abstracts providers through a unified interface. Currently tested with OpenAI (gpt-4o, gpt-4o-mini) and Anthropic (claude-3.5-sonnet). Switching providers is a single environment variable change — no code needed.',
  },
]

function FAQItem({ q, a, idx }: { q: string; a: string; idx: number }) {
  const [open, setOpen] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, ease, delay: idx * 0.06 }}
      className="border-b border-white/[0.06]"
    >
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-start justify-between gap-4 py-6 text-left"
      >
        <span className="text-[17px] tracking-[0.01em] font-normal text-zinc-100">{q}</span>
        <span
          className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center text-zinc-500 transition-transform duration-300 ${open ? 'rotate-45' : ''}`}
        >
          <Plus size={14} />
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.32, ease }}
            className="overflow-hidden"
          >
            <p className="pb-6 pr-12 text-[15px] leading-relaxed text-zinc-400">{a}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function FAQSection() {
  return (
    <section id="faq" className="border-t border-white/[0.05] bg-[#09090b] py-28 md:py-40">
      <div className="mx-auto max-w-7xl px-6 md:px-12">
        <div className="grid gap-20 md:grid-cols-[1fr_1.5fr] lg:gap-32">
          {/* Left Column: Header and CTA */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, ease }}
              className="flex flex-col items-start text-left"
            >
              <p className="mb-6 text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500">
                FAQ
              </p>
              <h2
                className="text-4xl font-medium tracking-[-0.03em] text-white md:text-[52px] md:leading-[1.1]"
                style={{ fontFamily: "'DM Sans', sans-serif" }}
              >
                Frequently <br className="hidden lg:block" /> asked questions
              </h2>
              <p className="mt-6 text-[15px] text-zinc-400">
                Everything you need to know about DocuMind.
              </p>
            </motion.div>

            {/* CTA */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, ease, delay: 0.2 }}
              className="mt-12 flex flex-col items-start gap-5"
            >
              <a
                href="https://github.com/ayush3739/RAG-QA/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="bg-[#111118] border border-white/[0.04] px-8 py-3.5 text-[11px] font-bold uppercase tracking-[0.15em] text-zinc-300 transition-all hover:bg-[#1a1a24] hover:text-white"
              >
                Open an issue
              </a>
              <a
                href="https://github.com/ayush3739/RAG-QA"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.15em] text-zinc-500 transition-colors hover:text-zinc-300"
              >
                <span className="border-b border-zinc-700/50 pb-0.5">Check out the GitHub</span>
                <ArrowRight className="h-3 w-3" />
              </a>
            </motion.div>
          </div>

          {/* Right Column: FAQ list */}
          <div>
            <p className="mb-6 text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500">
              GENERAL
            </p>
            <div className="border-t border-white/[0.06]">
              {FAQS.map((item, i) => (
                <FAQItem key={i} q={item.q} a={item.a} idx={i} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
