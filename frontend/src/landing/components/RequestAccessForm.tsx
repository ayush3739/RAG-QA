import { useState, type FormEvent } from 'react';
import { toast } from 'sonner';
import { motion } from 'framer-motion';

export default function RequestAccessForm() {
  const handleScroll = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault()
    const target = document.querySelector(href)
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section className="relative py-24 md:py-32 overflow-hidden" style={{ backgroundColor: '#050505' }}>
      {/* Gradient orb */}
      <div
        className="linear-orb left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-[#7C5CFF] opacity-[0.06]"
        aria-hidden="true"
      />

      <div className="relative mx-auto max-w-2xl px-6 text-center">
        {/* Headline */}
        <motion.h2
          className="text-4xl font-light tracking-tight md:text-6xl linear-gradient-text"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          Build reliable document intelligence.
        </motion.h2>

        {/* Sub */}
        <motion.p
          className="mx-auto mt-4 max-w-md text-base text-[#707070]"
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          Try the sandbox app workspace, inspect the repository code, or review the system design diagrams.
        </motion.p>

        {/* Action Buttons */}
        <motion.div
          className="mx-auto mt-8 flex flex-col sm:flex-row justify-center items-center gap-4 max-w-md w-full"
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <button
            onClick={() => { window.location.hash = '#/chat/new' }}
            className="rounded-full bg-white px-6 py-2.5 text-sm font-medium text-black transition hover:bg-white/90 w-full cursor-pointer"
          >
            Live Demo
          </button>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-white/10 bg-white/5 px-6 py-2.5 text-sm font-medium text-white transition hover:bg-white/10 flex items-center justify-center w-full"
          >
            GitHub Code
          </a>
          <a
            href="#architecture"
            onClick={(e) => handleScroll(e, '#architecture')}
            className="rounded-full border border-white/10 bg-white/5 px-6 py-2.5 text-sm font-medium text-white transition hover:bg-white/10 flex items-center justify-center w-full"
          >
            Architecture
          </a>
        </motion.div>

        {/* Learning project note */}
        <motion.p
          className="mt-6 font-mono text-[11px] text-[#444]"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          Built with Python, LangGraph, pgvector, and React.
        </motion.p>
      </div>
    </section>
  );
}
