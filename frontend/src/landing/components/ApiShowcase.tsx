import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ApiTab {
  id: string
  name: string
  endpoint: string
  method: string
  request: string
  response: JSX.Element
}

const tabs: ApiTab[] = [
  {
    id: 'chat',
    name: 'Chat API',
    endpoint: '/api/v1/chat',
    method: 'POST',
    request: `{\n  "message": "Verify the Q3 margin details",\n  "collection_id": "finance_docs_2024"\n}`,
    response: (
      <>
        <span className="text-emerald-400">&quot;event&quot;</span>: <span className="text-amber-400">&quot;text_stream&quot;</span>,
        {'\n  '}
        <span className="text-indigo-400">&quot;chunk&quot;</span>: <span className="text-emerald-400">&quot;Q3 operating margin expanded to 28.4%...&quot;</span>,
        {'\n  '}
        <span className="text-indigo-400">&quot;confidence&quot;</span>: <span className="text-amber-400">0.89</span>,
        {'\n  '}
        <span className="text-indigo-400">&quot;citations&quot;</span>: [<span className="text-emerald-400">&quot;report_page_12.pdf&quot;</span>]
      </>
    )
  },
  {
    id: 'research',
    name: 'Research API',
    endpoint: '/api/v1/research',
    method: 'POST',
    request: `{\n  "topic": "Summarize Vercel hosting latency benchmarks",\n  "include_web": true\n}`,
    response: (
      <>
        <span className="text-indigo-400">&quot;summary&quot;</span>: <span className="text-emerald-400">&quot;Vercel edge networks averaged 15ms global TTFB...&quot;</span>,
        {'\n  '}
        <span className="text-indigo-400">&quot;sources&quot;</span>: [
        {'\n    '}{'{'} <span className="text-indigo-400">&quot;type&quot;</span>: <span className="text-emerald-400">&quot;web&quot;</span>, <span className="text-indigo-400">&quot;url&quot;</span>: <span className="text-emerald-400">&quot;https://vercel.com/blog/...&quot;</span> {'}'}
        {'\n  '}],
        {'\n  '}
        <span className="text-indigo-400">&quot;tool_trace&quot;</span>: [<span className="text-emerald-400">&quot;web_search&quot;</span>, <span className="text-emerald-400">&quot;evidence_synthesis&quot;</span>]
      </>
    )
  },
  {
    id: 'upload',
    name: 'Upload API',
    endpoint: '/api/v1/collections/upload',
    method: 'POST',
    request: `// multipart/form-data\nfile: path/to/quarterly_audit.pdf\ncollection_id: audit_log`,
    response: (
      <>
        <span className="text-indigo-400">&quot;status&quot;</span>: <span className="text-emerald-400">&quot;processing&quot;</span>,
        {'\n  '}
        <span className="text-indigo-400">&quot;job_id&quot;</span>: <span className="text-emerald-400">&quot;job_984cf82&quot;</span>,
        {'\n  '}
        <span className="text-indigo-400">&quot;chunks_expected&quot;</span>: <span className="text-amber-400">420</span>,
        {'\n  '}
        <span className="text-indigo-400">&quot;parent_child_mapping&quot;</span>: <span className="text-emerald-400">&quot;enabled&quot;</span>
      </>
    )
  }
]

export default function ApiShowcase() {
  const [activeTab, setActiveTab] = useState('chat')
  const currentTabData = tabs.find(t => t.id === activeTab) || tabs[0]

  return (
    <section id="api" className="border-t border-white/[0.05] bg-[#09090b] py-28">
      <div className="mx-auto max-w-5xl px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-zinc-600">
            Case Study · 06
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-4xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            REST API Showcase
          </h2>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-zinc-500">
            Every feature is exposed via a documented REST API. Built with FastAPI for high-concurrency streaming.
          </p>
        </motion.div>

        {/* Showcase Tool */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="overflow-hidden rounded-2xl border border-white/[0.05] bg-white/[0.02]"
        >
          {/* Tab Navigation */}
          <div className="flex flex-wrap border-b border-white/[0.05] bg-[#060608]">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`relative px-6 py-4 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-white'
                    : 'text-zinc-500 hover:bg-white/[0.02] hover:text-zinc-300'
                }`}
              >
                {tab.name}
                {activeTab === tab.id && (
                  <motion.div
                    layoutId="api-tab-indicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-500"
                    transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                  />
                )}
              </button>
            ))}
          </div>

          {/* Content Area */}
          <AnimatePresence mode="wait">
            <motion.div
              key={currentTabData.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-1 divide-y divide-white/[0.05] md:grid-cols-2 md:divide-x md:divide-y-0"
            >
              {/* Request Pane */}
              <div className="p-6 md:p-8">
                <div className="mb-4 flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">Request</span>
                  <span className="rounded-full bg-indigo-500/10 px-2 py-0.5 font-mono text-[10px] uppercase text-indigo-400">
                    {currentTabData.method}
                  </span>
                </div>
                <div className="font-mono text-[13px] text-zinc-300">
                  <span className="text-indigo-400">{currentTabData.endpoint}</span>
                </div>
                <pre className="mt-6 overflow-x-auto font-mono text-[11px] leading-relaxed text-zinc-400">
                  {currentTabData.request}
                </pre>
              </div>

              {/* Response Pane */}
              <div className="bg-[#060608] p-6 md:p-8">
                <div className="mb-4 flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">Response</span>
                  <div className="flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] uppercase text-emerald-400">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    200 OK
                  </div>
                </div>
                <pre className="overflow-x-auto font-mono text-[11px] leading-relaxed text-zinc-400">
                  {'{'}{'\n  '}
                  {currentTabData.response}
                  {'\n}'}
                </pre>
              </div>
            </motion.div>
          </AnimatePresence>
        </motion.div>
      </div>
    </section>
  )
}
