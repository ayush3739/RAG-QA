import { useRef, useState, useEffect } from 'react';
import { motion, useInView, useReducedMotion } from 'framer-motion';

interface ApiTab {
  id: string;
  name: string;
  endpoint: string;
  method: string;
  request: string;
  response: JSX.Element;
}

export default function ApiShowcase() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: '-20%' });
  const prefersReduced = useReducedMotion();
  const [mounted, setMounted] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');

  useEffect(() => { setMounted(true) }, []);

  const tabs: ApiTab[] = [
    {
      id: 'chat',
      name: 'Chat API',
      endpoint: '/api/v1/chat',
      method: 'POST',
      request: `{\n  "message": "Verify the Q3 margin details",\n  "collection_id": "finance_docs_2024"\n}`,
      response: (
        <code>
          <span className="text-[#3FD8C4]">&quot;event&quot;</span>: <span className="text-[#F5B942]">&quot;text_stream&quot;</span>,
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;chunk&quot;</span>: <span className="text-[#3FD8C4]">&quot;Q3 operating margin expanded to 28.4%...&quot;</span>,
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;confidence&quot;</span>: <span className="text-[#F5B942]">0.89</span>,
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;citations&quot;</span>: [<span className="text-[#3FD8C4]">&quot;report_page_12.pdf&quot;</span>]
        </code>
      )
    },
    {
      id: 'research',
      name: 'Research API',
      endpoint: '/api/v1/research',
      method: 'POST',
      request: `{\n  "topic": "Summarize Vercel hosting latency benchmarks",\n  "include_web": true\n}`,
      response: (
        <code>
          <span className="text-[#7C5CFF]">&quot;summary&quot;</span>: <span className="text-[#3FD8C4]">&quot;Vercel edge networks averaged 15ms global TTFB...&quot;</span>,
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;sources&quot;</span>: [
          {'\n    '}{'{'} <span className="text-[#7C5CFF]">&quot;type&quot;</span>: <span className="text-[#3FD8C4]">&quot;web&quot;</span>, <span className="text-[#7C5CFF]">&quot;url&quot;</span>: <span className="text-[#3FD8C4]">&quot;https://vercel.com/blog/...&quot;</span> {'}'}
          {'\n  '}],
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;tool_trace&quot;</span>: [<span className="text-[#3FD8C4]">&quot;web_search&quot;</span>, <span className="text-[#3FD8C4]">&quot;evidence_synthesis&quot;</span>]
        </code>
      )
    },
    {
      id: 'upload',
      name: 'Upload API',
      endpoint: '/api/v1/collections/upload',
      method: 'POST',
      request: `// multipart/form-data\nfile: path/to/quarterly_audit.pdf\ncollection_id: audit_log`,
      response: (
        <code>
          <span className="text-[#7C5CFF]">&quot;status&quot;</span>: <span className="text-[#3FD8C4]">&quot;processing&quot;</span>,
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;job_id&quot;</span>: <span className="text-[#3FD8C4]">&quot;job_984cf82&quot;</span>,
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;chunks_expected&quot;</span>: <span className="text-[#F5B942]">420</span>,
          {'\n  '}
          <span className="text-[#7C5CFF]">&quot;parent_child_mapping&quot;</span>: <span className="text-[#3FD8C4]">&quot;enabled&quot;</span>
        </code>
      )
    }
  ];

  const currentTabData = tabs.find(t => t.id === activeTab) || tabs[0];

  return (
    <section id="api" ref={sectionRef} className="relative py-20 md:py-24" style={{ backgroundColor: '#050505' }}>
      <div className="mx-auto max-w-5xl px-6">
        {/* Divider */}
        <div className="linear-gradient-line mb-12" />

        {/* Header */}
        <motion.div
          initial={prefersReduced ? false : { opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#7C5CFF]/60">
            API Specs
          </span>
          <h2 className="mt-3 text-3xl font-light tracking-tight md:text-5xl linear-gradient-text-subtle">
            REST API Showcase
          </h2>
          <p className="mt-3 max-w-lg text-[#707070] text-base">
            Clean developer interfaces returning structured responses with complete confidence metrics and tool traces.
          </p>
        </motion.div>

        {/* Tabs Control */}
        <div className="flex gap-2 border-b border-white/5 pb-4 mb-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-1.5 rounded-full font-mono text-[11px] uppercase tracking-wider transition-all cursor-pointer ${
                activeTab === tab.id
                  ? 'bg-white text-black font-semibold'
                  : 'bg-white/[0.02] text-[#707070] hover:text-white border border-white/5'
              }`}
            >
              {tab.name}
            </button>
          ))}
        </div>

        {/* Code Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Request Panel */}
          <div className="linear-gradient-border p-0">
            <div className="linear-code p-0 overflow-hidden h-full flex flex-col justify-between">
              <div className="flex items-center justify-between border-b border-[#1A1A1A] px-5 py-3 bg-white/[0.01]">
                <span className="font-mono text-[10px] text-[#444] uppercase tracking-wider">Request payload</span>
                <span className="font-mono text-[10px] text-[#3FD8C4]">{currentTabData.method} {currentTabData.endpoint}</span>
              </div>
              <div className="p-5 overflow-x-auto font-mono text-[12px] leading-relaxed text-[#888] flex-1 min-h-[160px]">
                <pre><code>{currentTabData.request}</code></pre>
              </div>
            </div>
          </div>

          {/* Response Panel */}
          <div className="linear-gradient-border p-0">
            <div className="linear-code p-0 overflow-hidden h-full flex flex-col justify-between">
              <div className="flex items-center justify-between border-b border-[#1A1A1A] px-5 py-3 bg-white/[0.01]">
                <span className="font-mono text-[10px] text-[#444] uppercase tracking-wider">Response JSON</span>
                <span className="font-mono text-[10px] text-[#F5B942]">200 OK</span>
              </div>
              <div className="p-5 overflow-x-auto font-mono text-[12px] leading-relaxed text-[#888] flex-1 min-h-[160px]">
                <pre>
                  <code>
                    <span className="text-[#333]">{'{'}</span>
                    {'\n  '}
                    {currentTabData.response}
                    {'\n'}
                    <span className="text-[#333]">{'}'}</span>
                  </code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
