import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, Database, Cpu, Search, CheckCircle } from 'lucide-react';

const steps = [
  { id: 1, text: 'Parsing query...', delay: 800, icon: Terminal, color: 'text-zinc-400' },
  { id: 2, text: 'Routing to Parent-Child Vector Store', delay: 1500, icon: Cpu, color: 'text-indigo-400' },
  { id: 3, text: 'Executing Reciprocal Rank Fusion (BM25 + Vector)', delay: 2200, icon: Search, color: 'text-indigo-400' },
  { id: 4, text: 'Querying pgvector via single-pass function call', delay: 3000, icon: Database, color: 'text-emerald-400' },
  { id: 5, text: 'Aggregating context (Latency: 124ms)', delay: 3500, icon: CheckCircle, color: 'text-emerald-500' },
];

export default function TerminalDemo() {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    // Sequence the steps based on their delays
    const timeouts = steps.map((step, index) => {
      return setTimeout(() => {
        setCurrentStep(index + 1);
      }, step.delay);
    });

    // Reset loop
    const resetTimeout = setTimeout(() => {
      setCurrentStep(0);
    }, 6000);

    return () => {
      timeouts.forEach(clearTimeout);
      clearTimeout(resetTimeout);
    };
  }, [currentStep === 0]); // Re-run when it resets to 0

  return (
    <section className="py-24 bg-[#09090b] relative overflow-hidden">
      {/* Background ambient glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-indigo-500/10 blur-[120px] rounded-full pointer-events-none" />
      
      <div className="max-w-4xl mx-auto px-6 relative z-10">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            Real-time <span className="text-indigo-400">Execution</span>
          </h2>
          <p className="text-zinc-400 max-w-xl mx-auto">
            Watch how DocuMind routes your query through its optimized retrieval pipeline under the hood.
          </p>
        </div>

        {/* Terminal Window */}
        <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0e] shadow-2xl overflow-hidden">
          {/* Mac Header */}
          <div className="flex items-center px-4 py-3 border-b border-white/[0.05] bg-[#050508]">
            <div className="flex space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500/80" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <div className="w-3 h-3 rounded-full bg-green-500/80" />
            </div>
            <div className="mx-auto flex items-center text-xs text-zinc-500 font-mono">
              <Terminal className="w-3.5 h-3.5 mr-2" />
              documind-agent ~ bash
            </div>
          </div>

          {/* Terminal Body */}
          <div className="p-6 font-mono text-sm min-h-[320px] flex flex-col">
            <div className="flex items-center text-zinc-300 mb-6">
              <span className="text-emerald-400 mr-2">➜</span>
              <span className="text-indigo-400 mr-2">~</span>
              <span className="typing-animation overflow-hidden whitespace-nowrap border-r-2 border-indigo-500 pr-1 inline-block">
                documind query "What is the system architecture?"
              </span>
            </div>

            <div className="space-y-4 flex-1">
              <AnimatePresence>
                {steps.slice(0, currentStep).map((step) => {
                  const Icon = step.icon;
                  return (
                    <motion.div
                      key={step.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-center space-x-3"
                    >
                      <Icon className={`w-4 h-4 ${step.color}`} />
                      <span className="text-zinc-400">{step.text}</span>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              
              {currentStep > 0 && currentStep < steps.length && (
                <motion.div 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }} 
                  className="flex items-center space-x-3 text-zinc-600 mt-4"
                >
                  <span className="animate-pulse">_</span>
                </motion.div>
              )}
            </div>

            {currentStep === steps.length && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="mt-8 p-4 rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-200"
              >
                <div className="text-xs uppercase tracking-widest text-indigo-400 mb-2">Final Output [JSON]</div>
                <pre className="text-xs overflow-x-auto">
{`{
  "answer": "DocuMind uses a hybrid retrieval architecture...",
  "sources": [
    {"doc": "architecture.pdf", "page": 4, "score": 0.92}
  ],
  "latency_ms": 124
}`}
                </pre>
              </motion.div>
            )}
          </div>
        </div>
      </div>
      
      {/* CSS for typing animation */}
      <style>{`
        .typing-animation {
          animation: typing 1.5s steps(40, end), blink-caret .75s step-end infinite;
        }
        @keyframes typing {
          from { width: 0 }
          to { width: 100% }
        }
        @keyframes blink-caret {
          from, to { border-color: transparent }
          50% { border-color: #6366f1; }
        }
      `}</style>
    </section>
  );
}
