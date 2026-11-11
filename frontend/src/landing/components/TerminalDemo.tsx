import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, Database, Cpu, Search, CheckCircle } from 'lucide-react';

const steps = [
  { id: 1, text: 'Parsing query...', delay: 800, icon: Terminal, color: 'text-zinc-400' },
  { id: 2, text: 'Routing to Parent-Child Vector Store', delay: 1500, icon: Cpu, color: 'text-indigo-400' },
  { id: 3, text: 'Executing Reciprocal Rank Fusion (BM25 + Vector)', delay: 2200, icon: Search, color: 'text-indigo-400' },
  { id: 4, text: 'Querying pgvector via single-pass function call', delay: 3000, icon: Database, color: 'text-emerald-400' },
  { id: 5, text: 'Aggregating context (Latency: 124ms)', delay: 3500, icon: CheckCircle, color: 'text-emerald-400' },
];

export default function TerminalDemo() {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    let timeouts: ReturnType<typeof setTimeout>[] = [];
    let isMounted = true;

    const startCycle = () => {
      if (!isMounted) return;
      setCurrentStep(0);
      timeouts = steps.map((step, index) =>
        setTimeout(() => {
          if (isMounted) {
            setCurrentStep(index + 1);
          }
        }, step.delay)
      );

      const loopTimeout = setTimeout(() => {
        if (isMounted) {
          startCycle();
        }
      }, 6500);
      timeouts.push(loopTimeout);
    };

    startCycle();

    return () => {
      isMounted = false;
      timeouts.forEach(clearTimeout);
    };
  }, []);

  return (
    <section className="py-24 bg-background relative overflow-hidden">
      <div className="max-w-4xl mx-auto px-6 relative z-10">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-foreground mb-4" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            Real-time <span className="text-primary">Execution</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Watch how KnowVex routes your query through its optimized retrieval pipeline under the hood.
          </p>
        </div>

        {/* Terminal Window - Dark in all modes for devtool contrast */}
        <div className="rounded-xl border border-zinc-800 bg-[#09090b] shadow-[0_20px_50px_rgba(0,0,0,0.35)] overflow-hidden">
          {/* Mac Header */}
          <div className="flex items-center px-4 py-3 border-b border-zinc-800/80 bg-[#121216]">
            <div className="flex space-x-2">
              <div className="w-3 h-3 rounded-full bg-rose-500/80" />
              <div className="w-3 h-3 rounded-full bg-amber-500/80" />
              <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
            </div>
            <div className="mx-auto flex items-center text-xs text-zinc-400 font-mono">
              <Terminal className="w-3.5 h-3.5 mr-2 text-zinc-400" aria-hidden="true" />
              KnowVex-agent terminal
            </div>
          </div>

          {/* Terminal Body */}
          <div className="p-6 text-sm min-h-[320px] flex flex-col bg-[#09090b]">
            <div className="flex items-center text-zinc-100 mb-6">
              <span className="text-emerald-400 mr-2 font-mono text-xs font-semibold">➜</span>
              <span className="text-cyan-400 mr-2 font-mono text-xs font-semibold">~</span>
              <code className="typing-animation overflow-hidden whitespace-nowrap border-r-2 border-indigo-400 pr-1 inline-block font-mono text-xs text-zinc-200">
                KnowVex query --topic system-architecture
              </code>
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
                      <Icon className={`w-4 h-4 shrink-0 ${step.color}`} aria-hidden="true" />
                      <span className="text-zinc-300 text-xs font-mono">{step.text}</span>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              
              {currentStep > 0 && currentStep < steps.length && (
                <motion.div 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }} 
                  className="flex items-center space-x-3 text-zinc-500 mt-4"
                >
                  <span className="motion-safe:animate-pulse motion-reduce:animate-none font-mono text-zinc-400">_</span>
                </motion.div>
              )}
            </div>

            {currentStep === steps.length && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="mt-8 p-4 rounded-lg bg-[#141419] border border-zinc-800 text-zinc-200 font-mono"
              >
                <div className="text-xs font-semibold text-indigo-400 mb-2 font-mono">Final Output (JSON)</div>
                <pre className="text-xs font-mono overflow-x-auto text-zinc-300 leading-relaxed">
{`{
  "answer": "KnowVex uses a hybrid retrieval architecture...",
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
          50% { border-color: #818cf8; }
        }
      `}</style>
    </section>
  );
}
