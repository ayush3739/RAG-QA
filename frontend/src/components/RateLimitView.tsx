import { navigate } from '../lib/navigation';
import React from "react";
import { motion } from "framer-motion";
import { Gauge, Home, RotateCcw, AlertCircle } from "lucide-react";
import { useStore } from "../store/useStore";

export default function RateLimitView() {
  const setCurrentTab = useStore((s) => s.setCurrentTab);

  const handleReturn = () => {
    setCurrentTab("conversations");
    navigate('/chat/new');
  };

  const handleDashboard = () => {
    setCurrentTab("dashboard");
    navigate('/dashboard');
  };

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 bg-background text-foreground select-none">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="relative z-10 max-w-md w-full text-center flex flex-col items-center"
      >
        {/* Diagnostic Telemetry Badge */}
        <div className="mb-5 flex flex-col items-center">
          <div className="w-16 h-16 rounded-2xl bg-warning-subtle border border-warning-subtle flex items-center justify-center text-warning mb-3 shadow-xs">
            <Gauge className="w-8 h-8 text-warning" />
          </div>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-warning-subtle border border-warning-subtle text-xs font-semibold text-warning">
            <span className="w-1.5 h-1.5 rounded-full bg-warning inline-block" />
            <span className="font-mono tabular-nums">429</span>
            <span>Capacity Limit Reached</span>
          </span>
        </div>

        {/* Heading & Prose */}
        <h1 className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2">
          Rate limit reached
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed mb-6 max-w-sm">
          You have reached the temporary throughput capacity or query frequency allocation. The token bucket refills automatically.
        </p>

        {/* Optimization Guidance Box */}
        <div className="w-full text-left p-4 rounded-xl bg-surface-container-lowest border border-border space-y-2 mb-6 text-xs text-muted-foreground">
          <div className="flex items-center text-foreground font-semibold gap-1.5">
            <AlertCircle className="w-4 h-4 text-warning" />
            <span>Optimization guidance</span>
          </div>
          <ul className="list-disc list-inside space-y-1.5 pl-0.5">
            <li>Wait approximately 30 to 60 seconds for token bucket replenishment.</li>
            <li>Unlink inactive documents to conserve prompt context tokens.</li>
            <li>Deconstruct extensive multi-part queries into focused research questions.</li>
          </ul>
        </div>

        {/* Actions Panel */}
        <div className="w-full space-y-2.5">
          <button
            type="button"
            onClick={handleReturn}
            className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-xs sm:text-sm flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 whitespace-nowrap transition-opacity cursor-pointer shadow-xs motion-reduce:transform-none"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Return to research</span>
          </button>

          <button
            type="button"
            onClick={handleDashboard}
            className="w-full py-2.5 px-4 rounded-xl bg-surface border border-border text-foreground font-medium text-xs sm:text-sm flex items-center justify-center gap-2 hover:bg-surface-container active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 transition-colors cursor-pointer motion-reduce:transform-none"
          >
            <Home className="w-4 h-4 text-muted-foreground" />
            <span>Go to Dashboard</span>
          </button>
        </div>

        {/* Diagnostic Watermark */}
        <div className="mt-10 text-xs text-muted-foreground flex items-center gap-2">
          <span>KnowVex Token Rate Governor</span>
          <span className="w-1 h-1 rounded-full bg-border" />
          <span>Throughput Shield</span>
        </div>
      </motion.div>
    </div>
  );
}
