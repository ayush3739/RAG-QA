import { navigate } from '../lib/navigation';
import React, { useState } from "react";
import { motion } from "framer-motion";
import { ServerCrash, RefreshCw, Home } from "lucide-react";
import { api } from "../lib/api";
import { useStore } from "../store/useStore";
import { cn } from "../lib/utils";

export default function ServerErrorView() {
  const [isRetrying, setIsRetrying] = useState(false);
  const setCurrentTab = useStore((s) => s.setCurrentTab);

  const handleRetry = async () => {
    setIsRetrying(true);
    try {
      await api.getHealth();
      setCurrentTab("dashboard");
      navigate('/dashboard');
    } catch {
      // Still offline
    } finally {
      setIsRetrying(false);
    }
  };

  const handleReturnHome = () => {
    window.history.pushState(null, '', '/');
    window.dispatchEvent(new PopStateEvent('popstate'));
    window.dispatchEvent(new HashChangeEvent('hashchange'));
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
          <div className="w-16 h-16 rounded-2xl bg-danger-subtle border border-danger-subtle flex items-center justify-center text-danger mb-3 shadow-xs">
            <ServerCrash className="w-8 h-8 text-danger" />
          </div>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-danger-subtle border border-danger-subtle text-xs font-semibold text-danger">
            <span className="w-1.5 h-1.5 rounded-full bg-danger inline-block" />
            <span className="font-mono tabular-nums">500</span>
            <span>Core Node Offline</span>
          </span>
        </div>

        {/* Heading & Prose */}
        <h1 className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2">
          Internal server error
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed mb-7 max-w-sm">
          The KnowVex core retrieval node encountered an unexpected exception or is temporarily unreachable. Please retry the connection.
        </p>

        {/* Action Panel */}
        <div className="w-full space-y-2.5">
          <button
            type="button"
            onClick={handleRetry}
            disabled={isRetrying}
            className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-xs sm:text-sm flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 transition-opacity cursor-pointer shadow-xs motion-reduce:transform-none"
          >
            <RefreshCw className={cn("w-4 h-4", isRetrying && "animate-spin motion-reduce:animate-none")} />
            <span>{isRetrying ? "Attempting reconnection…" : "Retry connection"}</span>
          </button>

          <button
            type="button"
            onClick={handleReturnHome}
            className="w-full py-2.5 px-4 rounded-xl bg-surface border border-border text-foreground font-medium text-xs sm:text-sm flex items-center justify-center gap-2 hover:bg-surface-container active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 transition-colors cursor-pointer motion-reduce:transform-none"
          >
            <Home className="w-4 h-4 text-muted-foreground" />
            <span>Return to Home</span>
          </button>
        </div>

        {/* Diagnostic Watermark */}
        <div className="mt-10 text-xs text-muted-foreground flex items-center gap-2">
          <span>KnowVex Telemetry Channel</span>
          <span className="w-1 h-1 rounded-full bg-border" />
          <span>Node knowvex-core-v2.0</span>
        </div>
      </motion.div>
    </div>
  );
}
