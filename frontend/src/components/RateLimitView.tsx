import React from "react";
import { motion } from "framer-motion";
import { Gauge, Home, RotateCcw, AlertOctagon } from "lucide-react";
import { useStore } from "../store/useStore";

export default function RateLimitView() {
  const setCurrentTab = useStore((s) => s.setCurrentTab);

  const handleReturn = () => {
    setCurrentTab("conversations");
    window.location.hash = "#/chat/new";
  };

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 bg-background text-foreground overflow-hidden select-none">
      {/* Ambient glowing orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="ambient-orb orb-1 bg-amber-500/10 w-[500px] h-[500px] -top-[100px] left-1/2 -translate-x-1/2 blur-3xl rounded-full" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative z-10 max-w-md w-full text-center flex flex-col items-center"
      >
        {/* Icon Badge */}
        <div className="relative mb-6">
          <div className="w-20 h-20 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-500 shadow-premium">
            <Gauge className="w-10 h-10 animate-pulse" />
          </div>
          <span className="absolute -bottom-2 -right-2 px-2 py-0.5 text-[11px] font-mono font-bold bg-amber-500 text-white rounded-full shadow-sm">
            429
          </span>
        </div>

        {/* Heading */}
        <h1 className="text-2xl font-extrabold text-foreground tracking-tight mb-2">
          Rate Limit Reached
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed mb-6 max-w-sm">
          You've reached the current model throughput capacity or query frequency limit. Please pause for a moment before retrying.
        </p>

        {/* Tips Box */}
        <div className="w-full text-left p-4 rounded-xl bg-surface-container border border-border space-y-2 mb-6 text-xs text-muted-foreground">
          <div className="flex items-center text-foreground font-semibold space-x-1.5">
            <AlertOctagon className="w-3.5 h-3.5 text-amber-500" />
            <span>Optimization Guidance:</span>
          </div>
          <ul className="list-disc list-inside space-y-1 pl-1">
            <li>Wait approximately <strong>30–60 seconds</strong> for the token bucket to refill.</li>
            <li>Unlink inactive documents to reduce prompt context tokens.</li>
            <li>Break large multi-part queries into shorter questions.</li>
          </ul>
        </div>

        {/* Actions */}
        <div className="w-full space-y-3">
          <button
            type="button"
            onClick={handleReturn}
            className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-sm flex items-center justify-center space-x-2 shadow-premium hover:shadow-premium-hover hover:scale-[1.01] active:scale-[0.99] transition-all cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Return to Chat</span>
          </button>

          <button
            type="button"
            onClick={() => { setCurrentTab("dashboard"); window.location.hash = "#/dashboard"; }}
            className="w-full py-2 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center space-x-1.5 transition-colors cursor-pointer"
          >
            <Home className="w-3.5 h-3.5" />
            <span>Go to Dashboard</span>
          </button>
        </div>
      </motion.div>
    </div>
  );
}
