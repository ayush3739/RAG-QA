import React, { useState } from "react";
import { motion } from "framer-motion";
import { ServerCrash, RefreshCw, Home } from "lucide-react";
import { api } from "../lib/api";
import { useStore } from "../store/useStore";

export default function ServerErrorView() {
  const [isRetrying, setIsRetrying] = useState(false);
  const setCurrentTab = useStore((s) => s.setCurrentTab);

  const handleRetry = async () => {
    setIsRetrying(true);
    try {
      await api.getHealth();
      setCurrentTab("dashboard");
      window.location.hash = "#/dashboard";
    } catch {
      // Still offline
    } finally {
      setIsRetrying(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 bg-background text-foreground overflow-hidden select-none">
      {/* Ambient glowing orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="ambient-orb orb-1 bg-red-500/10 w-[500px] h-[500px] -top-[100px] left-1/2 -translate-x-1/2 blur-3xl rounded-full" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative z-10 max-w-md w-full text-center flex flex-col items-center"
      >
        {/* Icon Badge */}
        <div className="relative mb-6">
          <div className="w-20 h-20 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-500 shadow-premium">
            <ServerCrash className="w-10 h-10 animate-bounce" />
          </div>
          <span className="absolute -bottom-2 -right-2 px-2 py-0.5 text-[11px] font-mono font-bold bg-red-500 text-white rounded-full shadow-sm">
            500
          </span>
        </div>

        {/* Heading */}
        <h1 className="text-2xl font-extrabold text-foreground tracking-tight mb-2">
          Internal Server Error
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed mb-8 max-w-sm">
          The DocuMind API server encountered an unexpected error or is temporarily unavailable. Please try reconnecting.
        </p>

        {/* Actions */}
        <div className="w-full space-y-3">
          <button
            type="button"
            onClick={handleRetry}
            disabled={isRetrying}
            className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-sm flex items-center justify-center space-x-2 shadow-premium hover:shadow-premium-hover hover:scale-[1.01] active:scale-[0.99] transition-all cursor-pointer disabled:opacity-60"
          >
            <RefreshCw className={`w-4 h-4 ${isRetrying ? "animate-spin" : ""}`} />
            <span>{isRetrying ? "Attempting Reconnection..." : "Retry Connection"}</span>
          </button>

          <button
            type="button"
            onClick={() => { window.location.hash = "#/"; }}
            className="w-full py-2 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center space-x-1.5 transition-colors cursor-pointer"
          >
            <Home className="w-3.5 h-3.5" />
            <span>Return to Home</span>
          </button>
        </div>
      </motion.div>
    </div>
  );
}
