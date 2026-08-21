import React from "react";
import { motion } from "framer-motion";
import { Home, Plus, FolderOpen, ArrowLeft, Compass } from "lucide-react";
import { useStore } from "../store/useStore";

interface NotFoundViewProps {
  onNavigate?: (tab: string) => void;
}

export default function NotFoundView({ onNavigate }: NotFoundViewProps) {
  const setCurrentTab = useStore((s) => s.setCurrentTab);
  const isAuthenticated = useStore((s) => s.isAuthenticated);

  const handleGoDashboard = () => {
    if (onNavigate) {
      onNavigate("dashboard");
    } else {
      setCurrentTab("dashboard");
      window.location.hash = "#/dashboard";
    }
  };

  const handleGoChat = () => {
    if (onNavigate) {
      onNavigate("conversations");
    } else {
      useStore.getState().setActiveConvId("");
      setCurrentTab("conversations");
      window.location.hash = "#/chat/new";
    }
  };

  const handleGoDocuments = () => {
    if (onNavigate) {
      onNavigate("documents");
    } else {
      setCurrentTab("documents");
      window.location.hash = "#/documents";
    }
  };

  const handleGoBack = () => {
    if (window.history.length > 1) {
      window.history.back();
    } else {
      handleGoDashboard();
    }
  };

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 bg-background text-foreground overflow-hidden select-none">
      {/* Ambient glowing orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="ambient-orb orb-1 bg-primary/10 w-[600px] h-[600px] -top-[150px] left-1/2 -translate-x-1/2 blur-3xl rounded-full" />
        <div className="ambient-orb orb-2 bg-secondary/10 w-[400px] h-[400px] bottom-[-100px] right-1/4 blur-3xl rounded-full" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative z-10 max-w-md w-full text-center flex flex-col items-center"
      >
        {/* Icon & Error Code */}
        <div className="relative mb-6">
          <div className="w-20 h-20 rounded-2xl bg-surface-container border border-border flex items-center justify-center text-primary shadow-premium">
            <Compass className="w-10 h-10 animate-spin" style={{ animationDuration: "20s" }} />
          </div>
          <span className="absolute -bottom-2 -right-2 px-2 py-0.5 text-[11px] font-mono font-bold bg-primary text-primary-foreground rounded-full shadow-sm">
            404
          </span>
        </div>

        {/* Heading */}
        <h1 className="text-3xl font-extrabold text-foreground tracking-tight mb-2">
          Page Not Found
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed mb-8 max-w-sm">
          The page or route you were looking for doesn't exist, has moved, or the link is invalid.
        </p>

        {/* Action Buttons */}
        <div className="w-full space-y-3">
          {isAuthenticated ? (
            <>
              <button
                type="button"
                onClick={handleGoDashboard}
                className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-sm flex items-center justify-center space-x-2 shadow-premium hover:shadow-premium-hover hover:scale-[1.01] active:scale-[0.99] transition-all cursor-pointer"
              >
                <Home className="w-4 h-4" />
                <span>Return to Dashboard</span>
              </button>

              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={handleGoChat}
                  className="py-2.5 px-3 rounded-xl bg-surface-container border border-border text-foreground font-medium text-xs flex items-center justify-center space-x-2 hover:bg-surface-container-high active:scale-[0.98] transition-all cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5 text-primary" />
                  <span>Start New Chat</span>
                </button>

                <button
                  type="button"
                  onClick={handleGoDocuments}
                  className="py-2.5 px-3 rounded-xl bg-surface-container border border-border text-foreground font-medium text-xs flex items-center justify-center space-x-2 hover:bg-surface-container-high active:scale-[0.98] transition-all cursor-pointer"
                >
                  <FolderOpen className="w-3.5 h-3.5 text-secondary" />
                  <span>Knowledge Base</span>
                </button>
              </div>
            </>
          ) : (
            <button
              type="button"
              onClick={() => { window.location.hash = "#/auth"; }}
              className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-sm flex items-center justify-center space-x-2 shadow-premium hover:scale-[1.01] active:scale-[0.99] transition-all cursor-pointer"
            >
              <span>Sign In to DocuMind</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleGoBack}
            className="w-full py-2 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center space-x-1 transition-colors cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Go back to previous page</span>
          </button>
        </div>

        {/* Brand footer watermark */}
        <div className="mt-12 text-[11px] font-mono text-muted-foreground/60 flex items-center space-x-1.5">
          <span>DocuMind Intelligence Core</span>
          <span>•</span>
          <span>v2.0</span>
        </div>
      </motion.div>
    </div>
  );
}
