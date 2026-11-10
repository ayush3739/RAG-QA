import { navigate } from '../lib/navigation';
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
      navigate('/dashboard');
    }
  };

  const handleGoChat = () => {
    if (onNavigate) {
      onNavigate("conversations");
    } else {
      useStore.getState().setActiveConvId("");
      setCurrentTab("conversations");
      navigate('/chat/new');
    }
  };

  const handleGoDocuments = () => {
    if (onNavigate) {
      onNavigate("documents");
    } else {
      setCurrentTab("documents");
      navigate('/documents');
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
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 bg-background text-foreground select-none">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="relative z-10 max-w-md w-full text-center flex flex-col items-center"
      >
        {/* Precision Diagnostic Badge */}
        <div className="mb-5 flex flex-col items-center">
          <div className="w-16 h-16 rounded-2xl bg-surface-container border border-border flex items-center justify-center text-primary mb-3 shadow-xs">
            <Compass className="w-8 h-8 text-primary motion-safe:animate-spin motion-reduce:animate-none" style={{ animationDuration: "24s" }} />
          </div>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-surface-container border border-border text-xs font-semibold text-muted-foreground">
            <span className="w-1.5 h-1.5 rounded-full bg-warning inline-block" />
            <span className="font-mono tabular-nums">404</span>
            <span>Unresolved Route</span>
          </span>
        </div>

        {/* Heading & Prose */}
        <h1 className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2">
          Page not found
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed mb-7 max-w-sm">
          The requested route or resource does not exist, has moved, or the link is no longer valid.
        </p>

        {/* Action Panel */}
        <div className="w-full space-y-2.5">
          {isAuthenticated ? (
            <>
              <button
                type="button"
                onClick={handleGoDashboard}
                className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-xs sm:text-sm flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 transition-opacity cursor-pointer shadow-xs motion-reduce:transform-none"
              >
                <Home className="w-4 h-4" />
                <span>Return to Dashboard</span>
              </button>

              <div className="grid grid-cols-2 gap-2.5">
                <button
                  type="button"
                  onClick={handleGoChat}
                  className="py-2.5 px-3 rounded-xl bg-surface border border-border text-foreground font-medium text-xs flex items-center justify-center gap-1.5 hover:bg-surface-container active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 transition-colors cursor-pointer motion-reduce:transform-none"
                >
                  <Plus className="w-3.5 h-3.5 text-primary" />
                  <span>Start Research</span>
                </button>

                <button
                  type="button"
                  onClick={handleGoDocuments}
                  className="py-2.5 px-3 rounded-xl bg-surface border border-border text-foreground font-medium text-xs flex items-center justify-center gap-1.5 hover:bg-surface-container active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 transition-colors cursor-pointer motion-reduce:transform-none"
                >
                  <FolderOpen className="w-3.5 h-3.5 text-muted-foreground" />
                  <span>Knowledge Base</span>
                </button>
              </div>
            </>
          ) : (
            <button
              type="button"
              onClick={() => { navigate('/login'); }}
              className="w-full py-2.5 px-4 rounded-xl bg-primary text-primary-foreground font-semibold text-sm flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 transition-opacity cursor-pointer shadow-xs motion-reduce:transform-none"
            >
              <span>Sign in to KnowVex</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleGoBack}
            className="w-full py-2 text-xs text-muted-foreground hover:text-foreground flex items-center justify-center gap-1.5 transition-colors cursor-pointer active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 whitespace-nowrap motion-reduce:transform-none"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Go back</span>
          </button>
        </div>

        {/* Diagnostic Watermark */}
        <div className="mt-10 text-xs text-muted-foreground flex items-center gap-2">
          <span>KnowVex Workstation Core</span>
          <span className="w-1 h-1 rounded-full bg-border" />
          <span>v2.0</span>
        </div>
      </motion.div>
    </div>
  );
}
