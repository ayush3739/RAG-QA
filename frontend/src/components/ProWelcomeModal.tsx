import React, { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Crown, Sparkles, Zap, Files, Cpu, ArrowRight, X, Check } from "lucide-react";
import { useStore } from "../store/useStore";

interface ProWelcomeModalProps {
  isOpen?: boolean;
  onClose?: () => void;
}

// Sparkle particle coordinates and timing for celebration
const CELEBRATION_PARTICLES = [
  { x: -140, y: -90, delay: 0.1, size: 14 },
  { x: 130, y: -80, delay: 0.25, size: 16 },
  { x: -160, y: 30, delay: 0.4, size: 12 },
  { x: 150, y: 40, delay: 0.15, size: 15 },
  { x: -90, y: -130, delay: 0.35, size: 13 },
  { x: 100, y: -120, delay: 0.2, size: 14 },
];

export function getProWelcomeStorageKeys(user: { id?: number | string; email?: string } | null | undefined): string[] {
  if (!user) return [];
  const keys: string[] = [];
  if (user.id !== undefined && user.id !== null) {
    keys.push(`knowvex_pro_welcomed_${user.id}`);
  }
  if (user.email) {
    keys.push(`knowvex_pro_welcomed_${user.email}`);
  }
  return keys;
}

export function isProWelcomed(user: { id?: number | string; email?: string } | null | undefined): boolean {
  const keys = getProWelcomeStorageKeys(user);
  return keys.some((k) => typeof window !== "undefined" && window.localStorage.getItem(k) === "true");
}

export function markProWelcomed(user: { id?: number | string; email?: string } | null | undefined): void {
  const keys = getProWelcomeStorageKeys(user);
  keys.forEach((k) => {
    try {
      window.localStorage.setItem(k, "true");
    } catch {
      // Storage quota or sandboxed iframe safeguard
    }
  });
}

export default function ProWelcomeModal({ isOpen, onClose }: ProWelcomeModalProps) {
  const isStoreOpen = useStore((s) => s.isProWelcomeOpen);
  const setIsStoreOpen = useStore((s) => s.setIsProWelcomeOpen);
  const user = useStore((s) => s.user);
  const primaryButtonRef = useRef<HTMLButtonElement>(null);

  const show = isOpen !== undefined ? isOpen : isStoreOpen;

  const handleDismiss = () => {
    markProWelcomed(user);
    if (onClose) {
      onClose();
    } else {
      setIsStoreOpen(false);
    }
  };

  // Lock body scroll while modal is mounted & active
  useEffect(() => {
    if (!show) return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    // Auto-focus primary CTA for keyboard navigation
    const timer = setTimeout(() => {
      primaryButtonRef.current?.focus();
    }, 150);
    return () => {
      document.body.style.overflow = originalOverflow;
      clearTimeout(timer);
    };
  }, [show]);

  // Keyboard navigation: Escape key dismisses modal
  useEffect(() => {
    if (!show) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        handleDismiss();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [show, user]);

  const perks = [
    {
      icon: Zap,
      title: "500 Daily Agent Queries",
      tag: "50x BOOST",
      desc: "Instant zero-queue answering with deep contextual retrieval across your workspaces.",
    },
    {
      icon: Sparkles,
      title: "25 Deep Research Syntheses",
      tag: "AGENTIC",
      desc: "Multi-angle autonomous agent research cross-synthesizing web and private documents.",
    },
    {
      icon: Files,
      title: "50 Knowledge Base Documents",
      tag: "300 PAGES/DOC",
      desc: "High-capacity vector indexing powered by BGE cross-encoder reranking algorithms.",
    },
    {
      icon: Cpu,
      title: "Priority Compute Pipeline",
      tag: "FAST LANE",
      desc: "Dedicated GPU queue routing for ultra-low latency real-time SSE token streaming.",
    },
  ];

  return (
    <AnimatePresence>
      {show && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="pro-welcome-title"
          aria-describedby="pro-welcome-desc"
          className="fixed inset-0 z-[120] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md"
        >
          {/* Backdrop Click Dismiss */}
          <div
            className="absolute inset-0 cursor-pointer"
            onClick={handleDismiss}
            aria-hidden="true"
          />

          {/* Modal Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 16 }}
            transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
            className="relative w-full max-w-xl overflow-hidden rounded-3xl border border-amber-500/35 bg-[#0B0C10] p-6 sm:p-8 text-foreground shadow-[0_0_80px_rgba(245,158,11,0.25)] select-none z-10"
          >
            {/* Ambient Warm Golden Radial Glows */}
            <div className="pointer-events-none absolute -top-32 -left-32 h-72 w-72 rounded-full bg-amber-500/18 blur-[110px]" />
            <div className="pointer-events-none absolute -bottom-32 -right-32 h-72 w-72 rounded-full bg-yellow-500/12 blur-[110px]" />

            {/* Floating Sparkle Particles */}
            <div className="pointer-events-none absolute inset-0 overflow-hidden">
              {CELEBRATION_PARTICLES.map((p, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.4 }}
                  animate={{
                    opacity: [0.2, 0.9, 0.3],
                    scale: [0.8, 1.25, 0.9],
                    y: [0, -12, 0],
                  }}
                  transition={{
                    duration: 3 + idx * 0.5,
                    repeat: Infinity,
                    delay: p.delay,
                    ease: "easeInOut",
                  }}
                  style={{
                    position: "absolute",
                    left: `calc(50% + ${p.x}px)`,
                    top: `calc(28% + ${p.y}px)`,
                  }}
                  className="text-amber-300 drop-shadow-[0_0_8px_rgba(251,191,36,0.8)]"
                >
                  <Sparkles style={{ width: p.size, height: p.size }} />
                </motion.div>
              ))}
            </div>

            {/* Top Close Button */}
            <button
              type="button"
              onClick={handleDismiss}
              className="absolute top-4 right-4 rounded-xl p-2 text-neutral-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-400"
              aria-label="Close celebration modal"
            >
              <X className="h-5 w-5" />
            </button>

            {/* Header Content */}
            <div className="flex flex-col items-center text-center relative z-10">
              {/* Luxury Crown Emblem */}
              <div className="relative mb-3 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-tr from-amber-500 via-yellow-400 to-amber-300 p-0.5 shadow-[0_0_35px_rgba(245,158,11,0.5)]">
                <div className="flex h-full w-full items-center justify-center rounded-[14px] bg-[#0E0F15]">
                  <Crown className="h-10 w-10 text-amber-400 fill-amber-400/25 drop-shadow-[0_0_12px_rgba(245,158,11,0.8)]" />
                </div>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 16, repeat: Infinity, ease: "linear" }}
                  className="absolute -top-1.5 -right-1.5"
                >
                  <Sparkles className="h-5 w-5 text-amber-300 fill-amber-300 drop-shadow-[0_0_8px_rgba(252,211,77,0.8)]" />
                </motion.div>
              </div>

              {/* Gold Pill Badge */}
              <div className="mb-2">
                <span className="text-[11px] font-extrabold px-3 py-1 rounded-full bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500 text-neutral-950 inline-flex items-center gap-1.5 shadow-[0_0_15px_rgba(245,158,11,0.35)] tracking-wider uppercase">
                  <Crown className="w-3.5 h-3.5 fill-neutral-950 text-neutral-950" />
                  PRO TIER ACTIVATED
                </span>
              </div>

              {/* Headline */}
              <h3
                id="pro-welcome-title"
                className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white leading-tight"
              >
                Congratulations! You're Now KnowVex Pro
              </h3>

              {/* Subtitle */}
              <p
                id="pro-welcome-desc"
                className="mt-2 text-xs sm:text-sm text-neutral-300 max-w-md mx-auto leading-relaxed"
              >
                Your account has been elevated to our premier tier. Here are your upgraded capabilities:
              </p>
            </div>

            {/* 4 Perk Cards Grid */}
            <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 relative z-10">
              {perks.map((perk, idx) => {
                const Icon = perk.icon;
                return (
                  <div
                    key={idx}
                    className="flex items-start gap-3 rounded-2xl border border-amber-500/20 bg-neutral-950/70 p-3.5 backdrop-blur-md transition-all hover:border-amber-500/40 hover:bg-neutral-950/90 shadow-[0_2px_12px_rgba(0,0,0,0.3)]"
                  >
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.2)]">
                      <Icon className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1 text-left">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-bold text-neutral-100 truncate">{perk.title}</span>
                        <span className="text-[9px] font-extrabold px-1.5 py-0.5 rounded bg-amber-400/10 text-amber-400 border border-amber-400/25 shrink-0">
                          {perk.tag}
                        </span>
                      </div>
                      <p className="mt-1 text-[11px] text-neutral-400 leading-snug">
                        {perk.desc}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Bottom Primary Action */}
            <div className="mt-7 flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-amber-500/20 relative z-10">
              <div className="flex items-center gap-1.5 text-xs text-amber-200/90 font-medium">
                <Check className="h-4 w-4 text-amber-400" />
                <span>All limits immediately refreshed</span>
              </div>

              <button
                ref={primaryButtonRef}
                type="button"
                onClick={handleDismiss}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-amber-400 via-yellow-400 to-amber-500 hover:from-amber-300 hover:to-amber-400 text-neutral-950 font-extrabold text-sm py-2.5 px-6 shadow-[0_0_20px_rgba(245,158,11,0.4)] hover:shadow-[0_0_30px_rgba(245,158,11,0.6)] transition-all active:scale-[0.98] cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-400"
              >
                <span>Enter Pro Workspace</span>
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
