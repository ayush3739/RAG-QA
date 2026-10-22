import type { Variants } from "framer-motion";

// ─── Easing tokens ───────────────────────────────────────────────────────────
export const premiumEase = [0.23, 1, 0.32, 1] as const;
export const premiumEaseOut = [0.16, 1, 0.3, 1] as const;
export const easeOutExpo = [0.16, 1, 0.3, 1] as const;
export const easeOutQuart = [0.25, 1, 0.5, 1] as const;

// ─── Duration tokens ─────────────────────────────────────────────────────────
export const DUR = {
  fast: 0.18,    // micro-interactions, press feedback
  base: 0.35,    // state changes, tooltips
  slow: 0.6,     // section entrances
  dramatic: 0.85 // hero, cinematic reveals
} as const;

// ─── Simple helpers ───────────────────────────────────────────────────────────
export function splitWords(text: string) {
  return text.trim().split(/\s+/).filter(Boolean);
}

// ─── Section-level variants ───────────────────────────────────────────────────

/** Basic fade-up: generic fallback */
export const pageFadeIn: Variants = {
  hidden: { opacity: 0, y: 18 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: DUR.slow, ease: premiumEase },
  },
};

/** Blur-fade up: for subheadings and descriptive copy */
export const blurFadeUp: Variants = {
  hidden: { opacity: 0, y: 22, filter: "blur(10px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: DUR.dramatic, ease: premiumEase },
  },
};

/** Slide from the left — for left-column content */
export const slideFromLeft: Variants = {
  hidden: { opacity: 0, x: -40, filter: "blur(6px)" },
  visible: {
    opacity: 1,
    x: 0,
    filter: "blur(0px)",
    transition: { duration: DUR.slow, ease: premiumEase },
  },
};

/** Slide from the right — for right-column content */
export const slideFromRight: Variants = {
  hidden: { opacity: 0, x: 40, filter: "blur(6px)" },
  visible: {
    opacity: 1,
    x: 0,
    filter: "blur(0px)",
    transition: { duration: DUR.slow, ease: premiumEase },
  },
};

/** Clip-path wipe from bottom — for cards and feature tiles */
export const clipWipeUp: Variants = {
  hidden: {
    opacity: 0,
    clipPath: "inset(100% 0 0 0 round 16px)",
    y: 12,
  },
  visible: {
    opacity: 1,
    clipPath: "inset(0% 0 0 0 round 16px)",
    y: 0,
    transition: { duration: DUR.slow, ease: premiumEase },
  },
};

/** Pop-in with scale: for badges, pills, small chips */
export const cardPopIn: Variants = {
  hidden: { opacity: 0, scale: 0.88, y: 8 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: DUR.base, ease: premiumEase },
  },
};

// ─── Word / line level ────────────────────────────────────────────────────────

export const revealWord = (delay = 0): Variants => ({
  hidden: { opacity: 0, y: 18, filter: "blur(8px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: {
      duration: 0.7,
      ease: premiumEase,
      delay,
    },
  },
});

export const revealLine: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: premiumEase },
  },
};

// ─── Container orchestration ─────────────────────────────────────────────────

export const staggerContainer = (staggerChildren = 0.08, delayChildren = 0): Variants => ({
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren,
      delayChildren,
    },
  },
});

/** Stagger with no opacity on container — children own their own reveal */
export const staggerChildren = (staggerMs = 0.08, delayMs = 0): Variants => ({
  hidden: {},
  visible: {
    transition: {
      staggerChildren: staggerMs,
      delayChildren: delayMs,
    },
  },
});
