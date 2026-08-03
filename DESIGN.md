---
name: Hexon RAG-QA
description: Intelligent document QA platform
colors:
  primary: "#7B6CF6"
  secondary: "#2DD4BF"
  neutral-bg: "#08090C"
  neutral-surface: "#0E0E11"
  neutral-border: "#27272A"
  neutral-muted: "#94A3B8"
typography:
  display:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    fontWeight: 600
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "Inter, sans-serif"
    fontWeight: 500
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Inter, sans-serif"
    fontWeight: 400
  label:
    fontFamily: "JetBrains Mono, monospace"
    fontWeight: 500
rounded:
  sm: "4px"
  md: "8px"
  lg: "12px"
  xl: "0.75rem"
  2xl: "1rem"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
components:
  premium-card:
    backgroundColor: "{colors.neutral-surface}"
    rounded: "{rounded.xl}"
    padding: "24px"
---

# Design System: Hexon RAG-QA

## 1. Overview

**Creative North Star: "The Precision Workspace"**

Premium, understated, and focused. This system is designed for high-density, analytical workflows where the UI gets out of the way of the data. It leans heavily on a dark-first, Linear-inspired aesthetic that prioritizes sharp contrast, extremely tight borders, and deliberate micro-interactions over decorative flair. The aesthetic is explicitly technical and crisp, avoiding loud gradients, excessive rounded corners, and playful marketing tropes.

**Key Characteristics:**
- Dark-first, high-density layout.
- Violet (Iris) and Teal (Mint) as strategic, high-contrast accents.
- Subtle ambient glows instead of hard drop shadows.
- Typography that pairs clean geometric sans (Inter) with monospace (JetBrains Mono) for data points.

## 2. Colors

Iris & Mint: Sophisticated tech with refined contrast against deep near-black surfaces.

### Primary
- **Iris** (#7B6CF6): Used for primary actions, active states, and glowing focus rings. It carries the brand's premium, technical feel.

### Secondary
- **Mint** (#2DD4BF): Used for data points, citations, active indicators, and secondary highlights. It provides a sharp, highly legible contrast against the dark background.

### Neutral
- **Deep Background** (#08090C): The absolute lowest layer of the application canvas.
- **Surface** (#0E0E11): The baseline background for cards, panels, and floating elements.
- **Border** (#27272A): Extremely subtle division lines. Never used thicker than 1px.
- **Muted Text** (#94A3B8): For secondary labels, timestamps, and deactivated states.

**The Focus Rule.** The primary accent is used sparingly. Its rarity guarantees that when a button or input ring glows Iris, the user's eye is drawn exactly where it needs to be.

## 3. Typography

**Display Font:** Inter (with system sans-serif fallback)
**Body Font:** Inter (with system sans-serif fallback)
**Label/Mono Font:** JetBrains Mono (with system monospace fallback)

**Character:** Technical, highly legible, and unopinionated. The pairing of Inter for prose and JetBrains Mono for data perfectly balances readability with a developer-centric edge.

### Hierarchy
- **Display** (600, clamp, -0.04em): Hero headlines and major page titles. Tightly tracked to maintain a compact, designed feel.
- **Headline** (500, -0.02em): Section headers and primary card titles.
- **Title** (500, normal): Subsections and list headers.
- **Body** (400, 1.5): Standard prose and long-form document reading. Maximum line length capped at 75ch.
- **Label** (500, JetBrains Mono, uppercase): Metadata, timestamps, citations, and micro-copy.

**The Tracking Rule.** Display headings must have letter-spacing set to at least -0.04em. Anything looser feels unpolished; anything tighter causes letters to touch.

## 4. Elevation

Ambient Depth: Surfaces are flat at rest, relying on subtle glows and tight borders. We do not use traditional hard drop shadows to simulate physical depth.

### Shadow Vocabulary
- **Premium Shadow** (`0 1px 3px rgba(13,15,26,0.08)`): The absolute baseline for cards, barely perceptible.
- **Ambient Glow** (`0 0 24px -4px rgba(91,79,232,0.28)`): Used strictly for focus states (like the chat input) and primary button hovers to create a "light emitting" effect rather than physical elevation.

**The Flat-By-Default Rule.** Surfaces are flat at rest. Depth is only revealed through interaction (hover/focus) or to separate a modal from the background canvas.

## 5. Components

Tactile and crisp, utilizing tight padding and sharp, composited transitions.

### Premium Cards
- **Shape:** Gently rounded (0.75rem / 12px). Never pill-shaped.
- **Background:** `rgba(13, 13, 15, 0.65)` with a 20px backdrop blur.
- **Border:** `rgba(255, 255, 255, 0.08)` at exactly 1px.
- **Hover:** Translates up slightly (-1.5px) with a subtle expansion of the ambient shadow.

### Chat Input
- **Style:** 1px subtle border, deep background.
- **Focus:** The signature `input-glow-ring` effect applies a multiple-layered box-shadow that includes a sharp 1px Iris ring and a diffuse 22px Iris ambient glow.

### Buttons
- **Shape:** Tight radius (0.5rem / 8px).
- **Primary:** Iris background with white text.
- **Hover / Focus:** Scale-down on press (`transform: scale(0.96)`) using an exponential ease-out curve (`cubic-bezier(0.16, 1, 0.3, 1)`).

## 6. Do's and Don'ts

Concrete, forceful guardrails to maintain the Precision Workspace aesthetic.

### Do:
- **Do** use `rgba(255, 255, 255, 0.08)` for borders on dark cards to keep them incredibly subtle.
- **Do** cap body text line lengths at 75ch for optimal reading.
- **Do** use `JetBrains Mono` for any technical data, citations, or timestamps.

### Don't:
- **Don't** use side-stripe borders (`border-left` greater than 1px) as colored accents on cards or callouts. Use full borders or nothing.
- **Don't** use gradient text (`background-clip: text`). It reads as a cheap SaaS cliché and destroys the understated premium feel.
- **Don't** apply `border-radius` larger than 16px to cards or sections. Over-rounding destroys the technical, crisp nature of the UI.
- **Don't** use the hero-metric template (big number, small label, gradient accent).
- **Don't** add tiny uppercase tracked eyebrows (`01 · ABOUT`) above every section. It reads as generated scaffolding.
