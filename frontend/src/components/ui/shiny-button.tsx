"use client"

import React from "react"
import { cn } from "../../lib/utils"

export interface ShinyButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode
  onClick?: () => void
  className?: string
  size?: "sm" | "default" | "lg"
  animated?: boolean
  variant?: "black" | "brand"
}

export function ShinyButton({
  children,
  onClick,
  className = "",
  size = "default",
  animated = true,
  variant = "black",
  ...props
}: ShinyButtonProps) {
  const sizeClasses = {
    sm: "shiny-cta-sm py-1.5 px-3.5 text-xs font-semibold",
    default: "py-2.5 px-6 text-sm font-semibold",
    lg: "shiny-cta-lg py-3.5 px-8 text-base font-semibold",
  }[size]

  return (
    <>
      <style>{`
        @property --gradient-angle {
          syntax: "<angle>";
          initial-value: 0deg;
          inherits: false;
        }

        @property --gradient-angle-offset {
          syntax: "<angle>";
          initial-value: 0deg;
          inherits: false;
        }

        @property --gradient-percent {
          syntax: "<percentage>";
          initial-value: 5%;
          inherits: false;
        }

        @property --gradient-shine {
          syntax: "<color>";
          initial-value: white;
          inherits: false;
        }

        .shiny-cta {
          --shiny-cta-bg: #000000;
          --shiny-cta-bg-subtle: #1a1818;
          --shiny-cta-fg: #ffffff;
          --shiny-cta-highlight: #3b82f6;
          --shiny-cta-highlight-subtle: #8484ff;
          --animation: gradient-angle linear infinite;
          --duration: 3s;
          --shadow-size: 2px;
          --transition: 800ms cubic-bezier(0.25, 1, 0.5, 1);
          
          isolation: isolate;
          position: relative;
          overflow: hidden;
          cursor: pointer;
          outline-offset: 4px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          transform: translateZ(0);
          will-change: transform;
          gap: 0.5rem;
          font-family: inherit;
          line-height: 1.2;
          border: 1px solid transparent;
          border-radius: 360px;
          color: var(--shiny-cta-fg);
          background: linear-gradient(var(--shiny-cta-bg), var(--shiny-cta-bg)) padding-box,
            conic-gradient(
              from calc(var(--gradient-angle) - var(--gradient-angle-offset)),
              transparent,
              var(--shiny-cta-highlight) var(--gradient-percent),
              var(--gradient-shine) calc(var(--gradient-percent) * 2),
              var(--shiny-cta-highlight) calc(var(--gradient-percent) * 3),
              transparent calc(var(--gradient-percent) * 4)
            ) border-box;
          box-shadow: inset 0 0 0 1px var(--shiny-cta-bg-subtle), 0 4px 16px -2px rgba(0, 0, 0, 0.5);
          transition: var(--transition);
          transition-property: --gradient-angle-offset, --gradient-percent, --gradient-shine;
        }

        /* Brand Variant: Colored background so it doesn't get submerged in dark themes */
        .shiny-cta.shiny-cta-brand {
          --shiny-cta-bg: #5041d8;
          --shiny-cta-bg-subtle: #6a5cee;
          --shiny-cta-fg: #ffffff;
          --shiny-cta-highlight: #93c5fd;
          --shiny-cta-highlight-subtle: #ffffff;
          background: linear-gradient(135deg, #5b4fe8 0%, #4338ca 100%) padding-box,
            conic-gradient(
              from calc(var(--gradient-angle) - var(--gradient-angle-offset)),
              transparent,
              var(--shiny-cta-highlight) var(--gradient-percent),
              var(--gradient-shine) calc(var(--gradient-percent) * 2),
              var(--shiny-cta-highlight) calc(var(--gradient-percent) * 3),
              transparent calc(var(--gradient-percent) * 4)
            ) border-box;
          box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.45), 0 8px 25px -4px rgba(91, 79, 232, 0.55);
        }

        .shiny-cta.shiny-cta-brand:hover {
          background: linear-gradient(135deg, #665af0 0%, #4a3fd8 100%) padding-box,
            conic-gradient(
              from calc(var(--gradient-angle) - var(--gradient-angle-offset)),
              transparent,
              var(--shiny-cta-highlight) var(--gradient-percent),
              var(--gradient-shine) calc(var(--gradient-percent) * 2),
              var(--shiny-cta-highlight) calc(var(--gradient-percent) * 3),
              transparent calc(var(--gradient-percent) * 4)
            ) border-box;
          box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.6), 0 12px 32px -2px rgba(91, 79, 232, 0.7);
        }

        .shiny-cta::before,
        .shiny-cta::after,
        .shiny-cta span::before {
          content: "";
          pointer-events: none;
          position: absolute;
          inset-inline-start: 50%;
          inset-block-start: 50%;
          translate: -50% -50%;
          z-index: -1;
        }

        .shiny-cta:active {
          translate: 0 1px;
        }

        /* Dots pattern */
        .shiny-cta::before {
          --size: calc(100% - var(--shadow-size) * 3);
          --position: 2px;
          --space: calc(var(--position) * 2);
          width: var(--size);
          height: var(--size);
          background: radial-gradient(
            circle at var(--position) var(--position),
            white calc(var(--position) / 4),
            transparent 0
          ) padding-box;
          background-size: var(--space) var(--space);
          background-repeat: space;
          mask-image: conic-gradient(
            from calc(var(--gradient-angle) + 45deg),
            black,
            transparent 10% 90%,
            black
          );
          border-radius: inherit;
          opacity: 0.35;
          z-index: -1;
        }

        /* Inner shimmer */
        .shiny-cta::after {
          --animation: shimmer linear infinite;
          width: 100%;
          aspect-ratio: 1;
          background: linear-gradient(
            -50deg,
            transparent,
            var(--shiny-cta-highlight),
            transparent
          );
          mask-image: radial-gradient(circle at bottom, transparent 40%, black);
          opacity: 0.55;
        }

        .shiny-cta > .shiny-cta-inner {
          z-index: 1;
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
        }

        .shiny-cta > .shiny-cta-inner::before {
          --size: calc(100% + 1rem);
          width: var(--size);
          height: var(--size);
          box-shadow: inset 0 -1ex 2rem 4px var(--shiny-cta-highlight);
          opacity: 0;
          transition: opacity var(--transition);
          animation: calc(var(--duration) * 1.5) breathe linear infinite;
        }

        /* Animate */
        .shiny-cta,
        .shiny-cta::before,
        .shiny-cta::after {
          animation: var(--animation) var(--duration),
            var(--animation) calc(var(--duration) / 0.4) reverse paused;
          animation-composition: add;
        }

        .shiny-cta:is(:hover, :focus-visible) {
          --gradient-percent: 20%;
          --gradient-angle-offset: 95deg;
          --gradient-shine: var(--shiny-cta-highlight-subtle);
        }

        .shiny-cta:is(:hover, :focus-visible),
        .shiny-cta:is(:hover, :focus-visible)::before,
        .shiny-cta:is(:hover, :focus-visible)::after {
          animation-play-state: running;
        }

        .shiny-cta:is(:hover, :focus-visible) > .shiny-cta-inner::before {
          opacity: 1;
        }

        /* Static non-animated mode (for Navbar) */
        .shiny-cta.shiny-cta-static,
        .shiny-cta.shiny-cta-static::before,
        .shiny-cta.shiny-cta-static::after {
          animation: none !important;
        }

        .shiny-cta.shiny-cta-static {
          background: linear-gradient(var(--shiny-cta-bg), var(--shiny-cta-bg)) padding-box,
            linear-gradient(135deg, rgba(255, 255, 255, 0.22), rgba(255, 255, 255, 0.08)) border-box;
          box-shadow: inset 0 0 0 1px var(--shiny-cta-bg-subtle), 0 1px 3px rgba(0, 0, 0, 0.3);
          transition: all 200ms ease;
        }

        .shiny-cta.shiny-cta-static::before {
          mask-image: none !important;
          opacity: 0.25;
        }

        .shiny-cta.shiny-cta-static:hover {
          background: linear-gradient(var(--shiny-cta-bg), var(--shiny-cta-bg)) padding-box,
            linear-gradient(135deg, #5b4fe8, rgba(255, 255, 255, 0.2)) border-box;
          box-shadow: inset 0 0 0 1px var(--shiny-cta-bg-subtle), 0 0 14px -3px rgba(91, 79, 232, 0.4);
          transform: translateY(-0.5px);
        }

        @keyframes gradient-angle {
          to {
            --gradient-angle: 360deg;
          }
        }

        @keyframes shimmer {
          to {
            rotate: 360deg;
          }
        }

        @keyframes breathe {
          from, to {
            scale: 1;
          }
          50% {
            scale: 1.2;
          }
        }
      `}</style>

      <button
        className={cn(
          "group shiny-cta",
          sizeClasses,
          variant === "brand" ? "shiny-cta-brand" : "shiny-cta-black",
          !animated && "shiny-cta-static",
          className
        )}
        onClick={onClick}
        {...props}
      >
        <span className="shiny-cta-inner">{children}</span>
      </button>
    </>
  )
}
