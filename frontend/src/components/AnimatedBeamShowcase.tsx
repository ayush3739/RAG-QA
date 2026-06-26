"use client";

import React, { useRef } from "react";
import { AnimatedBeam } from "./magicui/animated-beam";
import { FileText, Database, BrainCircuit, User } from "lucide-react";

const Circle = React.forwardRef<
  HTMLDivElement,
  { className?: string; children?: React.ReactNode }
>(({ className, children }, ref) => {
  return (
    <div
      ref={ref}
      className={`z-10 flex h-16 w-16 items-center justify-center rounded-full border-2 border-white/10 bg-black/50 backdrop-blur-md shadow-[0_0_20px_-12px_rgba(255,255,255,0.8)] ${className}`}
    >
      {children}
    </div>
  );
});

Circle.displayName = "Circle";

export function AnimatedBeamShowcase() {
  const containerRef = useRef<HTMLDivElement>(null);
  const div1Ref = useRef<HTMLDivElement>(null);
  const div2Ref = useRef<HTMLDivElement>(null);
  const div3Ref = useRef<HTMLDivElement>(null);
  const div4Ref = useRef<HTMLDivElement>(null);
  const div5Ref = useRef<HTMLDivElement>(null);
  const div6Ref = useRef<HTMLDivElement>(null);
  const div7Ref = useRef<HTMLDivElement>(null);

  return (
    <div
      className="relative flex h-full w-full max-w-[500px] items-center justify-center p-10"
      ref={containerRef}
    >
      <div className="flex h-full w-full flex-row items-stretch justify-between gap-10">
        <div className="flex flex-col justify-center gap-10">
          <Circle ref={div1Ref}>
            <FileText className="text-white/80" />
          </Circle>
          <Circle ref={div2Ref}>
            <FileText className="text-white/80" />
          </Circle>
          <Circle ref={div3Ref}>
            <FileText className="text-white/80" />
          </Circle>
        </div>
        <div className="flex flex-col justify-center">
          <Circle ref={div4Ref} className="h-24 w-24 border-primary/30 shadow-[0_0_30px_-5px_rgba(59,130,246,0.3)]">
            <Database className="h-10 w-10 text-primary" />
          </Circle>
        </div>
        <div className="flex flex-col justify-center gap-10">
          <Circle ref={div6Ref}>
            <BrainCircuit className="text-white/80" />
          </Circle>
          <Circle ref={div7Ref}>
            <User className="text-white/80" />
          </Circle>
        </div>
      </div>

      <AnimatedBeam
        containerRef={containerRef}
        fromRef={div1Ref}
        toRef={div4Ref}
        curvature={-50}
        pathOpacity={0.1}
        gradientStartColor="#000000"
        gradientStopColor="#3b82f6"
        duration={3}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={div2Ref}
        toRef={div4Ref}
        curvature={0}
        pathOpacity={0.1}
        gradientStartColor="#000000"
        gradientStopColor="#3b82f6"
        duration={3}
        delay={1}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={div3Ref}
        toRef={div4Ref}
        curvature={50}
        pathOpacity={0.1}
        gradientStartColor="#000000"
        gradientStopColor="#3b82f6"
        duration={3}
        delay={2}
      />
      
      {/* Brain connection */}
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={div4Ref}
        toRef={div6Ref}
        curvature={-20}
        pathOpacity={0.1}
        gradientStartColor="#3b82f6"
        gradientStopColor="#8b5cf6"
        duration={3}
        delay={0.5}
      />

      {/* User connection */}
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={div6Ref}
        toRef={div7Ref}
        curvature={20}
        pathOpacity={0.1}
        gradientStartColor="#8b5cf6"
        gradientStopColor="#ffffff"
        duration={3}
        delay={1.5}
      />
    </div>
  );
}
