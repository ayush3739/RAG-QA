import React from "react";
import { useStore } from "../store/useStore";
import AnimatedGradientBackground from "./ui/animated-gradient-background";
import { AnimatedShinyText } from "./magicui/animated-shiny-text";
import { BrainCircuit } from "lucide-react";
import { AnimatedBeamShowcase } from "./AnimatedBeamShowcase";
import { SignIn, SignUp } from '@clerk/react';

export default function AuthView() {
  const authMode = useStore((s) => s.authMode);
  const setAuthMode = useStore((s) => s.setAuthMode);

  return (
    <div className="relative min-h-screen w-full flex bg-black overflow-hidden selection:bg-white/30 selection:text-white">
      {/* Background Animated Gradient */}
      <AnimatedGradientBackground
        Breathing={true}
        animationSpeed={0.015}
        gradientColors={["#000000", "#040b16", "#0a192f", "#112240", "#233554", "#0a192f", "#000000"]}
        gradientStops={[20, 40, 50, 60, 75, 90, 100]}
      />

      <div className="relative z-10 w-full grid grid-cols-1 lg:grid-cols-2 h-screen">

        {/* Left Column: Showcase (Hidden on smaller screens) */}
        <div className="hidden lg:flex flex-col items-center justify-center p-12 relative border-r border-white/5 overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.15)_0%,transparent_70%)]" />
          <div className="absolute top-12 left-12 flex items-center gap-3 z-10">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-lg shadow-primary/20">
              <BrainCircuit size={20} className="text-primary-foreground" />
            </div>
            <AnimatedShinyText className="text-xl font-black tracking-tight cursor-default m-0">
              DocuMind
            </AnimatedShinyText>
          </div>

          <div className="flex flex-col items-center justify-center w-full h-full max-h-[600px] z-10">
            <AnimatedBeamShowcase />
            <div className="mt-8 text-center max-w-sm space-y-2 z-10">
              <h3 className="text-xl font-bold text-white tracking-tight">Curated Knowledge.</h3>
              <p className="text-sm text-neutral-400">
                Connect your documents to the Deep Forest vector engine and extract precise insights in real-time.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column: Clerk Auth Components */}
        <div className="flex items-center justify-center w-full p-6 lg:p-12 bg-[#090d16]/80 backdrop-blur-xl relative border-l border-white/5">
          <div className="w-full max-w-[440px] flex flex-col items-center justify-center">
            {/* Top Company Logo (Mobile Only) */}
            <div className="flex lg:hidden justify-center items-center gap-2 mb-12 text-white">
              <BrainCircuit size={24} />
              <span className="text-xl font-semibold tracking-tight">DocuMind Inc.</span>
            </div>

            <div className="flex flex-col items-center w-full space-y-6">
              {authMode === "login" ? (
                <SignIn 
                  routing="hash"
                  fallbackRedirectUrl="/"
                  appearance={{
                    elements: {
                      footer: {
                        display: "none"
                      }
                    }
                  }}
                />
              ) : (
                <SignUp 
                  routing="hash"
                  fallbackRedirectUrl="/"
                  appearance={{
                    elements: {
                      footer: {
                        display: "none"
                      }
                    }
                  }}
                />
              )}

              {/* Custom Navigation Toggle */}
              <div className="text-center">
                <button
                  onClick={() => setAuthMode(authMode === "login" ? "register" : "login")}
                  className="text-xs font-semibold text-zinc-400 hover:text-white transition-colors"
                >
                  {authMode === "login"
                    ? "Don't have an account? Sign up"
                    : "Already have an account? Sign in"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
