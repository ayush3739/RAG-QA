import React, { useEffect, useState } from "react";
import AnimatedGradientBackground from "./ui/animated-gradient-background";
import { AnimatedShinyText } from "./magicui/animated-shiny-text";
import { BrainCircuit, Loader2, CheckCircle2, AlertCircle, ArrowRight } from "lucide-react";
import { api, ApiError } from "../lib/api";

interface VerifyEmailViewProps {
  token: string;
  onNavigateToLogin: () => void;
}

export default function VerifyEmailView({ token, onNavigateToLogin }: VerifyEmailViewProps) {
  const [status, setStatus] = useState<"verifying" | "success" | "error">("verifying");
  const [message, setMessage] = useState<string>("Verifying your email address...");

  useEffect(() => {
    let isMounted = true;

    async function doVerify() {
      if (!token) {
        if (isMounted) {
          setStatus("error");
          setMessage("Verification token is missing from the link.");
        }
        return;
      }

      try {
        const res = await api.verifyEmail(token);
        if (isMounted) {
          setStatus("success");
          setMessage(res.message || "Email verified successfully! You can now log in.");
        }
      } catch (err: any) {
        if (isMounted) {
          setStatus("error");
          if (err instanceof ApiError) {
            setMessage(err.message);
          } else {
            setMessage(err.message || "Failed to verify email. The link may be invalid or expired.");
          }
        }
      }
    }

    doVerify();

    return () => {
      isMounted = false;
    };
  }, [token]);

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center bg-black overflow-hidden selection:bg-white/30 selection:text-white p-4">
      {/* Background Animated Gradient */}
      <AnimatedGradientBackground
        Breathing={true}
        animationSpeed={0.015}
        gradientColors={["#000000", "#040b16", "#0a192f", "#112240", "#233554", "#0a192f", "#000000"]}
        gradientStops={[20, 40, 50, 60, 75, 90, 100]}
      />

      <div className="relative z-10 w-full max-w-md bg-zinc-950/80 border border-white/10 backdrop-blur-xl rounded-2xl p-8 shadow-2xl">
        {/* Header Logo */}
        <div className="flex justify-center items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-lg shadow-primary/20">
            <BrainCircuit size={20} className="text-primary-foreground" />
          </div>
          <AnimatedShinyText className="text-xl font-black tracking-tight cursor-default m-0">
            DocuMind
          </AnimatedShinyText>
        </div>

        {/* Content depending on status */}
        {status === "verifying" && (
          <div className="flex flex-col items-center text-center space-y-4 py-6">
            <div className="w-14 h-14 rounded-full bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
              <Loader2 className="w-7 h-7 text-blue-400 animate-spin" />
            </div>
            <h2 className="text-xl font-semibold text-white">Verifying Email</h2>
            <p className="text-sm text-zinc-400 max-w-xs">{message}</p>
          </div>
        )}

        {status === "success" && (
          <div className="flex flex-col items-center text-center space-y-4 py-6">
            <div className="w-14 h-14 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <h2 className="text-xl font-semibold text-white">Email Verified!</h2>
            <p className="text-sm text-zinc-400 max-w-xs">{message}</p>

            <button
              onClick={onNavigateToLogin}
              className="w-full flex items-center justify-center gap-2 bg-white hover:bg-neutral-200 text-black font-semibold text-sm py-2.5 rounded-lg transition-all duration-300 mt-4 shadow-lg"
            >
              <span>Proceed to Login</span>
              <ArrowRight size={16} />
            </button>
          </div>
        )}

        {status === "error" && (
          <div className="flex flex-col items-center text-center space-y-4 py-6">
            <div className="w-14 h-14 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400">
              <AlertCircle className="w-8 h-8" />
            </div>
            <h2 className="text-xl font-semibold text-white">Verification Failed</h2>
            <p className="text-sm text-red-300 max-w-xs bg-red-500/10 border border-red-500/20 rounded-lg p-3">
              {message}
            </p>

            <button
              onClick={onNavigateToLogin}
              className="w-full flex items-center justify-center gap-2 bg-white/10 hover:bg-white/20 border border-white/20 text-white font-medium text-sm py-2.5 rounded-lg transition-all duration-300 mt-4"
            >
              <span>Back to Login</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
