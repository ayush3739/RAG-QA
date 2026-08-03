import React, { useState } from "react";
import AnimatedGradientBackground from "./ui/animated-gradient-background";
import { AnimatedShinyText } from "./magicui/animated-shiny-text";
import { BrainCircuit, Eye, EyeOff, Loader2, CheckCircle2, KeyRound } from "lucide-react";
import { api, ApiError } from "../lib/api";

interface ResetPasswordViewProps {
  token: string;
  onNavigateToLogin: () => void;
}

export default function ResetPasswordView({ token, onNavigateToLogin }: ResetPasswordViewProps) {
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!token) {
      setError("Reset token is missing from the URL link.");
      return;
    }

    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setIsLoading(true);

    try {
      const res = await api.resetPassword(token, newPassword);
      setIsSuccess(true);
      setSuccessMessage(res.message || "Password successfully reset. Please log in with your new password.");
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(err.message || "Failed to reset password. Token may be invalid or expired.");
      }
    } finally {
      setIsLoading(false);
    }
  };

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
        <div className="flex justify-center items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-lg shadow-primary/20">
            <BrainCircuit size={20} className="text-primary-foreground" />
          </div>
          <AnimatedShinyText className="text-xl font-black tracking-tight cursor-default m-0">
            DocuMind
          </AnimatedShinyText>
        </div>

        {isSuccess ? (
          <div className="flex flex-col items-center text-center space-y-4 py-4">
            <div className="w-14 h-14 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <h2 className="text-xl font-semibold text-white">Password Reset</h2>
            <p className="text-sm text-zinc-400 max-w-xs">{successMessage}</p>

            <button
              onClick={onNavigateToLogin}
              className="w-full flex items-center justify-center gap-2 bg-white hover:bg-neutral-200 text-black font-semibold text-sm py-2.5 rounded-lg transition-all duration-300 mt-4 shadow-lg"
            >
              Sign In with New Password
            </button>
          </div>
        ) : (
          <div className="flex flex-col space-y-6">
            <div className="flex flex-col items-center text-center space-y-1.5">
              <div className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-white mb-2">
                <KeyRound size={22} />
              </div>
              <h1 className="text-xl font-semibold text-white tracking-tight">Set New Password</h1>
              <p className="text-xs text-neutral-400">
                Please enter your new password below.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs px-3 py-2 rounded-lg font-medium">
                  {error}
                </div>
              )}

              <div className="space-y-1.5">
                <label className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">
                  New Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full h-10 px-3 pr-10 rounded-lg bg-[#24292e] border border-[#3b4148] text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-150"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">
                  Confirm Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full h-10 px-3 pr-10 rounded-lg bg-[#24292e] border border-[#3b4148] text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-150"
                    required
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex items-center justify-center gap-2 bg-white hover:bg-neutral-200 text-black font-semibold text-sm py-2.5 rounded-lg transition-all duration-300 mt-6 shadow-lg disabled:opacity-70 disabled:cursor-not-allowed"
              >
                {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                Reset Password
              </button>
            </form>

            <div className="text-center">
              <button
                onClick={onNavigateToLogin}
                className="text-xs font-medium text-zinc-400 hover:text-white transition-colors"
              >
                Back to Sign In
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
