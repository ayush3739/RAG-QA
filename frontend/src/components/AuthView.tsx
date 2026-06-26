import React, { useState } from "react";
import { useStore } from "../store/useStore";
import AnimatedGradientBackground from "./ui/animated-gradient-background";
import { AnimatedShinyText } from "./magicui/animated-shiny-text";
import { BrainCircuit, ArrowRight } from "lucide-react";
import { cn } from "../lib/utils";

export default function AuthView() {
  const setIsAuthenticated = useStore((s) => s.setIsAuthenticated);
  const authMode = useStore((s) => s.authMode);
  const setAuthMode = useStore((s) => s.setAuthMode);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Simulate auth success
    setIsAuthenticated(true);
  };

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center bg-black overflow-hidden selection:bg-primary/20 selection:text-primary">
      {/* Background Animated Gradient */}
      <AnimatedGradientBackground 
        Breathing={true}
        animationSpeed={0.015}
        gradientColors={["#000000", "#040b16", "#0a192f", "#112240", "#233554", "#0a192f", "#000000"]}
        gradientStops={[20, 40, 50, 60, 75, 90, 100]}
      />

      {/* Floating Auth Card */}
      <div className="relative z-10 w-full max-w-[420px] px-6">
        {/* Top Company Logo (like shadcn example) */}
        <div className="flex justify-center items-center gap-2 mb-8 text-white">
          <BrainCircuit size={20} />
          <span className="font-semibold tracking-tight">DocuMind Inc.</span>
        </div>

        <div className="relative flex flex-col bg-[#0A0A0A] rounded-[16px] p-8 border border-white/10 shadow-2xl">
          
          {/* Header */}
          <div className="flex flex-col items-center text-center space-y-2 mb-6">
            <h1 className="text-2xl font-semibold text-white tracking-tight">
              {authMode === "login" ? "Welcome back" : "Create an account"}
            </h1>
            <p className="text-sm text-neutral-400">
              {authMode === "login" 
                ? "Login with your Apple or Google account" 
                : "Sign up with your Apple or Google account"}
            </p>
          </div>

          {/* Social Buttons */}
          <div className="space-y-3 mb-6">
            <button className="w-full flex items-center justify-center gap-2 py-2.5 rounded-md border border-neutral-800 hover:bg-neutral-900 transition-colors text-sm font-medium text-white">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.379.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.161 22 16.416 22 12c0-5.523-4.477-10-10-10z"/>
              </svg>
              Login with GitHub
            </button>
            <button className="w-full flex items-center justify-center gap-2 py-2.5 rounded-md border border-neutral-800 hover:bg-neutral-900 transition-colors text-sm font-medium text-white">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" className="w-4 h-4">
                <path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/>
                <path fill="#FF3D00" d="m6.306 14.691 6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/>
                <path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/>
                <path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/>
              </svg>
              Login with Google
            </button>
          </div>

          <div className="relative flex items-center justify-center mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-neutral-800"></div>
            </div>
            <div className="relative bg-[#0A0A0A] px-2 text-xs text-neutral-500">
              Or continue with
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {authMode === "register" && (
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-foreground/80 px-1">Full Name</label>
                <input 
                  type="text" 
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="John Doe"
                  className="w-full bg-secondary/50 border-none rounded-xl px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-primary/50 transition-all outline-none"
                  required
                />
              </div>
            )}
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-white">Email</label>
              <input 
                type="email" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="m@example.com"
                className="w-full bg-transparent border border-neutral-800 rounded-md px-3 py-2 text-sm text-white placeholder:text-neutral-500 focus:outline-none focus:ring-2 focus:ring-white/20 transition-all"
                required
              />
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium text-white">Password</label>
                {authMode === "login" && (
                  <button type="button" className="text-sm font-medium text-neutral-400 hover:text-white transition-colors">
                    Forgot your password?
                  </button>
                )}
              </div>
              <input 
                type="password" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-transparent border border-neutral-800 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-white/20 transition-all"
                required
              />
            </div>

            <button 
              type="submit" 
              className="w-full bg-white hover:bg-neutral-200 text-black font-medium text-sm py-2.5 rounded-md transition-colors mt-4"
            >
              {authMode === "login" ? "Login" : "Sign Up"}
            </button>
          </form>

          {/* Toggle */}
          <div className="mt-8 text-center">
            <button 
              onClick={() => setAuthMode(authMode === "login" ? "register" : "login")}
              className="text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              {authMode === "login" 
                ? "Don't have an account? Sign up" 
                : "Already have an account? Sign in"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
