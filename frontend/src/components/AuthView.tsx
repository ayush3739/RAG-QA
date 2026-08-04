import React, { useState, useMemo } from "react";
import { useStore, AuthMode } from "../store/useStore";
import AnimatedGradientBackground from "./ui/animated-gradient-background";
import { AnimatedShinyText } from "./magicui/animated-shiny-text";
import { BrainCircuit, Eye, EyeOff, Loader2, Mail, CheckCircle2, AlertTriangle, ArrowLeft, ShieldCheck } from "lucide-react";
import { AnimatedBeamShowcase } from "./AnimatedBeamShowcase";
import { api, ApiError, NetworkError } from "../lib/api";

// Password strength helper
function getPasswordStrength(pw: string): { score: number; label: string; color: string } {
  if (pw.length === 0) return { score: 0, label: "", color: "" };
  let score = 0;
  if (pw.length >= 8) score++;
  if (pw.length >= 12) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  if (score <= 1) return { score, label: "Weak", color: "bg-red-500" };
  if (score <= 2) return { score, label: "Fair", color: "bg-orange-400" };
  if (score <= 3) return { score, label: "Good", color: "bg-yellow-400" };
  return { score, label: "Strong", color: "bg-emerald-500" };
}

export default function AuthView() {
  const authMode = useStore((s) => s.authMode);
  const setAuthMode = useStore((s) => s.setAuthMode);
  const setTokens = useStore((s) => s.setTokens);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [unverifiedNotice, setUnverifiedNotice] = useState<string | null>(null);
  const [successInfo, setSuccessInfo] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const passwordStrength = useMemo(() => getPasswordStrength(password), [password]);
  const isPasswordInvalid = authMode === "register" && password.length > 0 && password.length < 8;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setUnverifiedNotice(null);
    setIsLoading(true);

    // Frontend validation
    if (authMode === "register" && password.length < 8) {
      setError("Password must be at least 8 characters.");
      setIsLoading(false);
      return;
    }

    try {
      if (authMode === "login") {
        const res = await api.login(email, password);
        setTokens(res.access_token, res.refresh_token);
      } else if (authMode === "register") {
        const res = await api.register(name, email, password);
        setSuccessInfo(res.message || "User registered successfully. Please verify your email before logging in.");
        setAuthMode("verification-sent");
      } else if (authMode === "forgot-password") {
        const res = await api.forgotPassword(email);
        setSuccessInfo(res.message || "If an account with that email exists, a password reset link has been sent.");
        setAuthMode("forgot-password-sent");
      }
    } catch (err: any) {
      if (err instanceof NetworkError) {
        setError('No internet connection. Please check your network and try again.');
      } else if (err instanceof ApiError) {
        if (err.status === 403) {
          setUnverifiedNotice(err.message || "Email not verified. Please check your inbox for the verification link.");
        } else {
          setError(err.message);
        }
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const resetFormState = (newMode: AuthMode) => {
    setError("");
    setUnverifiedNotice(null);
    setAuthMode(newMode);
  };

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
          <div className="absolute inset-0 pointer-events-none z-0 opacity-30">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.15)_0%,transparent_70%)]" />
          </div>

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

        {/* Right Column: Auth Form */}
        <div className="flex items-center justify-center w-full p-6 lg:p-12 overflow-y-auto">
          <div className="w-full max-w-[440px]">
            {/* Top Company Logo (Mobile Only) */}
            <div className="flex lg:hidden justify-center items-center gap-2 mb-12 text-white">
              <BrainCircuit size={24} />
              <span className="text-xl font-semibold tracking-tight">DocuMind Inc.</span>
            </div>

            <div className="relative flex flex-col w-full">
              {/* Verification Sent View */}
              {authMode === "verification-sent" && (
                <div className="flex flex-col items-center text-center space-y-5 py-4">
                  <div className="w-16 h-16 rounded-full bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                    <Mail size={32} />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-white tracking-tight">Check your email</h2>
                    <p className="text-sm text-zinc-400 max-w-xs leading-relaxed">
                      {successInfo || "We sent a verification link to your email. Please verify your email before logging in."}
                    </p>
                  </div>
                  <button
                    onClick={() => resetFormState("login")}
                    className="w-full flex items-center justify-center gap-2 bg-white hover:bg-neutral-200 text-black font-semibold text-sm py-2.5 rounded-lg transition-all duration-300 mt-4 shadow-lg"
                  >
                    Proceed to Sign In
                  </button>
                </div>
              )}

              {/* Forgot Password Sent View */}
              {authMode === "forgot-password-sent" && (
                <div className="flex flex-col items-center text-center space-y-5 py-4">
                  <div className="w-16 h-16 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                    <CheckCircle2 size={32} />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-white tracking-tight">Reset Link Sent</h2>
                    <p className="text-sm text-zinc-400 max-w-xs leading-relaxed">
                      {successInfo}
                    </p>
                  </div>
                  <button
                    onClick={() => resetFormState("login")}
                    className="w-full flex items-center justify-center gap-2 bg-white hover:bg-neutral-200 text-black font-semibold text-sm py-2.5 rounded-lg transition-all duration-300 mt-4 shadow-lg"
                  >
                    Back to Sign In
                  </button>
                </div>
              )}

              {/* Login, Register, Forgot Password Forms */}
              {(authMode === "login" || authMode === "register" || authMode === "forgot-password") && (
                <>
                  {/* Header */}
                  <div className="flex flex-col items-center text-center space-y-2 mb-6">
                    <h1 className="text-2xl font-semibold text-white tracking-tight">
                      {authMode === "login"
                        ? "Welcome back"
                        : authMode === "register"
                        ? "Create an account"
                        : "Reset your password"}
                    </h1>
                    <p className="text-sm text-neutral-400">
                      {authMode === "login"
                        ? "Enter your credentials to access your account"
                        : authMode === "register"
                        ? "Sign up to start building with DocuMind"
                        : "Enter your email address and we'll send you a reset link"}
                    </p>
                  </div>

                  {authMode !== "forgot-password" && (
                    <div className="space-y-4 mb-6">
                      <button
                        type="button"
                        onClick={() => { window.location.href = "http://localhost:8000/api/v1/auth/github/login"; }}
                        className="w-full flex items-center justify-center gap-3 py-2.5 rounded-lg bg-[#24292e] border border-[#3b4148] hover:bg-[#2f363d] transition-all duration-300 text-sm font-medium text-white shadow-sm"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                          <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.379.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.161 22 16.416 22 12c0-5.523-4.477-10-10-10z" />
                        </svg>
                        Continue with GitHub
                      </button>
                      <button
                        type="button"
                        onClick={() => { window.location.href = "http://localhost:8000/api/v1/auth/google/login"; }}
                        className="w-full flex items-center justify-center gap-3 py-2.5 rounded-lg bg-[#24292e] border border-[#3b4148] hover:bg-[#2f363d] transition-all duration-300 text-sm font-medium text-white shadow-sm"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" className="w-4 h-4">
                          <path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z" />
                          <path fill="#FF3D00" d="m6.306 14.691 6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z" />
                          <path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z" />
                          <path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z" />
                        </svg>
                        Continue with Google
                      </button>

                      <div className="relative flex items-center justify-center my-6">
                        <div className="absolute inset-0 flex items-center">
                          <div className="w-full border-t border-white/10"></div>
                        </div>
                        <div className="relative px-4 text-xs font-medium text-neutral-400 bg-black/20 backdrop-blur-sm rounded-full">
                          Or continue with
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Form */}
                  <form onSubmit={handleSubmit} className="space-y-4">
                    {error && (
                      <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs px-3.5 py-2.5 rounded-lg font-medium">
                        {error}
                      </div>
                    )}

                    {unverifiedNotice && (
                      <div className="bg-amber-500/10 border border-amber-500/25 text-amber-300 text-xs p-3 rounded-lg flex items-start gap-2.5">
                        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                        <div className="space-y-1">
                          <p className="font-semibold text-amber-200">Email Verification Required</p>
                          <p className="leading-normal">{unverifiedNotice}</p>
                        </div>
                      </div>
                    )}

                    {authMode === "register" && (
                      <div className="space-y-1.5">
                        <label className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">
                          Full Name
                        </label>
                        <input
                          type="text"
                          value={name}
                          onChange={(e) => setName(e.target.value)}
                          placeholder="John Doe"
                          className="w-full h-10 px-3 rounded-lg bg-[#24292e] border border-[#3b4148] text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-150"
                          required
                        />
                      </div>
                    )}

                    <div className="space-y-1.5">
                      <label className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">
                        Email
                      </label>
                      <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="m@example.com"
                        className="w-full h-10 px-3 rounded-lg bg-[#24292e] border border-[#3b4148] text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all duration-150"
                        required
                      />
                    </div>

                    {authMode !== "forgot-password" && (
                      <div className="space-y-1.5">
                        <div className="flex justify-between items-center">
                          <label className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">
                            Password
                          </label>
                          {authMode === "login" && (
                            <button
                              type="button"
                              onClick={() => resetFormState("forgot-password")}
                              className="text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors"
                            >
                              Forgot your password?
                            </button>
                          )}
                        </div>
                        <div className="relative">
                          <input
                            type={showPassword ? "text" : "password"}
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                          placeholder="••••••••"
                            autoComplete={authMode === "register" ? "new-password" : "current-password"}
                            className={`w-full h-10 px-3 pr-10 rounded-lg bg-[#24292e] border text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-1 transition-all duration-150 ${
                              isPasswordInvalid
                                ? "border-red-500/60 focus:border-red-500 focus:ring-red-500"
                                : "border-[#3b4148] focus:border-blue-500 focus:ring-blue-500"
                            }`}
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

                        {/* Password strength bar (register only) */}
                        {authMode === "register" && password.length > 0 && (
                          <div className="mt-2 space-y-1">
                            <div className="flex gap-1">
                              {[1,2,3,4].map((i) => (
                                <div
                                  key={i}
                                  className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                                    passwordStrength.score >= i ? passwordStrength.color : "bg-white/10"
                                  }`}
                                />
                              ))}
                            </div>
                            <div className="flex items-center justify-between">
                              <p className={`text-[11px] font-medium ${
                                passwordStrength.score <= 1 ? "text-red-400" :
                                passwordStrength.score <= 2 ? "text-orange-400" :
                                passwordStrength.score <= 3 ? "text-yellow-400" : "text-emerald-400"
                              }`}>
                                {passwordStrength.label}
                              </p>
                              {isPasswordInvalid && (
                                <p className="text-[11px] text-red-400">Min. 8 characters</p>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    <button
                      type="submit"
                      disabled={isLoading || isPasswordInvalid}
                      className="w-full flex items-center justify-center gap-2 bg-white hover:bg-neutral-200 text-black font-semibold text-sm py-2.5 rounded-lg transition-all duration-300 mt-6 shadow-[inset_0_1px_0_rgba(255,255,255,0.4)] shadow-[0_0_20px_-5px_rgba(255,255,255,0.3)] hover:-translate-y-0.5 disabled:opacity-50 disabled:hover:translate-y-0 disabled:cursor-not-allowed"
                    >
                      {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                      {authMode === "login"
                        ? "Login"
                        : authMode === "register"
                        ? "Sign Up"
                        : "Send Password Reset Link"}
                    </button>
                  </form>

                  {/* Toggle / Nav actions */}
                  <div className="mt-8 text-center space-y-3">
                    {authMode === "forgot-password" ? (
                      <button
                        onClick={() => resetFormState("login")}
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-300 hover:text-white transition-colors"
                      >
                        <ArrowLeft size={14} />
                        Back to Sign In
                      </button>
                    ) : (
                      <button
                        onClick={() => resetFormState(authMode === "login" ? "register" : "login")}
                        className="text-xs font-medium text-zinc-300 hover:text-white transition-colors"
                      >
                        {authMode === "login"
                          ? "Don't have an account? Sign up"
                          : "Already have an account? Sign in"}
                      </button>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
