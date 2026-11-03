import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RotateCcw, Terminal, ChevronDown, Copy, Check } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  copied: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
    copied: false,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    // Production telemetry / console reporting
    console.error("KnowVex Uncaught Application Error:", error, errorInfo);
  }

  private handleReload = (): void => {
    window.location.href = "/";
  };

  private handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      copied: false,
    });
  };

  private handleCopy = (e: React.MouseEvent): void => {
    e.stopPropagation();
    const details = [
      `Error: ${this.state.error?.message}`,
      `Stack: ${this.state.error?.stack || "No stack trace"}`,
      `Component Stack: ${this.state.errorInfo?.componentStack || "No component stack"}`,
    ].join("\n\n");

    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(details).then(() => {
        this.setState({ copied: true });
        setTimeout(() => this.setState({ copied: false }), 2000);
      });
    }
  };

  public render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        if (typeof this.props.fallback === "function") {
          return this.props.fallback(
            this.state.error || new Error("Unknown error"),
            this.handleReset
          );
        }
        return this.props.fallback;
      }

      const errorMessage = this.state.error?.message || "An unexpected error occurred.";
      const errorStack = this.state.error?.stack || this.state.errorInfo?.componentStack || "No trace available.";

      return (
        <div className="min-h-screen w-full flex items-center justify-center p-4 sm:p-6 select-none bg-[#020617] text-slate-100 relative overflow-hidden font-sans">
          {/* Cyberpunk Ambient Lighting */}
          <div className="absolute -top-32 -left-32 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-purple-600/15 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.06),transparent_70%)] pointer-events-none" />

          {/* Error Container Card */}
          <div className="relative max-w-2xl w-full rounded-2xl bg-[#090D1A]/95 backdrop-blur-xl border border-indigo-500/30 shadow-[0_0_50px_-10px_rgba(99,102,241,0.25)] p-6 sm:p-10 flex flex-col items-center text-center z-10">
            {/* Glowing Icon */}
            <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-4 shadow-[0_0_20px_rgba(99,102,241,0.2)]">
              <AlertTriangle className="w-7 h-7 text-indigo-400" />
            </div>

            {/* Sub-heading / Badge */}
            <div className="font-mono text-[11px] tracking-widest uppercase text-indigo-400 bg-indigo-500/10 border border-indigo-500/30 px-3 py-1 rounded-full mb-3">
              KnowVex Runtime Exception
            </div>

            {/* Main Heading */}
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white mb-2">
              Something went wrong
            </h1>

            {/* Explanation */}
            <p className="text-sm text-slate-400 max-w-md leading-relaxed mb-6 font-normal">
              KnowVex encountered an unexpected runtime error. System state has been quarantined to prevent data loss. You can inspect the diagnostic details below or reload the session to recover.
            </p>

            {/* Collapsible Error Monospace Box */}
            <details className="w-full text-left border border-indigo-500/20 bg-slate-950/80 rounded-xl overflow-hidden mb-6 group transition-all">
              <summary className="px-4 py-3 bg-slate-900/60 hover:bg-slate-900/90 text-xs font-mono text-slate-300 flex items-center justify-between cursor-pointer select-none transition-colors border-b border-indigo-500/10">
                <span className="flex items-center gap-2 text-indigo-300 font-semibold">
                  <Terminal className="w-4 h-4 text-indigo-400" />
                  Diagnostic Stack Trace
                </span>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={this.handleCopy}
                    className="flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-indigo-300 transition-colors px-2 py-0.5 rounded bg-slate-800/60 border border-slate-700/60 cursor-pointer"
                    title="Copy error trace"
                  >
                    {this.state.copied ? (
                      <>
                        <Check className="w-3 h-3 text-emerald-400" />
                        <span className="text-emerald-400">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                  <ChevronDown className="w-4 h-4 text-slate-400 transition-transform duration-200 group-open:rotate-180" />
                </div>
              </summary>
              <div className="p-4 overflow-x-auto max-h-56 overflow-y-auto space-y-2 font-mono text-xs select-text">
                <div className="text-rose-400 font-semibold break-all">
                  {errorMessage}
                </div>
                <pre className="text-slate-400 text-[11px] leading-relaxed whitespace-pre-wrap break-all select-text font-mono">
                  {errorStack}
                </pre>
              </div>
            </details>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
              <button
                type="button"
                onClick={this.handleReload}
                className="w-full sm:w-auto bg-gradient-to-r from-indigo-600 via-indigo-500 to-purple-600 hover:from-indigo-500 hover:via-indigo-400 hover:to-purple-500 text-white font-medium text-sm px-6 py-2.5 rounded-xl shadow-[0_0_25px_rgba(99,102,241,0.4)] hover:shadow-[0_0_35px_rgba(99,102,241,0.6)] transition-all cursor-pointer flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Reload Application</span>
              </button>

              <button
                type="button"
                onClick={this.handleReset}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl border border-slate-800 hover:border-slate-700 bg-slate-900/60 hover:bg-slate-900 text-slate-300 hover:text-white text-sm font-medium transition-colors cursor-pointer flex items-center justify-center"
              >
                Try Recovering
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

