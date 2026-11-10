import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RotateCcw, Terminal, ChevronDown, Copy, Check, RefreshCw } from "lucide-react";

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
        <div className="min-h-screen w-full flex items-center justify-center p-4 sm:p-6 select-none bg-background text-foreground">
          {/* Workstation Quarantine Panel */}
          <div className="max-w-xl w-full rounded-2xl bg-surface-container-lowest border border-border p-6 sm:p-8 flex flex-col items-center text-center shadow-sm">
            {/* Diagnostic Badge */}
            <div className="mb-4 flex flex-col items-center">
              <div className="w-14 h-14 rounded-2xl bg-danger-subtle border border-danger-subtle flex items-center justify-center text-danger mb-3 shadow-xs">
                <AlertTriangle className="w-7 h-7 text-danger" />
              </div>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-danger-subtle border border-danger-subtle text-xs font-semibold text-danger">
                <span className="w-1.5 h-1.5 rounded-full bg-danger inline-block" />
                <span>Runtime Quarantine</span>
              </span>
            </div>

            {/* Main Heading */}
            <h1 className="text-2xl font-bold tracking-tight text-foreground mb-2">
              Application runtime exception
            </h1>

            {/* Explanation */}
            <p className="text-xs sm:text-sm text-muted-foreground max-w-md leading-relaxed mb-6 font-normal">
              KnowVex encountered an unhandled exception. The component tree has been quarantined to prevent state corruption. You can inspect the diagnostic trace or reload the application.
            </p>

            {/* Diagnostic Stack Trace Box */}
            <details className="w-full text-left border border-border bg-surface-container rounded-xl overflow-hidden mb-6 group">
              <summary className="px-3.5 py-2.5 bg-surface hover:bg-surface-container text-xs text-foreground flex items-center justify-between cursor-pointer select-none transition-colors border-b border-border/70 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40">
                <span className="flex items-center gap-2 font-medium">
                  <Terminal className="w-4 h-4 text-muted-foreground" />
                  <span>Diagnostic stack trace</span>
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={this.handleCopy}
                    className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors px-2 py-0.5 rounded bg-surface border border-border cursor-pointer active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 motion-reduce:transform-none"
                    title="Copy error trace"
                    aria-label="Copy error trace"
                  >
                    {this.state.copied ? (
                      <>
                        <Check className="w-3 h-3 text-success" />
                        <span className="text-success font-semibold">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                  <ChevronDown className="w-3.5 h-3.5 text-muted-foreground transition-transform duration-200 group-open:rotate-180" />
                </div>
              </summary>
              <div className="p-3.5 overflow-x-auto max-h-52 overflow-y-auto space-y-2 text-xs select-text bg-surface-container-lowest">
                <div className="text-danger font-semibold break-all">
                  {errorMessage}
                </div>
                <pre className="text-muted-foreground text-[11px] leading-relaxed whitespace-pre-wrap break-all select-text font-mono">
                  {errorStack}
                </pre>
              </div>
            </details>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center gap-2.5 w-full">
              <button
                type="button"
                onClick={this.handleReload}
                className="w-full sm:flex-1 bg-primary text-primary-foreground font-semibold text-xs sm:text-sm px-4 py-2.5 rounded-xl hover:opacity-90 transition-opacity cursor-pointer flex items-center justify-center gap-2 shadow-xs active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transform-none"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Reload application</span>
              </button>

              <button
                type="button"
                onClick={this.handleReset}
                className="w-full sm:flex-1 px-4 py-2.5 rounded-xl border border-border bg-surface hover:bg-surface-container text-foreground text-xs sm:text-sm font-medium transition-colors cursor-pointer flex items-center justify-center gap-1.5 active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transform-none"
              >
                <RefreshCw className="w-3.5 h-3.5 text-muted-foreground" />
                <span>Attempt recovery</span>
              </button>
            </div>

            {/* Diagnostic Watermark */}
            <div className="mt-8 text-xs text-muted-foreground flex items-center gap-2">
              <span>KnowVex Client Runtime Shield</span>
              <span className="w-1 h-1 rounded-full bg-border" />
              <span>Quarantine Mode</span>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
