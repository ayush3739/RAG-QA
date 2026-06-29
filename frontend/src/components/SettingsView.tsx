import React, { useState } from "react";
import { 
  Settings, Cpu, Sliders, User, Shield, RefreshCw, Moon, Sun, LogOut, Trash2, Save, Check
} from "lucide-react";
import { ModelParams } from "../types";
import { cn } from "../lib/utils";
import { useQuery } from "@tanstack/react-query";
import { useStore } from "../store/useStore";
import { api } from "../lib/api";
import ConfirmDialog from "./ConfirmDialog";

interface SettingsViewProps {
  params: ModelParams;
  onParamChange: (newParams: Partial<ModelParams>) => void;
}

export default function SettingsView({ params, onParamChange }: SettingsViewProps) {
  // Local preferences states
  const [useOllama, setUseOllama] = useState(() => {
    return localStorage.getItem("setting_use_ollama") === "true";
  });
  const [showChunks, setShowChunks] = useState(() => {
    return localStorage.getItem("setting_show_chunks") === "true";
  });
  const [inspectorOpen, setInspectorOpen] = useState(() => {
    return localStorage.getItem("setting_inspector_open") !== "false";
  });
  
  const user = useStore(s => s.user);
  const logout = useStore(s => s.logout);
  const [displayName, setDisplayName] = useState(user?.name || "");
  const [nameSaved, setNameSaved] = useState(false);
  const [isSavingName, setIsSavingName] = useState(false);
  const [showDeleteAccount, setShowDeleteAccount] = useState(false);

  const theme = useStore(s => s.theme);
  const setTheme = useStore(s => s.setTheme);

  const handleToggleOllama = (val: boolean) => {
    setUseOllama(val);
    localStorage.setItem("setting_use_ollama", String(val));
    onParamChange({ selectedModel: val ? "qwen3:4b" : "gemini-3.5-flash" });
  };

  const handleToggleChunks = (val: boolean) => {
    setShowChunks(val);
    localStorage.setItem("setting_show_chunks", String(val));
  };

  const handleToggleInspector = (val: boolean) => {
    setInspectorOpen(val);
    localStorage.setItem("setting_inspector_open", String(val));
  };

  const { data: rawHealth, isFetching: isCheckingHealth, refetch, isError } = useQuery({
    queryKey: ['rawHealth'],
    queryFn: async () => {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Could not contact API endpoints.");
      return res.json();
    },
    enabled: false,
    retry: false
  });

  const handleCheckHealth = () => {
    refetch();
  };

  const handleSaveName = async () => {
    if (!displayName.trim() || isSavingName) return;
    setIsSavingName(true);
    try {
      await api.updateMe(displayName.trim());
      setNameSaved(true);
      setTimeout(() => setNameSaved(false), 2000);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSavingName(false);
    }
  };

  const handleDeleteAccount = async () => {
    try {
      await api.deleteMe();
    } catch (e) {
      // 204 means success
    } finally {
      logout();
    }
  };

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-12 py-10 space-y-10 bg-background selection:bg-primary/10">
      
      {/* Title */}
      <div className="max-w-4xl mx-auto">
        <h2 className="text-2xl font-bold text-foreground flex items-center tracking-tight">
          <Settings className="w-5 h-5 mr-3 text-muted-foreground" />
          Settings
        </h2>
        <p className="text-sm text-muted-foreground mt-1.5 font-medium">
          Adjust model provider parameters, toggle general workspace options, inspect backend health endpoints, and manage user profile data.
        </p>
      </div>

      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Left Column: LLM & Web Preferences */}
        <div className="space-y-6">
          
          {/* Section 1: LLM Selection Toggles */}
          <div className="bg-surface-container-lowest p-6 rounded-2xl border border-border space-y-4 shadow-sm">
            <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider flex items-center">
              <Cpu className="w-3.5 h-3.5 mr-2" />
              LLM Provider Option
            </h3>

            <div className="space-y-3">
              <div 
                onClick={() => handleToggleOllama(true)}
                className={cn(
                  "p-4 rounded-xl border cursor-pointer transition-all",
                  useOllama 
                    ? "bg-primary/5 border-primary text-foreground" 
                    : "bg-surface border-border text-muted-foreground hover:bg-surface-container-low/50"
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-foreground">Local Model (Ollama)</span>
                  <input type="radio" checked={useOllama} readOnly className="accent-foreground" />
                </div>
                <p className="text-xs mt-1.5 leading-relaxed text-muted-foreground">
                  Queries stay stored within your isolated container environment. Excellent for offline high-privacy requirements.
                </p>
              </div>

              <div 
                onClick={() => handleToggleOllama(false)}
                className={cn(
                  "p-4 rounded-xl border cursor-pointer transition-all",
                  !useOllama 
                    ? "bg-primary/5 border-primary text-foreground" 
                    : "bg-surface border-border text-muted-foreground hover:bg-surface-container-low/50"
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-foreground">Cloud Model (Gemini)</span>
                  <input type="radio" checked={!useOllama} readOnly className="accent-foreground" />
                </div>
                <p className="text-xs mt-1.5 leading-relaxed text-muted-foreground">
                  Leverages powerful cloud models to execute deep reasoning operations with cited grounding context.
                </p>
              </div>
            </div>
          </div>

          {/* Section 2: Developer Debug Options */}
          <div className="bg-surface-container-lowest p-6 rounded-2xl border border-border space-y-4 shadow-sm">
            <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider flex items-center">
              <Sliders className="w-3.5 h-3.5 mr-2" />
              Developer Options
            </h3>

            <div className="space-y-3 divide-y divide-border">
              
              <div className="flex items-center justify-between py-2">
                <div>
                  <span className="text-sm font-semibold text-foreground block">Show Grounded Chunks</span>
                  <p className="text-xs text-muted-foreground leading-none mt-1">Expose dense vector retrieval math chunks in panel</p>
                </div>
                <input
                  type="checkbox"
                  checked={showChunks}
                  onChange={(e) => handleToggleChunks(e.target.checked)}
                  className="accent-foreground w-4 h-4 cursor-pointer"
                />
              </div>

              <div className="flex items-center justify-between pt-3">
                <div>
                  <span className="text-sm font-semibold text-foreground block">Inspector Auto-Open</span>
                  <p className="text-xs text-muted-foreground leading-none mt-1">Slide open context inspector automatically on stream</p>
                </div>
                <input
                  type="checkbox"
                  checked={inspectorOpen}
                  onChange={(e) => handleToggleInspector(e.target.checked)}
                  className="accent-foreground w-4 h-4 cursor-pointer"
                />
              </div>

            </div>
          </div>

          {/* Section: Appearance (Theme) */}
          <div className="bg-surface-container-lowest p-6 rounded-2xl border border-border space-y-4 shadow-sm">
            <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider flex items-center">
              <Sun className="w-3.5 h-3.5 mr-2" />
              Appearance
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div 
                onClick={() => setTheme("light")}
                className={cn(
                  "p-3 flex items-center justify-center space-x-2 rounded-xl border cursor-pointer transition-all font-semibold text-sm shadow-sm",
                  theme === "light" 
                    ? "bg-primary text-primary-foreground border-primary" 
                    : "bg-surface border-border text-muted-foreground hover:bg-surface-container-high"
                )}
              >
                <Sun className="w-4 h-4" />
                <span>Light</span>
              </div>
              <div 
                onClick={() => setTheme("dark")}
                className={cn(
                  "p-3 flex items-center justify-center space-x-2 rounded-xl border cursor-pointer transition-all font-semibold text-sm shadow-sm",
                  theme === "dark" 
                    ? "bg-primary text-primary-foreground border-primary" 
                    : "bg-surface border-border text-muted-foreground hover:bg-surface-container-high"
                )}
              >
                <Moon className="w-4 h-4" />
                <span>Dark</span>
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Profile & API Health Checks */}
        <div className="space-y-6">
          
          {/* Section 3: User Profile */}
          <div className="bg-surface-container-lowest p-6 rounded-2xl border border-border space-y-4 shadow-sm">
            <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider flex items-center">
              <User className="w-3.5 h-3.5 mr-2" />
              Account Identity
            </h3>

            <div className="space-y-4">
              
              {/* Display Name */}
              <div className="space-y-1.5">
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block">Display Name</label>
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
                    className="flex-1 text-sm font-medium bg-background border border-border rounded-lg px-3 py-2 text-foreground focus:outline-none focus:ring-1 focus:ring-foreground/20 shadow-sm transition-all"
                  />
                  <button
                    onClick={handleSaveName}
                    disabled={isSavingName || !displayName.trim()}
                    className="p-2 rounded-lg bg-surface border border-border hover:bg-surface-container text-foreground transition-all cursor-pointer disabled:opacity-40"
                    title="Save name"
                  >
                    {nameSaved ? <Check className="w-4 h-4 text-emerald-500" /> : <Save className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Readonly Email */}
              <div className="space-y-1.5">
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block">System Email (Read-Only)</label>
                <input
                  type="text"
                  readOnly
                  value={user?.email || ""}
                  className="w-full text-sm font-medium bg-surface border border-border rounded-lg px-3 py-2 text-muted-foreground cursor-not-allowed focus:outline-none"
                />
              </div>

              {/* Password simulation */}
              <div className="space-y-1.5">
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block">Password</label>
                <input
                  type="password"
                  value="••••••••••••••"
                  readOnly
                  className="w-full text-sm font-medium bg-surface border border-border rounded-lg px-3 py-2 text-muted-foreground cursor-not-allowed focus:outline-none"
                />
              </div>
              
              {/* Logout + Delete Account */}
              <div className="pt-2 space-y-2">
                <button
                  onClick={logout}
                  className="w-full flex items-center justify-center space-x-2 bg-surface hover:bg-surface-container border border-border text-foreground font-semibold text-sm py-2.5 rounded-lg transition-all"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Log Out</span>
                </button>
                <button
                  onClick={() => setShowDeleteAccount(true)}
                  className="w-full flex items-center justify-center space-x-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 font-semibold text-sm py-2.5 rounded-lg transition-all border border-red-500/20"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Delete Account</span>
                </button>
              </div>

              <ConfirmDialog
                isOpen={showDeleteAccount}
                title="Delete your account?"
                description="This will permanently delete your account, all your documents, sessions, and messages. This action is irreversible."
                confirmLabel="Yes, Delete Everything"
                onConfirm={handleDeleteAccount}
                onCancel={() => setShowDeleteAccount(false)}
              />

            </div>
          </div>

          {/* Section 4: System Status Logs */}
          <div className="bg-surface-container-lowest p-6 rounded-2xl border border-border space-y-4 shadow-sm">
            <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider flex items-center">
              <Shield className="w-3.5 h-3.5 mr-2" />
              Health Diagnostics
            </h3>

            <div className="space-y-3">
              <p className="text-xs text-muted-foreground leading-relaxed">
                Connect directly to your isolated workspace server endpoint to verify memory caches, system uptime, and core dependencies.
              </p>

              <button
                onClick={handleCheckHealth}
                disabled={isCheckingHealth}
                className="px-4 py-2 bg-primary text-primary-foreground font-semibold text-xs rounded-md cursor-pointer hover:bg-primary/90 active:scale-95 transition-all flex items-center space-x-1.5 shadow-sm"
              >
                <RefreshCw className={cn("w-3.5 h-3.5", isCheckingHealth ? "animate-spin" : "")} />
                <span>{isCheckingHealth ? "Querying..." : "Query Node Status"}</span>
              </button>

              {rawHealth && !isError && (
                <div className="bg-background border border-border rounded-lg p-3 max-h-40 overflow-y-auto">
                  <pre className="text-[11px] font-mono text-muted-foreground whitespace-pre-wrap">
                    {JSON.stringify(rawHealth, null, 2)}
                  </pre>
                </div>
              )}
              {isError && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 max-h-40 overflow-y-auto">
                  <pre className="text-[11px] font-mono text-red-500 whitespace-pre-wrap">
                    {`{\n  "error": "Could not contact API endpoints."\n}`}
                  </pre>
                </div>
              )}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
