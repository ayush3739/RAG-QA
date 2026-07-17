import React, { useState, useEffect } from "react";
import { 
  Cpu, Sliders, User, Shield, RefreshCw, Moon, Sun, LogOut, Trash2, Save, Check,
  Server, Palette, AlertTriangle
} from "lucide-react";
import { ModelParams } from "../types";
import { cn } from "../lib/utils";
import { useQuery } from "@tanstack/react-query";
import { useStore } from "../store/useStore";
import { api } from "../lib/api";
import { motion, AnimatePresence } from "framer-motion";
import ConfirmDialog from "./ConfirmDialog";

interface SettingsViewProps {
  params: ModelParams;
  onParamChange: (newParams: Partial<ModelParams>) => void;
}

// Reusable toggle switch
function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none",
        checked ? "bg-primary" : "bg-surface-container-high"
      )}
    >
      <span
        className={cn(
          "pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-lg ring-0 transition duration-200",
          checked ? "translate-x-4" : "translate-x-0"
        )}
      />
    </button>
  );
}

// Section wrapper
function SettingsSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1 pt-2 first:pt-0">
      <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground px-1 mb-3">{title}</h4>
      {children}
    </div>
  );
}

// Row within a card
function SettingsRow({ label, desc, children, last = false }: { label: string; desc?: string; children: React.ReactNode; last?: boolean }) {
  return (
    <div className={cn("flex items-center justify-between py-3.5", !last && "border-b border-border/50")}>
      <div className="min-w-0 pr-6">
        <span className="text-sm font-semibold text-foreground block">{label}</span>
        {desc && <p className="text-[12px] text-muted-foreground mt-0.5 leading-snug">{desc}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

const SECTIONS = [
  { id: "llm", label: "LLM Provider", icon: Cpu },
  { id: "appearance", label: "Appearance", icon: Palette },
  { id: "developer", label: "Developer", icon: Sliders },
  { id: "account", label: "Account", icon: User },
  { id: "health", label: "System Health", icon: Shield },
];

export default function SettingsView({ params, onParamChange }: SettingsViewProps) {
  const [activeSection, setActiveSection] = useState("llm");
  const [useOllama, setUseOllama] = useState(() => localStorage.getItem("setting_use_ollama") === "true");
  const [showChunks, setShowChunks] = useState(() => localStorage.getItem("setting_show_chunks") === "true");
  const [inspectorOpen, setInspectorOpen] = useState(() => localStorage.getItem("setting_inspector_open") !== "false");

  const user = useStore(s => s.user);
  const logout = useStore(s => s.logout);
  const theme = useStore(s => s.theme);
  const setTheme = useStore(s => s.setTheme);

  const [displayName, setDisplayName] = useState(user?.name || "");
  const [nameSaved, setNameSaved] = useState(false);
  const [isSavingName, setIsSavingName] = useState(false);
  const [showDeleteAccount, setShowDeleteAccount] = useState(false);

  useEffect(() => { if (user?.name) setDisplayName(user.name); }, [user]);

  const handleToggleOllama = (val: boolean) => {
    setUseOllama(val);
    localStorage.setItem("setting_use_ollama", String(val));
    onParamChange({ selectedModel: val ? "qwen3:4b" : "gemini-3.5-flash" });
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

  const handleSaveName = async () => {
    if (!displayName.trim() || isSavingName) return;
    setIsSavingName(true);
    try {
      await api.updateMe({ name: displayName.trim() });
      setNameSaved(true);
      setTimeout(() => setNameSaved(false), 2000);
    } catch (e) { console.error(e); }
    finally { setIsSavingName(false); }
  };

  const handleDeleteAccount = async () => {
    try { await api.deleteMe(); } catch (e) { /* 204 = ok */ } finally { logout(); }
  };


  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">

      {/* Page Header */}
      <div className="shrink-0 px-8 md:px-12 pt-10 pb-0">
        <h2 className="text-2xl font-bold text-foreground tracking-tight">Settings</h2>
        <p className="text-sm text-muted-foreground mt-1 font-medium">Workspace configuration and preferences</p>

        {/* Top Tab Bar */}
        <div className="flex items-center gap-1 mt-6 border-b border-border">
          {SECTIONS.map((s) => {
            const isActive = activeSection === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setActiveSection(s.id)}
                className={cn(
                  "relative flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors cursor-pointer rounded-t-lg",
                  isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                )}
              >
                <s.icon className={cn("w-3.5 h-3.5 shrink-0", isActive ? "text-primary" : "text-muted-foreground")} />
                <span>{s.label}</span>
                {isActive && (
                  <motion.div
                    layoutId="settings-tab-indicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-full"
                    transition={{ type: "spring", stiffness: 500, damping: 35 }}
                  />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-8 md:px-12 py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeSection}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="max-w-2xl space-y-8"
          >

            {/* LLM Provider */}
            {activeSection === "llm" && (
              <div className="space-y-4">
                {[
                  {
                    val: true,
                    label: "Local Model (Ollama)",
                    model: "qwen3:4b",
                    desc: "Queries stay isolated within your container environment. Best for offline, high-privacy requirements.",
                    badge: "Private"
                  },
                  {
                    val: false,
                    label: "Cloud Model (Gemini)",
                    model: "gemini-3.5-flash",
                    desc: "Leverages powerful cloud reasoning models with grounded citations and multi-step tool use.",
                    badge: "Cloud"
                  }
                ].map((opt) => (
                  <button
                    key={String(opt.val)}
                    type="button"
                    onClick={() => handleToggleOllama(opt.val)}
                    className={cn(
                      "w-full p-5 rounded-2xl border text-left transition-all duration-200 cursor-pointer group",
                      useOllama === opt.val
                        ? "bg-primary/5 border-primary/40 shadow-[0_0_0_1px_var(--primary)]"
                        : "bg-surface-container-lowest border-border hover:border-border/80 hover:bg-surface"
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-bold text-foreground">{opt.label}</span>
                          <span className={cn(
                            "text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-sm",
                            useOllama === opt.val ? "bg-primary/20 text-primary" : "bg-surface-container text-muted-foreground"
                          )}>{opt.badge}</span>
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed max-w-sm">{opt.desc}</p>
                        <p className="text-[10px] font-mono text-muted-foreground/60 mt-2">{opt.model}</p>
                      </div>
                      <div className={cn(
                        "w-5 h-5 rounded-full border-2 shrink-0 mt-0.5 flex items-center justify-center transition-colors",
                        useOllama === opt.val ? "border-primary bg-primary" : "border-border"
                      )}>
                        {useOllama === opt.val && <div className="w-2 h-2 rounded-full bg-white" />}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}

            {/* Appearance */}
            {activeSection === "appearance" && (
              <div className="premium-card p-6 space-y-4">
                <SettingsSection title="Color Theme">
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { id: "light", label: "Light", icon: Sun, desc: "Clean bright workspace" },
                      { id: "dark", label: "Dark", icon: Moon, desc: "Premium dark environment" }
                    ].map((t) => (
                      <button
                        key={t.id}
                        type="button"
                        onClick={() => setTheme(t.id as "light" | "dark")}
                        className={cn(
                          "relative p-4 rounded-xl border text-left cursor-pointer transition-all group",
                          theme === t.id
                            ? "border-primary/50 bg-primary/5"
                            : "border-border bg-surface hover:bg-surface-container-low"
                        )}
                      >
                        {theme === t.id && (
                          <div className="absolute top-2 right-2">
                            <div className="w-4 h-4 rounded-full bg-primary flex items-center justify-center">
                              <Check className="w-2.5 h-2.5 text-white" />
                            </div>
                          </div>
                        )}
                        <t.icon className={cn("w-5 h-5 mb-2", theme === t.id ? "text-primary" : "text-muted-foreground")} />
                        <p className="text-sm font-semibold text-foreground">{t.label}</p>
                        <p className="text-[11px] text-muted-foreground mt-0.5">{t.desc}</p>
                      </button>
                    ))}
                  </div>
                </SettingsSection>
              </div>
            )}

            {/* Developer */}
            {activeSection === "developer" && (
              <div className="premium-card p-6">
                <SettingsSection title="Pipeline Options">
                  <SettingsRow
                    label="Show Grounded Chunks"
                    desc="Expose dense vector retrieval chunks in the inspector panel for debugging."
                  >
                    <Toggle checked={showChunks} onChange={(v) => { setShowChunks(v); localStorage.setItem("setting_show_chunks", String(v)); }} />
                  </SettingsRow>
                  <SettingsRow
                    label="Inspector Auto-Open"
                    desc="Automatically slide open the context inspector when streaming begins."
                    last
                  >
                    <Toggle checked={inspectorOpen} onChange={(v) => { setInspectorOpen(v); localStorage.setItem("setting_inspector_open", String(v)); }} />
                  </SettingsRow>
                </SettingsSection>
              </div>
            )}

            {/* Account */}
            {activeSection === "account" && (
              <div className="space-y-6">
                {/* Avatar + Info strip */}
                <div className="flex items-center gap-4 p-5 premium-card">
                  <div className="w-14 h-14 rounded-2xl bg-primary/20 text-primary flex items-center justify-center font-bold text-xl shrink-0 ring-2 ring-primary/20">
                    {user?.name?.charAt(0)?.toUpperCase() || "U"}
                  </div>
                  <div className="min-w-0">
                    <p className="text-base font-bold text-foreground truncate">{user?.name || "User"}</p>
                    <p className="text-sm text-muted-foreground truncate">{user?.email || "No email"}</p>
                  </div>
                </div>

                {/* Editable fields */}
                <div className="premium-card p-6 space-y-4">
                  <SettingsSection title="Profile Details">
                    <div className="space-y-4 pt-1">
                      <div>
                        <label className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground block mb-1.5">Display Name</label>
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={displayName}
                            onChange={(e) => setDisplayName(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSaveName()}
                            className="flex-1 text-sm font-medium bg-background border border-border rounded-lg px-3 py-2.5 text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 transition-all"
                          />
                          <button
                            onClick={handleSaveName}
                            disabled={isSavingName || !displayName.trim()}
                            className="px-4 py-2.5 rounded-lg bg-surface border border-border hover:bg-surface-container text-foreground transition-all cursor-pointer disabled:opacity-40 flex items-center gap-2 text-sm font-semibold"
                          >
                            {nameSaved ? <Check className="w-4 h-4 text-emerald-500" /> : <Save className="w-4 h-4" />}
                            {nameSaved ? "Saved" : "Save"}
                          </button>
                        </div>
                      </div>
                      <div>
                        <label className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground block mb-1.5">Email (Read-Only)</label>
                        <input
                          type="text"
                          readOnly
                          value={user?.email || ""}
                          className="w-full text-sm font-medium bg-surface border border-border rounded-lg px-3 py-2.5 text-muted-foreground cursor-not-allowed focus:outline-none"
                        />
                      </div>
                    </div>
                  </SettingsSection>
                </div>

                {/* Danger Zone */}
                <div className="p-5 rounded-2xl border border-red-500/20 bg-red-500/[0.03] space-y-3">
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                    <h4 className="text-sm font-bold text-red-500 uppercase tracking-wider">Danger Zone</h4>
                  </div>
                  <button
                    onClick={logout}
                    className="w-full flex items-center justify-center gap-2 bg-surface hover:bg-surface-container border border-border text-foreground font-semibold text-sm py-2.5 rounded-xl transition-all cursor-pointer"
                  >
                    <LogOut className="w-4 h-4" />
                    Log Out
                  </button>
                  <button
                    onClick={() => setShowDeleteAccount(true)}
                    className="w-full flex items-center justify-center gap-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 font-semibold text-sm py-2.5 rounded-xl transition-all border border-red-500/20 cursor-pointer"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Account
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
            )}

            {/* Health */}
            {activeSection === "health" && (
              <div className="space-y-4">
                <div className="premium-card p-6 space-y-4">
                  <SettingsSection title="Backend Connectivity">
                    <div className="pt-1 space-y-3">
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Verify memory caches, uptime, and core RAG pipeline dependencies by querying the server health endpoint directly.
                      </p>
                      <button
                        onClick={() => refetch()}
                        disabled={isCheckingHealth}
                        className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground font-semibold text-sm rounded-lg cursor-pointer hover:bg-primary/90 active:scale-95 transition-all shadow-sm disabled:opacity-60"
                      >
                        <RefreshCw className={cn("w-4 h-4", isCheckingHealth && "animate-spin")} />
                        {isCheckingHealth ? "Querying…" : "Query Node Status"}
                      </button>

                      <AnimatePresence>
                        {rawHealth && !isError && (
                          <motion.div
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-4"
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                              <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">Node Online</span>
                            </div>
                            <pre className="text-[11px] font-mono text-muted-foreground whitespace-pre-wrap">
                              {JSON.stringify(rawHealth, null, 2)}
                            </pre>
                          </motion.div>
                        )}
                        {isError && (
                          <motion.div
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-red-500/5 border border-red-500/20 rounded-xl p-4"
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <span className="w-2 h-2 rounded-full bg-red-500" />
                              <span className="text-xs font-semibold text-red-500">Connection Failed</span>
                            </div>
                            <pre className="text-[11px] font-mono text-red-400 whitespace-pre-wrap">
                              {'{\n  "error": "Could not contact API endpoints."\n}'}
                            </pre>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </SettingsSection>
                </div>
              </div>
            )}

          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
