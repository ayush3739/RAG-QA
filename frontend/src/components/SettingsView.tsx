import React, { useState, useEffect } from "react";
import { 
  Cpu, Sliders, User, Shield, RefreshCw, Moon, Sun, LogOut, Trash2, Save, Check,
  Server, Palette, AlertTriangle, Lock, Eye, EyeOff, Clock,
  CreditCard, ArrowUpRight, ChevronRight
} from "lucide-react";
import { toast } from "sonner";
import { ModelParams } from "../types";
import { cn } from "../lib/utils";
import { useQuery } from "@tanstack/react-query";
import { useStore } from "../store/useStore";
import { api } from "../lib/api";
import { formatResetTime } from "../lib/formatters";
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
        "relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
        checked ? "bg-primary" : "bg-surface-container-high"
      )}
    >
      <span
        className={cn(
          "pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow-lg ring-0 transition-transform duration-200 motion-reduce:transition-none",
          checked ? "translate-x-4" : "translate-x-0"
        )}
      />
    </button>
  );
}

// Section heading — sentence case, body face, one step smaller
function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h4 className="text-xs font-semibold text-muted-foreground mb-3">{children}</h4>
  );
}

// Row within a card
function SettingsRow({ label, desc, children, last = false }: { label: string; desc?: string; children: React.ReactNode; last?: boolean }) {
  return (
    <div className={cn("flex items-center justify-between py-3.5", !last && "border-b border-border/50")}>
      <div className="min-w-0 pr-6">
        <span className="text-sm font-semibold text-foreground block">{label}</span>
        {desc && <p className="text-xs text-muted-foreground mt-0.5 leading-snug">{desc}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

// Quota meter — reusable across subscription tab
function QuotaMeter({ label, used, limit, detail }: { label: string; used: number; limit: number | null; detail: string }) {
  const pct = limit ? Math.min(100, (used / limit) * 100) : 100;
  const remaining = limit !== null ? Math.max(0, limit - used) : null;
  const isExhausted = limit !== null && used >= limit;
  const isWarning = limit !== null && !isExhausted && used / limit >= 0.8;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-foreground">{label}</span>
        <span className="tabular-nums font-medium text-foreground">
          {limit === null ? "Unlimited" : `${used} / ${limit}`}
        </span>
      </div>
      <div className="w-full h-1.5 bg-surface-container rounded-full overflow-hidden">
        <div 
          className={cn(
            "h-full rounded-full origin-left transition-transform duration-300 motion-reduce:transition-none",
            isExhausted ? "bg-danger" : isWarning ? "bg-warning" : "bg-primary"
          )}
          style={{ transform: `scaleX(${pct / 100})`, width: "100%" }}
        />
      </div>
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>{detail}</span>
        <span className="tabular-nums">
          {remaining !== null ? `${remaining} remaining` : "Unmetered"}
        </span>
      </div>
    </div>
  );
}

const SECTIONS = [
  { id: "llm", label: "LLM Provider", icon: Cpu },
  { id: "appearance", label: "Appearance", icon: Palette },
  { id: "developer", label: "Developer", icon: Sliders },
  { id: "account", label: "Account", icon: User },
  { id: "subscription", label: "Subscription", icon: CreditCard },
  { id: "health", label: "System Health", icon: Shield },
];

export default function SettingsView({ params, onParamChange }: SettingsViewProps) {
  const [activeSection, setActiveSection] = useState(() => {
    return localStorage.getItem("settings_active_section") || "account";
  });
  const [useOllama, setUseOllama] = useState(() => localStorage.getItem("setting_use_ollama") === "true");
  const [showChunks, setShowChunks] = useState(() => localStorage.getItem("setting_show_chunks") === "true");
  const [inspectorOpen, setInspectorOpen] = useState(() => localStorage.getItem("setting_inspector_open") !== "false");

  const user = useStore(s => s.user);
  const quota = useStore(s => s.quota);
  const documents = useStore(s => s.documents);
  const logout = () => { api.logout(); };
  const theme = useStore(s => s.theme);
  const setTheme = useStore(s => s.setTheme);

  const role = (user?.role || "user").toLowerCase();
  const isAdmin = role === "admin";
  const isPro = role === "pro";
  const isFree = !isAdmin && !isPro;

  const docLimit = isAdmin ? null : isPro ? 50 : 3;
  const pageLimit = isAdmin ? null : isPro ? 300 : 15;
  const resetInfo = formatResetTime(quota?.reset_at);

  const [displayName, setDisplayName] = useState(user?.name || "");
  const [nameSaved, setNameSaved] = useState(false);
  const [isSavingName, setIsSavingName] = useState(false);
  const [showDeleteAccount, setShowDeleteAccount] = useState(false);

  // Password change state
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // Sync activeSection with external changes
  useEffect(() => {
    const handleStorage = () => {
      const saved = localStorage.getItem("settings_active_section");
      if (saved && saved !== activeSection) {
        setActiveSection(saved);
      }
    };
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, [activeSection]);

  const handleTabChange = (sectionId: string) => {
    setActiveSection(sectionId);
    localStorage.setItem("settings_active_section", sectionId);
  };

  const handleChangePassword = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!currentPassword || !newPassword || isChangingPassword) return;
    if (newPassword.length < 8) {
      toast.error("New password must be at least 8 characters.");
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error("New passwords do not match.");
      return;
    }
    setIsChangingPassword(true);
    try {
      await api.changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      toast.success("Password changed successfully.");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err: any) {
      toast.error(err?.message || "Failed to change password. Please check your current password.");
    } finally {
      setIsChangingPassword(false);
    }
  };

  useEffect(() => { if (user?.name) setDisplayName(user.name); }, [user]);

  const handleToggleOllama = (val: boolean) => {
    setUseOllama(val);
    localStorage.setItem("setting_use_ollama", String(val));
    onParamChange({ selectedModel: val ? "qwen3:4b" : "gemini-3.5-flash" });
  };

  const { data: rawHealth, isFetching: isCheckingHealth, refetch, isError } = useQuery({
    queryKey: ['rawHealth'],
    queryFn: async () => {
      return api.getHealth();
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

  // Plan badge styling
  const planBadge = cn(
    "text-[10px] font-semibold px-2 py-0.5 rounded border inline-flex items-center gap-1",
    isAdmin
      ? "bg-primary/8 text-primary border-primary/15"
      : isPro
      ? "bg-primary/8 text-primary border-primary/15"
      : "bg-surface-container text-muted-foreground border-border/60"
  );

  const planLabel = isAdmin ? "Admin" : isPro ? "Pro" : "Free";

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">

      {/* Page Header */}
      <div className="shrink-0 px-8 md:px-12 pt-10 pb-0">
        <h2 className="text-2xl font-bold text-foreground tracking-tight">Settings</h2>
        <p className="text-sm text-muted-foreground mt-1">Workspace configuration and preferences</p>

        {/* Top Tab Bar */}
        <div className="flex items-center gap-1 mt-6 border-b border-border overflow-x-auto">
          {SECTIONS.map((s) => {
            const isActive = activeSection === s.id;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => handleTabChange(s.id)}
                className={cn(
                  "relative flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors cursor-pointer rounded-t-lg shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
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
            className="max-w-2xl md:max-w-3xl space-y-6"
          >

            {/* LLM Provider */}
            {activeSection === "llm" && (
              <div className="space-y-3">
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
                      "w-full p-5 rounded-xl border text-left transition-colors duration-200 cursor-pointer group focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
                      useOllama === opt.val
                        ? "bg-primary/5 border-primary/40 shadow-[0_0_0_1px_var(--primary)]"
                        : "bg-surface-container-lowest border-border hover:border-border/80 hover:bg-surface"
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-semibold text-foreground">{opt.label}</span>
                          <span className={cn(
                            "text-[10px] font-semibold px-1.5 py-0.5 rounded",
                            useOllama === opt.val ? "bg-primary/15 text-primary" : "bg-surface-container text-muted-foreground"
                          )}>{opt.badge}</span>
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed max-w-sm">{opt.desc}</p>
                        <p className="text-[11px] font-mono text-muted-foreground/60 mt-2 tabular-nums">{opt.model}</p>
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
                <SectionHeading>Color theme</SectionHeading>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { id: "light", label: "Light", icon: Sun, desc: "Clean bright workspace" },
                    { id: "dark", label: "Dark", icon: Moon, desc: "Dark environment" }
                  ].map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setTheme(t.id as "light" | "dark")}
                      className={cn(
                        "relative p-4 rounded-xl border text-left cursor-pointer transition-colors group focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
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
              </div>
            )}

            {/* Developer */}
            {activeSection === "developer" && (
              <div className="premium-card p-6">
                <SectionHeading>Pipeline options</SectionHeading>
                <SettingsRow
                  label="Show grounded chunks"
                  desc="Expose dense vector retrieval chunks in the inspector panel for debugging."
                >
                  <Toggle checked={showChunks} onChange={(v) => { setShowChunks(v); localStorage.setItem("setting_show_chunks", String(v)); }} />
                </SettingsRow>
                <SettingsRow
                  label="Inspector auto-open"
                  desc="Automatically slide open the context inspector when streaming begins."
                  last
                >
                  <Toggle checked={inspectorOpen} onChange={(v) => { setInspectorOpen(v); localStorage.setItem("setting_inspector_open", String(v)); }} />
                </SettingsRow>
              </div>
            )}

            {/* Account */}
            {activeSection === "account" && (
              <div className="space-y-6">
                {/* Avatar + Info strip */}
                <div className="flex items-center gap-4 p-5 premium-card">
                  <div className="w-14 h-14 rounded-2xl bg-primary/15 text-primary flex items-center justify-center font-bold text-xl shrink-0 ring-1 ring-primary/15">
                    {user?.name?.charAt(0)?.toUpperCase() || "U"}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="text-base font-bold text-foreground truncate">{user?.name || "User"}</p>
                      <span className={planBadge}>{planLabel}</span>
                    </div>
                    <p className="text-sm text-muted-foreground truncate mt-0.5">{user?.email || "No email"}</p>
                  </div>
                </div>

                {/* Editable fields */}
                <div className="premium-card p-6 space-y-4">
                  <SectionHeading>Profile details</SectionHeading>
                  <div className="space-y-4 pt-1">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground block mb-1.5">Display name</label>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={displayName}
                          onChange={(e) => setDisplayName(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && handleSaveName()}
                          className="flex-1 text-sm bg-background border border-border rounded-lg px-3 py-2.5 text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:border-primary/50 transition-colors"
                        />
                        <button
                          type="button"
                          onClick={handleSaveName}
                          disabled={isSavingName || !displayName.trim()}
                          className="px-4 py-2.5 rounded-lg bg-surface border border-border hover:bg-surface-container text-foreground transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                        >
                          {nameSaved ? <Check className="w-4 h-4 text-success" /> : <Save className="w-4 h-4" />}
                          {nameSaved ? "Saved" : "Save"}
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground block mb-1.5">Email (read-only)</label>
                      <input
                        type="text"
                        readOnly
                        value={user?.email || ""}
                        className="w-full text-sm bg-surface border border-border rounded-lg px-3 py-2.5 text-muted-foreground cursor-not-allowed focus:outline-none"
                      />
                    </div>

                    <div className="pt-2 border-t border-border/50 flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Assigned plan</span>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-foreground">
                          {isAdmin ? "Enterprise Admin" : isPro ? "KnowVex Pro" : "Community Free"}
                        </span>
                        <button
                          type="button"
                          onClick={() => handleTabChange("subscription")}
                          className="text-primary hover:underline font-medium inline-flex items-center gap-0.5 cursor-pointer ml-1 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                        >
                          <span>Manage limits</span>
                          <ChevronRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Security / Password Change */}
                <div className="premium-card p-6 space-y-4">
                  <SectionHeading>Security</SectionHeading>
                  <form onSubmit={handleChangePassword} className="space-y-4 pt-1">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                        Current password
                      </label>
                      <div className="relative">
                        <input
                          type={showPassword ? "text" : "password"}
                          value={currentPassword}
                          onChange={(e) => setCurrentPassword(e.target.value)}
                          placeholder="Enter current password"
                          className="w-full text-sm bg-background border border-border rounded-lg pl-3 pr-10 py-2.5 text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:border-primary/50 transition-colors font-mono"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                        >
                          {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                          New password
                        </label>
                        <input
                          type={showPassword ? "text" : "password"}
                          value={newPassword}
                          onChange={(e) => setNewPassword(e.target.value)}
                          placeholder="At least 8 characters"
                          className="w-full text-sm bg-background border border-border rounded-lg px-3 py-2.5 text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:border-primary/50 transition-colors font-mono"
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                          Confirm new password
                        </label>
                        <input
                          type={showPassword ? "text" : "password"}
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                          placeholder="Re-type new password"
                          className="w-full text-sm bg-background border border-border rounded-lg px-3 py-2.5 text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:border-primary/50 transition-colors font-mono"
                        />
                      </div>
                    </div>

                    <div className="flex justify-end pt-1">
                      <button
                        type="submit"
                        disabled={isChangingPassword || !currentPassword || !newPassword || !confirmPassword}
                        className="px-4 py-2.5 rounded-lg bg-surface border border-border hover:bg-surface-container text-foreground transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                      >
                        <Lock className="w-4 h-4" />
                        {isChangingPassword ? "Updating\u2026" : "Update password"}
                      </button>
                    </div>
                  </form>
                </div>

                {/* Danger Zone */}
                <div className="p-5 rounded-xl border border-danger-subtle bg-danger-subtle/30 space-y-3">
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle className="w-4 h-4 text-danger" />
                    <h4 className="text-sm font-semibold text-danger">Destructive actions</h4>
                  </div>
                  <button
                    type="button"
                    onClick={logout}
                    className="w-full flex items-center justify-center gap-2 bg-surface hover:bg-surface-container border border-border text-foreground font-semibold text-sm py-2.5 rounded-lg transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                  >
                    <LogOut className="w-4 h-4" />
                    Log out this device
                  </button>
                  <button
                    type="button"
                    onClick={() => { api.logoutAll(); }}
                    className="w-full flex items-center justify-center gap-2 bg-surface hover:bg-surface-container border border-border text-foreground font-semibold text-sm py-2.5 rounded-lg transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                  >
                    <Server className="w-4 h-4" />
                    Log out all devices
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowDeleteAccount(true)}
                    className="w-full flex items-center justify-center gap-2 bg-danger-subtle hover:bg-danger-subtle text-danger font-semibold text-sm py-2.5 rounded-lg transition-colors border border-danger-subtle cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete account
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

            {/* Subscription */}
            {activeSection === "subscription" && (
              <div className="space-y-6">
                {/* Active Plan Overview */}
                <div className="premium-card p-6 space-y-5">
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-2.5">
                        <h3 className="text-base font-bold text-foreground tracking-tight">
                          {isAdmin ? "Enterprise Administrator" : isPro ? "KnowVex Pro" : "Community Free"}
                        </h3>
                        <span className={planBadge}>
                          {isAdmin ? "Admin" : "Active"}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed max-w-lg">
                        {isAdmin
                          ? "Unmetered infrastructure access with uncapped agent routing, deep research, and storage."
                          : isPro
                          ? "500 daily agent queries, 25 deep research runs, 50 document slots."
                          : "10 daily agent queries, 1 deep research run, 3 document slots."}
                      </p>
                    </div>

                    {isFree && (
                      <button
                        type="button"
                        onClick={() => {
                          toast.info("Pro plan includes 500 queries/day, 25 deep research runs, and 50 document slots.", {
                            description: "To upgrade, reach out to your team admin or email admin@knowvex.ai"
                          });
                        }}
                        className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-primary bg-primary/10 hover:bg-primary/15 border border-primary/15 rounded-lg transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                      >
                        <span>Upgrade to Pro</span>
                        <ArrowUpRight className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>

                  {/* Cycle metadata — three stats, no cards-in-cards */}
                  <div className="pt-4 border-t border-border flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-8 text-xs">
                    <div>
                      <div className="text-muted-foreground mb-0.5">Allocation cycle</div>
                      <div className="font-medium text-foreground">Daily at midnight UTC</div>
                    </div>
                    <div className="hidden sm:block w-px h-6 bg-border" />
                    <div>
                      <div className="text-muted-foreground mb-0.5">Local reset</div>
                      <div className="font-medium text-foreground">{resetInfo.localTime}</div>
                    </div>
                    <div className="hidden sm:block w-px h-6 bg-border" />
                    <div>
                      <div className="text-muted-foreground mb-0.5">Time remaining</div>
                      <div className="font-medium text-foreground tabular-nums">{resetInfo.timeRemaining}</div>
                    </div>
                  </div>
                </div>

                {/* Metered Resource Usage */}
                <div className="premium-card p-6 space-y-5">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-muted-foreground">
                      Resource usage
                    </h4>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Clock className="w-3.5 h-3.5 text-primary" />
                      <span>{resetInfo.summary}</span>
                    </div>
                  </div>

                  <div className="space-y-4 pt-1">
                    <QuotaMeter
                      label="Daily agent queries"
                      used={quota?.queries_used ?? 0}
                      limit={quota?.query_limit ?? null}
                      detail="Resets daily at 00:00 UTC"
                    />

                    <div className="border-t border-border/50 pt-4">
                      <QuotaMeter
                        label="Deep research runs"
                        used={quota?.research_used ?? 0}
                        limit={quota?.research_limit ?? null}
                        detail="Multi-agent research executions"
                      />
                    </div>

                    <div className="border-t border-border/50 pt-4">
                      <QuotaMeter
                        label="Document library"
                        used={documents.length}
                        limit={docLimit}
                        detail={`Max ${pageLimit ?? "unlimited"} pages per document`}
                      />
                    </div>
                  </div>

                  <div className="pt-3 border-t border-border/50 text-xs text-muted-foreground flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-primary shrink-0" />
                    <span>Next cycle begins at {resetInfo.localTime} ({resetInfo.timeRemaining} remaining). All daily meters reset.</span>
                  </div>
                </div>

                {/* Tier Specifications Table */}
                <div className="premium-card p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-semibold text-muted-foreground">
                        Plan specifications
                      </h4>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Constraints enforced at application runtime
                      </p>
                    </div>
                  </div>

                  <div className="overflow-x-auto -mx-2">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-border text-xs text-muted-foreground">
                          <th className="py-2.5 px-3 font-medium">Resource</th>
                          <th className={cn("py-2.5 px-3 font-medium", isFree && "text-primary")}>
                            Community {isFree && "(current)"}
                          </th>
                          <th className={cn("py-2.5 px-3 font-medium", isPro && "text-primary")}>
                            Pro {isPro && "(current)"}
                          </th>
                          <th className={cn("py-2.5 px-3 font-medium", isAdmin && "text-primary")}>
                            Enterprise {isAdmin && "(current)"}
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/40 text-xs">
                        {[
                          { resource: "Daily agent queries", free: "10 / day", pro: "500 / day", admin: "Unlimited" },
                          { resource: "Deep research runs", free: "1 / day", pro: "25 / day", admin: "Unlimited" },
                          { resource: "Document library", free: "3 documents", pro: "50 documents", admin: "Unlimited" },
                          { resource: "Max pages per document", free: "15 pages", pro: "300 pages", admin: "Unlimited" },
                          { resource: "Web grounding", free: "Tavily", pro: "Tavily", admin: "High-throughput" },
                          { resource: "RAG pipeline", free: "Hybrid vector + BM25", pro: "Hybrid vector + BM25", admin: "Hybrid vector + BM25" },
                          { resource: "Cross-encoder reranking", free: "BGE reranker", pro: "BGE reranker", admin: "BGE reranker" },
                          { resource: "Execution timeout", free: "120 s", pro: "300 s", admin: "Uncapped" },
                        ].map((row) => (
                          <tr key={row.resource} className="hover:bg-surface-container/20 transition-colors">
                            <td className="py-2.5 px-3 font-medium text-foreground">{row.resource}</td>
                            <td className={cn("py-2.5 px-3 tabular-nums", isFree && "font-medium text-foreground")}>{row.free}</td>
                            <td className={cn("py-2.5 px-3 tabular-nums", isPro && "font-medium text-foreground")}>{row.pro}</td>
                            <td className={cn("py-2.5 px-3 tabular-nums", isAdmin && "font-medium text-foreground")}>{row.admin}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Health */}
            {activeSection === "health" && (
              <div className="space-y-4">
                <div className="premium-card p-6 space-y-4">
                  <SectionHeading>Backend connectivity</SectionHeading>
                  <div className="pt-1 space-y-3">
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      Verify memory caches, uptime, and core RAG pipeline dependencies by querying the server health endpoint directly.
                    </p>
                    <button
                      type="button"
                      onClick={() => refetch()}
                      disabled={isCheckingHealth}
                      className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground font-semibold text-sm rounded-lg cursor-pointer hover:bg-primary/90 active:scale-[0.98] transition-colors shadow-sm disabled:opacity-55 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary motion-reduce:transform-none"
                    >
                      <RefreshCw className={cn("w-4 h-4", isCheckingHealth && "animate-spin")} />
                      {isCheckingHealth ? "Querying\u2026" : "Query node status"}
                    </button>

                    <AnimatePresence>
                      {rawHealth && !isError && (
                        <motion.div
                          initial={{ opacity: 0, y: 6 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="bg-success-subtle border border-success-subtle rounded-xl p-4"
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <span className="w-2 h-2 rounded-full bg-success" />
                            <span className="text-xs font-semibold text-success">Node online</span>
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
                          className="bg-danger-subtle border border-danger-subtle rounded-xl p-4"
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <span className="w-2 h-2 rounded-full bg-danger" />
                            <span className="text-xs font-semibold text-danger">Connection failed</span>
                          </div>
                          <pre className="text-[11px] font-mono text-danger whitespace-pre-wrap">
                            {'{\n  "error": "Could not contact API endpoints."\n}'}
                          </pre>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            )}

          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
