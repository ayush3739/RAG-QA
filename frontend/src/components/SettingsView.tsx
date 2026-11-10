import React, { useState, useEffect } from "react";
import { 
  Cpu, Sliders, User, Shield, RefreshCw, Moon, Sun, LogOut, Trash2, Save, Check,
  Server, Palette, AlertTriangle, Crown, Sparkles, Lock, Eye, EyeOff, Clock,
  CreditCard, ArrowUpRight, CheckCircle2, FileText, Zap, ChevronRight
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
                onClick={() => handleTabChange(s.id)}
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
            className="max-w-2xl md:max-w-3xl space-y-8"
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
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="text-base font-bold text-foreground truncate">{user?.name || "User"}</p>
                      <span className={cn(
                        "text-[10px] font-mono font-extrabold uppercase px-2 py-0.5 rounded tracking-wider flex items-center gap-1",
                        user?.role?.toLowerCase() === "pro"
                          ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                          : user?.role?.toLowerCase() === "admin"
                          ? "bg-purple-500/15 text-purple-400 border border-purple-500/30"
                          : "bg-surface-container-high text-muted-foreground border border-border/60"
                      )}>
                        {user?.role?.toLowerCase() === "pro" && <Crown className="w-3 h-3 text-amber-400" />}
                        {user?.role?.toLowerCase() === "admin" ? "ADMIN" : user?.role?.toLowerCase() === "pro" ? "PRO PLAN" : "FREE PLAN"}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground truncate mt-0.5">{user?.email || "No email"}</p>
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

                      <div className="pt-2 border-t border-border/50 flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Assigned Plan</span>
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-semibold text-foreground">
                            {isAdmin ? "Enterprise Admin" : isPro ? "KnowVex Pro" : "Community Free"}
                          </span>
                          <button
                            type="button"
                            onClick={() => handleTabChange("subscription")}
                            className="text-primary hover:underline font-medium inline-flex items-center gap-0.5 cursor-pointer ml-1"
                          >
                            <span>Manage Plan & Limits</span>
                            <ChevronRight className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </SettingsSection>
                </div>

                {/* Security / Password Change */}
                <div className="premium-card p-6 space-y-4">
                  <SettingsSection title="Security & Authentication">
                    <form onSubmit={handleChangePassword} className="space-y-4 pt-1">
                      <div>
                        <label className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground block mb-1.5">
                          Current Password
                        </label>
                        <div className="relative">
                          <input
                            type={showPassword ? "text" : "password"}
                            value={currentPassword}
                            onChange={(e) => setCurrentPassword(e.target.value)}
                            placeholder="Enter current password"
                            className="w-full text-sm font-medium bg-background border border-border rounded-lg pl-3 pr-10 py-2.5 text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 transition-all font-mono"
                          />
                          <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground cursor-pointer"
                          >
                            {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <label className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground block mb-1.5">
                            New Password
                          </label>
                          <input
                            type={showPassword ? "text" : "password"}
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            placeholder="At least 8 characters"
                            className="w-full text-sm font-medium bg-background border border-border rounded-lg px-3 py-2.5 text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 transition-all font-mono"
                          />
                        </div>
                        <div>
                          <label className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground block mb-1.5">
                            Confirm New Password
                          </label>
                          <input
                            type={showPassword ? "text" : "password"}
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            placeholder="Re-type new password"
                            className="w-full text-sm font-medium bg-background border border-border rounded-lg px-3 py-2.5 text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 transition-all font-mono"
                          />
                        </div>
                      </div>

                      <div className="flex justify-end pt-1">
                        <button
                          type="submit"
                          disabled={isChangingPassword || !currentPassword || !newPassword || !confirmPassword}
                          className="px-4 py-2.5 rounded-lg bg-surface border border-border hover:bg-surface-container text-foreground transition-all cursor-pointer disabled:opacity-40 flex items-center gap-2 text-sm font-semibold"
                        >
                          <Lock className="w-4 h-4" />
                          {isChangingPassword ? "Updating..." : "Update Password"}
                        </button>
                      </div>
                    </form>
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
                    Log Out Current Device
                  </button>
                  <button
                    onClick={() => { api.logoutAll(); }}
                    className="w-full flex items-center justify-center gap-2 bg-surface hover:bg-surface-container border border-border text-foreground font-semibold text-sm py-2.5 rounded-xl transition-all cursor-pointer"
                  >
                    <Server className="w-4 h-4" />
                    Log Out All Devices
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

            {/* Subscription */}
            {activeSection === "subscription" && (
              <div className="space-y-6">
                {/* Active Plan Overview */}
                <div className="premium-card p-6 space-y-5">
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2.5">
                        <h3 className="text-base font-bold text-foreground tracking-tight">
                          {isAdmin ? "Enterprise Administrator" : isPro ? "KnowVex Pro Tier" : "Community Free Tier"}
                        </h3>
                        <span className={cn(
                          "text-[10px] font-mono font-bold tracking-wider px-2 py-0.5 rounded border uppercase",
                          isAdmin 
                            ? "bg-purple-500/10 text-purple-400 border-purple-500/20"
                            : isPro 
                            ? "bg-primary/10 text-primary border-primary/20" 
                            : "bg-surface-container text-muted-foreground border-border"
                        )}>
                          {isAdmin ? "SYSTEM ADMIN" : "ACTIVE"}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed max-w-xl">
                        {isAdmin
                          ? "Unmetered infrastructure access with uncapped agent routing, deep research, and storage."
                          : isPro
                          ? "Professional tier with 500 daily agent queries, 25 deep research executions, and 50 document slots."
                          : "Standard metered tier with 10 daily agent queries, 1 deep research execution, and 3 document storage slots."}
                      </p>
                    </div>

                    {isFree && (
                      <button
                        type="button"
                        onClick={() => {
                          toast.info("Pro plan includes 500 queries/day, 25 deep research runs, and 50 document slots.", {
                            description: "To upgrade your organization account, reach out to your team admin or email admin@knowvex.ai"
                          });
                        }}
                        className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-primary bg-primary/10 hover:bg-primary/20 border border-primary/20 rounded-lg transition-colors cursor-pointer"
                      >
                        <span>Upgrade to Pro</span>
                        <ArrowUpRight className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>

                  {/* Telemetry metadata strip */}
                  <div className="pt-4 border-t border-border grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                    <div className="p-3 rounded-lg bg-surface-container/30 border border-border/50">
                      <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Allocation Cycle</div>
                      <div className="text-xs font-semibold text-foreground mt-0.5">Daily Midnight UTC</div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-container/30 border border-border/50">
                      <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Local Reset Window</div>
                      <div className="text-xs font-semibold text-foreground mt-0.5">{resetInfo.localTime}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-container/30 border border-border/50">
                      <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Cycle Status</div>
                      <div className="text-xs font-mono font-semibold text-foreground mt-0.5">{resetInfo.timeRemaining} remaining</div>
                    </div>
                  </div>
                </div>

                {/* Metered Resource Usage */}
                <div className="premium-card p-6 space-y-5">
                  <div className="flex items-center justify-between">
                    <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                      Metered Resource Usage
                    </h4>
                    <div className="flex items-center gap-1.5 text-[11px] font-mono text-muted-foreground">
                      <Clock className="w-3.5 h-3.5 text-primary" />
                      <span>{resetInfo.summary}</span>
                    </div>
                  </div>

                  <div className="space-y-4 pt-1">
                    {/* Daily Queries */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-foreground">Daily Agent Queries</span>
                        <span className="font-mono font-medium text-foreground">
                          {quota ? (quota.query_limit === null ? "Unlimited" : `${quota.queries_used} / ${quota.query_limit}`) : "..."}
                        </span>
                      </div>
                      <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                        <div 
                          className={cn(
                            "h-full rounded-full transition-all duration-300",
                            quota && quota.query_limit !== null && quota.queries_used >= quota.query_limit
                              ? "bg-red-500"
                              : quota && quota.query_limit !== null && quota.queries_used / quota.query_limit >= 0.8
                              ? "bg-amber-500"
                              : "bg-primary"
                          )}
                          style={{ 
                            width: `${quota && quota.query_limit ? Math.min(100, (quota.queries_used / quota.query_limit) * 100) : (quota?.query_limit === null ? 100 : 0)}%` 
                          }}
                        />
                      </div>
                      <div className="flex justify-between text-[11px] text-muted-foreground font-mono">
                        <span>Resets daily at 00:00 UTC</span>
                        <span>
                          {quota?.query_limit !== null && quota
                            ? `${Math.max(0, (quota.query_limit ?? 0) - quota.queries_used)} queries remaining`
                            : "Unmetered inference"}
                        </span>
                      </div>
                    </div>

                    {/* Deep Research Executions */}
                    <div className="space-y-2 pt-3 border-t border-border/50">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-foreground">Deep Research Runs</span>
                        <span className="font-mono font-medium text-foreground">
                          {quota ? (quota.research_limit === null ? "Unlimited" : `${quota.research_used ?? 0} / ${quota.research_limit}`) : "..."}
                        </span>
                      </div>
                      <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                        <div 
                          className={cn(
                            "h-full rounded-full transition-all duration-300",
                            quota && quota.research_limit !== null && (quota.research_used ?? 0) >= quota.research_limit
                              ? "bg-red-500"
                              : quota && quota.research_limit !== null && (quota.research_used ?? 0) / quota.research_limit >= 0.8
                              ? "bg-amber-500"
                              : "bg-primary"
                          )}
                          style={{ 
                            width: `${quota && quota.research_limit ? Math.min(100, ((quota.research_used ?? 0) / quota.research_limit) * 100) : (quota?.research_limit === null ? 100 : 0)}%` 
                          }}
                        />
                      </div>
                      <div className="flex justify-between text-[11px] text-muted-foreground font-mono">
                        <span>Autonomous multi-agent research runs</span>
                        <span>
                          {quota?.research_limit !== null && quota
                            ? `${Math.max(0, (quota.research_limit ?? 0) - (quota.research_used ?? 0))} runs remaining`
                            : "Unmetered runs"}
                        </span>
                      </div>
                    </div>

                    {/* Document Library Slots */}
                    <div className="space-y-2 pt-3 border-t border-border/50">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-foreground">Document Library Capacity</span>
                        <span className="font-mono font-medium text-foreground">
                          {docLimit === null ? `${documents.length} / Unlimited` : `${documents.length} / ${docLimit} files`}
                        </span>
                      </div>
                      <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                        <div 
                          className={cn(
                            "h-full rounded-full transition-all duration-300",
                            docLimit !== null && documents.length >= docLimit
                              ? "bg-red-500"
                              : docLimit !== null && documents.length / docLimit >= 0.8
                              ? "bg-amber-500"
                              : "bg-primary"
                          )}
                          style={{ 
                            width: `${docLimit ? Math.min(100, (documents.length / docLimit) * 100) : 100}%` 
                          }}
                        />
                      </div>
                      <div className="flex justify-between text-[11px] text-muted-foreground font-mono">
                        <span>Max {pageLimit ?? "unlimited"} pages per document</span>
                        <span>
                          {docLimit !== null
                            ? `${Math.max(0, docLimit - documents.length)} file slots available`
                            : "Uncapped document store"}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-border/50 text-[11px] text-muted-foreground font-mono flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-primary shrink-0" />
                    <span>Next allocation cycle begins at {resetInfo.localTime} ({resetInfo.timeRemaining} remaining). All daily meters reset to zero.</span>
                  </div>
                </div>

                {/* Tier Specifications Table */}
                <div className="premium-card p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                        Plan Resource Specifications
                      </h4>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Deterministic constraints enforced across application runtime
                      </p>
                    </div>
                    <span className="text-[11px] font-mono text-muted-foreground">
                      RBAC Engine
                    </span>
                  </div>

                  <div className="overflow-x-auto -mx-2">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-border text-[11px] text-muted-foreground uppercase tracking-wider font-mono">
                          <th className="py-2.5 px-3 font-semibold">Resource / Capability</th>
                          <th className={cn("py-2.5 px-3 font-semibold", isFree && "text-primary")}>
                            Community {isFree && "(Current)"}
                          </th>
                          <th className={cn("py-2.5 px-3 font-semibold", isPro && "text-primary")}>
                            KnowVex Pro {isPro && "(Current)"}
                          </th>
                          <th className={cn("py-2.5 px-3 font-semibold", isAdmin && "text-primary")}>
                            Enterprise {isAdmin && "(Current)"}
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/40 font-mono text-[11px]">
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">Daily Agent Inferences</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>10 / day</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>500 / day</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>Unlimited</td>
                        </tr>
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">Deep Research Runs</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>1 / day</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>25 / day</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>Unlimited</td>
                        </tr>
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">Document Library Capacity</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>3 documents</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>50 documents</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>Unlimited</td>
                        </tr>
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">Max Pages Per Document</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>15 pages</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>300 pages</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>Unlimited</td>
                        </tr>
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">Real-Time Web Grounding</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>Standard (Tavily)</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>Standard (Tavily)</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>High-Throughput</td>
                        </tr>
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">RAG Retrieval Pipeline</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>Hybrid Vector + BM25</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>Hybrid Vector + BM25</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>Hybrid Vector + BM25</td>
                        </tr>
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">Cross-Encoder Reranking</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>CrossEncoder (bge)</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>CrossEncoder (bge)</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>CrossEncoder (bge)</td>
                        </tr>
                        <tr className="hover:bg-surface-container/20 transition-colors">
                          <td className="py-2.5 px-3 font-sans text-foreground font-medium text-xs">Execution Timeout</td>
                          <td className={cn("py-2.5 px-3", isFree && "font-semibold text-foreground")}>120s max</td>
                          <td className={cn("py-2.5 px-3", isPro && "font-semibold text-foreground")}>300s max</td>
                          <td className={cn("py-2.5 px-3", isAdmin && "font-semibold text-foreground")}>Uncapped</td>
                        </tr>
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
