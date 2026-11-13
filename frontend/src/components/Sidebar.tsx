import React, { useState, useRef, useEffect } from "react";
import { 
  Database, Plus, FolderOpen, MessageSquare, 
  LayoutDashboard, Settings, LogOut, Sun, Moon, PanelLeftClose, PanelLeftOpen, Sparkles, Crown, Clock, CreditCard, ShieldCheck, Zap
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "../lib/utils";
import { useStore } from "../store/useStore";
import { api } from "../lib/api";
import { formatResetTime } from "../lib/formatters";
interface SidebarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  onNewResearch: () => void;
  conversationsCount: number;
}

export default function Sidebar({ currentTab, setCurrentTab, onNewResearch, conversationsCount }: SidebarProps) {
  const theme = useStore(s => s.theme);
  const setTheme = useStore(s => s.setTheme);
  const user = useStore(s => s.user);
  const quota = useStore(s => s.quota);
  const setQuota = useStore(s => s.setQuota);
  const isAuthenticated = useStore(s => s.isAuthenticated);
  const logout = () => { api.logout(); };
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isEdgeHovering, setIsEdgeHovering] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isAuthenticated) return;
    let isMounted = true;
    api.getQuota()
      .then((q) => {
        if (isMounted) setQuota(q);
      })
      .catch(() => {
        // Fallback gracefully if quota endpoint fails
      });
    return () => {
      isMounted = false;
    };
  }, [isAuthenticated, setQuota]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const isAdmin = user?.role?.toLowerCase() === "admin";
  const isPro = user?.role?.toLowerCase() === "pro";

  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, shortcut: "D" },
    { id: "chats", label: "Chats", icon: MessageSquare, count: conversationsCount, shortcut: "T" },
    { id: "documents", label: "Documents", icon: FolderOpen, shortcut: "U" },
    { id: "settings", label: "Settings", icon: Settings, shortcut: "S" },
    ...(isAdmin ? [{ id: "admin", label: "Admin", icon: ShieldCheck, shortcut: "A" }] : []),
  ];

  return (
    <aside
      id="rag-sidebar"
      onMouseEnter={() => isCollapsed && setIsEdgeHovering(true)}
      onMouseLeave={() => setIsEdgeHovering(false)}
      className={cn(
        "group/sidebar relative hidden md:flex h-full flex-col border-r border-border select-none overflow-visible bg-surface-container-lowest dark:bg-transparent transition-colors duration-200",
        isCollapsed ? "w-16 px-2 py-4 space-y-5" : "w-[260px] p-5 space-y-6"
      )}
    >
      {/* Brand Logo & Header */}
      <div className={cn("flex items-center mt-2", isCollapsed ? "justify-center" : "space-x-3 px-3")}>
        {isCollapsed ? (
          <button
            type="button"
            onClick={() => setIsCollapsed(false)}
            className={cn(
              "group/sidebar-toggle relative flex h-10 w-10 items-center justify-center rounded-xl border shadow-premium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
              isEdgeHovering
                ? "border-border bg-surface text-muted-foreground"
                : "border-transparent bg-primary text-on-primary-container hover:bg-surface-container-high hover:text-foreground"
            )}
            title="Open sidebar"
            aria-label="Open sidebar"
          >
            <Database
              className={cn(
                "absolute h-4 w-4 transition-opacity transition-transform duration-150 motion-reduce:transition-none group-hover/sidebar-toggle:scale-75 group-hover/sidebar-toggle:opacity-0",
                isEdgeHovering && "scale-75 opacity-0"
              )}
            />
            <PanelLeftOpen
              className={cn(
                "absolute h-4 w-4 scale-75 opacity-0 transition-opacity transition-transform duration-150 motion-reduce:transition-none group-hover/sidebar-toggle:scale-100 group-hover/sidebar-toggle:opacity-100",
                isEdgeHovering && "scale-100 opacity-100"
              )}
            />
          </button>
        ) : (
          <>
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-on-primary-container shadow-premium">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <h1 className="text-base font-bold text-on-surface tracking-tight leading-tight">
                KnowVex
              </h1>
            </div>
            <button
              type="button"
              onClick={() => setIsCollapsed(true)}
              className="ml-auto rounded-md p-1.5 text-muted-foreground transition-colors duration-200 hover:bg-surface-container-high hover:text-foreground active:scale-95 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              title="Collapse sidebar"
              aria-label="Collapse sidebar"
            >
              <PanelLeftClose className="h-5 w-5" />
            </button>
          </>
        )}
      </div>

      {/* Primary CTA Button: New Research */}
      <button
        type="button"
        id="btn-new-research"
        onClick={onNewResearch}
        className={cn(
          "bg-surface border border-border hover:bg-surface-container-high text-on-surface font-medium text-sm flex items-center cursor-pointer shadow-premium transition-colors duration-200 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
          isCollapsed ? "h-10 w-10 justify-center self-center rounded-xl p-0" : "w-full rounded-lg py-2 px-3 justify-between"
        )}
        title="New Chat"
        aria-label="New Chat"
      >
        <span className={cn("flex items-center", !isCollapsed && "space-x-2")}>
          <Plus className="w-4 h-4 text-muted-foreground" />
          {!isCollapsed && <span>New Chat</span>}
        </span>
        {!isCollapsed && <span className="text-[10px] font-mono text-muted-foreground border border-border rounded px-1.5 py-0.5">⌘K</span>}
      </button>

      {/* Navigation Links */}
      <nav className={cn("flex-1 space-y-1", isCollapsed && "pt-2")}>
        {tabs.map((item) => {
          const isActive = currentTab === item.id || (item.id === "chats" && currentTab === "conversations");
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setCurrentTab(item.id)}
              className={cn(
                "relative flex items-center text-sm tracking-tight transition-colors duration-200 cursor-pointer group focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
                isCollapsed ? "mx-auto h-10 w-10 justify-center rounded-xl px-0" : "w-full rounded-md px-3 py-2",
                isActive ? "text-foreground font-semibold" : "text-muted-foreground hover:text-foreground font-medium"
              )}
              title={item.label}
              aria-label={item.label}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-indicator"
                  className={cn("absolute inset-0 bg-surface-container-high", isCollapsed ? "rounded-xl" : "rounded-md")}
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <item.icon className={cn("relative z-10 w-4 h-4 transition-colors", !isCollapsed && "mr-3", isActive ? "text-foreground" : "text-muted-foreground group-hover:text-foreground")} />
              {!isCollapsed && <span className="relative z-10">{item.label}</span>}
              {!isCollapsed && (
                <div className="relative z-10 ml-auto flex items-center justify-end h-5 w-8">
                  {item.shortcut && (
                    <span className={cn(
                      "absolute right-0 text-[10px] font-mono text-muted-foreground border border-border rounded px-1.5 py-0.5 transition-opacity transition-transform duration-200 motion-reduce:transition-none",
                      "opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100"
                    )}>
                      {item.shortcut}
                    </span>
                  )}
                  {item.count !== undefined && (
                    <span className={cn(
                      "absolute right-0 text-[10px] font-semibold px-1.5 py-0.5 rounded-md transition-colors duration-200 tabular-nums",
                      isActive ? "bg-background text-foreground shadow-sm" : "bg-transparent text-muted-foreground group-hover:bg-surface-container-high",
                      item.shortcut && "group-hover:opacity-0"
                    )}>
                      {item.count}
                    </span>
                  )}
                </div>
              )}
              {item.count !== undefined && isCollapsed && item.count > 0 && (
                <span className="absolute right-1.5 top-1.5 z-10 h-2 w-2 rounded-full bg-primary" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Profile & Logout */}
      <div className={cn("relative", isCollapsed ? "mt-auto pt-2 space-y-1" : "pt-4 space-y-1")} ref={menuRef}>
        
        {/* Collapsed mode: theme + logout icons above avatar */}
        {isCollapsed && (
          <>
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl text-muted-foreground hover:bg-surface-container hover:text-foreground transition-colors cursor-pointer"
              title="Toggle Theme"
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            <button
              onClick={logout}
              className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl text-muted-foreground hover:bg-danger-subtle hover:text-danger transition-colors cursor-pointer"
              title="Log Out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </>
        )}

        {/* Dropdown Menu (expanded mode) */}
        {showUserMenu && !isCollapsed && (
          <div className="absolute bottom-full left-2 right-2 mb-2 bg-surface border border-border rounded-xl overflow-hidden z-50">
            {/* Plan Info Strip */}
            <div 
              onClick={() => {
                localStorage.setItem("settings_active_section", "subscription");
                setCurrentTab("settings");
                setShowUserMenu(false);
              }}
              className={cn(
                "px-3.5 py-2.5 border-b flex items-center justify-between cursor-pointer transition-all group relative overflow-hidden",
                isPro
                  ? "bg-gradient-to-r from-amber-500/15 via-yellow-500/10 to-amber-500/5 border-amber-500/30 shadow-[inset_0_0_15px_rgba(245,158,11,0.08)] hover:from-amber-500/20 hover:border-amber-500/40"
                  : "bg-surface-container/50 border-border/60 hover:bg-surface-container"
              )}
            >
              {isPro && (
                <div className="absolute -top-6 -right-6 w-16 h-16 bg-amber-400/15 rounded-full blur-xl pointer-events-none" />
              )}
              <div className="relative z-10">
                <p className={cn(
                  "text-xs font-bold transition-colors flex items-center gap-1.5",
                  isPro ? "text-amber-700 dark:text-amber-300" : "text-foreground group-hover:text-primary"
                )}>
                  {isPro && <Crown className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400 fill-amber-500 dark:fill-amber-400" />}
                  {isPro ? "KnowVex Pro — Active" : "Current plan"}
                </p>
                <p className={cn("text-[11px]", isPro ? "text-amber-800/90 dark:text-amber-200/80 font-medium" : "text-muted-foreground")}>
                  {isAdmin ? "System Administrator" : isPro ? "Priority agents & research tier" : "Community Free"}
                </p>
              </div>
              <div className="relative z-10">
                {isPro ? (
                  <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500 text-neutral-950 inline-flex items-center gap-1 shadow-[0_0_12px_rgba(245,158,11,0.35)] tracking-wide">
                    <Crown className="w-2.5 h-2.5 fill-neutral-950 text-neutral-950" />
                    PRO
                  </span>
                ) : (
                  <span className={cn(
                    "text-[10px] font-semibold px-2 py-0.5 rounded border inline-flex items-center gap-1",
                    isAdmin ? "bg-primary/10 text-primary border-primary/20" : "bg-surface-container text-muted-foreground border-border"
                  )}>
                    {isAdmin ? "Admin" : "Free"}
                  </span>
                )}
              </div>
            </div>

            {/* Daily Quota Progress */}
            <div className="px-3.5 py-2.5 bg-surface/50 border-b border-border/60">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-muted-foreground flex items-center gap-1.5 font-medium">
                  {isPro ? (
                    <Zap className="w-3 h-3 text-amber-500 dark:text-amber-400" />
                  ) : (
                    <Sparkles className="w-3 h-3 text-primary" />
                  )}
                  Daily queries
                </span>
                <span className="tabular-nums font-medium text-foreground text-xs">
                  {quota ? (quota.query_limit === null ? "Unlimited" : `${quota.queries_used} / ${quota.query_limit}`) : "..."}
                </span>
              </div>
              {quota && quota.query_limit !== null ? (
                <div className="w-full h-1.5 bg-surface-container rounded-full overflow-hidden">
                  <div 
                    className={cn(
                      "h-full rounded-full origin-left transition-transform duration-300 motion-reduce:transition-none",
                      quota.queries_used >= quota.query_limit
                        ? "bg-danger"
                        : quota.queries_used / quota.query_limit > 0.8
                        ? "bg-warning"
                        : isPro
                        ? "bg-gradient-to-r from-amber-400 to-yellow-500 shadow-[0_0_8px_rgba(245,158,11,0.4)]"
                        : "bg-primary"
                    )}
                    style={{ transform: `scaleX(${Math.min(1, quota.queries_used / quota.query_limit)})`, width: "100%" }}
                  />
                </div>
              ) : (
                <div className={cn("text-xs font-semibold", isPro ? "text-amber-600 dark:text-amber-400" : "text-primary/80")}>
                  Full unmetered access
                </div>
              )}
            </div>

            {/* Deep Research Quota Progress */}
            <div className="px-3.5 py-2.5 bg-surface/30 border-b border-border/60">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-muted-foreground font-medium flex items-center gap-1.5">
                  {isPro ? (
                    <Sparkles className="w-3 h-3 text-amber-500 dark:text-amber-400" />
                  ) : null}
                  Deep research
                </span>
                <span className="tabular-nums font-medium text-foreground text-xs">
                  {quota ? (quota.research_limit === null ? "Unlimited" : `${quota.research_used ?? 0} / ${quota.research_limit}`) : "..."}
                </span>
              </div>
              {quota && quota.research_limit !== null ? (
                <div className="w-full h-1.5 bg-surface-container rounded-full overflow-hidden">
                  <div 
                    className={cn(
                      "h-full rounded-full origin-left transition-transform duration-300 motion-reduce:transition-none",
                      (quota.research_used ?? 0) >= quota.research_limit
                        ? "bg-danger"
                        : (quota.research_used ?? 0) / quota.research_limit >= 0.8
                        ? "bg-warning"
                        : isPro
                        ? "bg-gradient-to-r from-amber-400 to-yellow-500 shadow-[0_0_8px_rgba(245,158,11,0.4)]"
                        : "bg-success"
                    )}
                    style={{ transform: `scaleX(${Math.min(1, (quota.research_used ?? 0) / quota.research_limit)})`, width: "100%" }}
                  />
                </div>
              ) : (
                <div className="text-xs text-success">
                  Full unmetered research
                </div>
              )}
            </div>

            {/* Daily Reset Time — Pure frontend calculation */}
            <div className="px-3.5 py-1.5 bg-surface/20 border-b border-border/60 flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground flex items-center gap-1">
                <Clock className="w-3 h-3 text-muted-foreground/80" /> Daily reset
              </span>
              <span className="tabular-nums text-muted-foreground">
                {formatResetTime().summary}
              </span>
            </div>

            <div className="p-1.5 space-y-1">
              <button 
                type="button"
                onClick={() => { 
                  localStorage.setItem("settings_active_section", "subscription");
                  setCurrentTab("settings"); 
                  setShowUserMenu(false); 
                }}
                className="w-full flex items-center px-3 py-2 text-xs text-foreground font-medium hover:bg-surface-container rounded-lg transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                <CreditCard className="w-3.5 h-3.5 mr-2 text-muted-foreground" />
                Subscription & Quota
              </button>
              <button 
                type="button"
                onClick={() => { 
                  localStorage.setItem("settings_active_section", "account");
                  setCurrentTab("settings"); 
                  setShowUserMenu(false); 
                }}
                className="w-full flex items-center px-3 py-2 text-xs text-foreground font-medium hover:bg-surface-container rounded-lg transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                <Settings className="w-3.5 h-3.5 mr-2 text-muted-foreground" />
                Account Settings
              </button>
              <button 
                type="button"
                onClick={logout}
                className="w-full flex items-center px-3 py-2 text-xs text-danger font-medium hover:bg-danger-subtle rounded-lg transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-500"
              >
                <LogOut className="w-3.5 h-3.5 mr-2" />
                Log Out
              </button>
            </div>
          </div>
        )}

        <div className={cn("flex items-center py-2", isCollapsed ? "justify-center px-0" : "justify-between px-2")}>
          <button
            type="button"
            onClick={() => setShowUserMenu(!showUserMenu)}
            aria-expanded={showUserMenu}
            aria-label="User menu"
            className={cn(
              "flex items-center rounded-lg hover:bg-surface-container-low transition-colors cursor-pointer border border-transparent hover:border-border/50 px-1 py-1 min-w-0 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
              isCollapsed ? "justify-center" : "space-x-3 flex-1"
            )}
          >
            <div className="relative flex-shrink-0">
              <div className={cn(
                "w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs ring-1 transition-all",
                isPro
                  ? "bg-gradient-to-tr from-amber-500/20 via-yellow-500/25 to-amber-400/20 text-amber-700 dark:text-amber-300 ring-amber-500/40 shadow-[0_0_10px_rgba(245,158,11,0.25)]"
                  : "bg-primary/20 text-primary ring-border"
              )}>
                {user?.name?.charAt(0)?.toUpperCase() || 'U'}
              </div>
              {isCollapsed && (isPro || isAdmin) && (
                <span className={cn(
                  "absolute -top-0.5 -right-0.5 rounded-full ring-2 ring-background",
                  isPro ? "w-2.5 h-2.5 bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.85)]" : "w-2 h-2 bg-warning"
                )} />
              )}
            </div>
            {!isCollapsed && (
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <p className="text-sm font-semibold text-on-surface truncate leading-tight tracking-tight">{user?.name || "User"}</p>
                {isPro ? (
                  <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500 text-neutral-950 inline-flex items-center gap-1 shadow-[0_0_12px_rgba(245,158,11,0.35)] tracking-wide shrink-0">
                    <Crown className="w-2.5 h-2.5 fill-neutral-950 text-neutral-950" />
                    PRO
                  </span>
                ) : (
                  <span className={cn(
                    "text-[10px] font-semibold px-1.5 py-0.5 rounded border inline-flex items-center gap-0.5 shrink-0",
                    isAdmin
                      ? "bg-primary/10 text-primary border-primary/20"
                      : "bg-surface-container-high text-muted-foreground border-border/60"
                  )}>
                    {isAdmin ? "Admin" : "Free"}
                  </span>
                )}
              </div>
              <span className="text-[10px] text-muted-foreground truncate block">{user?.email || "No email"}</span>
            </div>
            )}
          </button>
          {!isCollapsed && <button 
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-1.5 text-muted-foreground hover:bg-surface-container hover:text-foreground rounded-md transition-colors cursor-pointer ml-1 flex-shrink-0"
            title="Toggle Theme"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>}
        </div>
      </div>
    </aside>
  );
}

