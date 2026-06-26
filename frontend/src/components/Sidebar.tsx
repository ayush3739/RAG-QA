import React from "react";
import { 
  Database, Plus, FolderOpen, MessageSquare, 
  LayoutDashboard, Search, List, Settings, LogOut, Sun, Moon
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "../lib/utils";
import { useStore } from "../store/useStore";
import AnimatedGradientBackground from "./ui/animated-gradient-background";

interface SidebarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  onNewResearch: () => void;
  conversationsCount: number;
}

export default function Sidebar({
  currentTab,
  setCurrentTab,
  onNewResearch,
  conversationsCount,
}: SidebarProps) {
  const theme = useStore(s => s.theme);
  const setTheme = useStore(s => s.setTheme);
  const tabs = [
    { id: "dashboard", icon: LayoutDashboard, label: "Dashboard" },
    { id: "library", icon: FolderOpen, label: "Library" },
    { id: "conversations", icon: MessageSquare, label: "Conversations", count: conversationsCount },
    { id: "research", icon: Search, label: "Research Workspace" },
    { id: "sessions", icon: List, label: "Sessions Manager" },
    { id: "settings", icon: Settings, label: "Settings" }
  ];

  return (
    <aside
      id="rag-sidebar"
      className="relative hidden md:flex h-full w-[260px] flex-col p-5 space-y-6 border-r border-border select-none overflow-hidden bg-surface-container-lowest dark:bg-transparent"
    >
      {theme === 'dark' && (
        <AnimatedGradientBackground 
          Breathing={true}
          animationSpeed={0.015}
          gradientColors={["#000000", "#040b16", "#0a192f", "#0d1b33", "#15243b", "#0d1b33", "#000000"]}
          gradientStops={[20, 40, 50, 60, 75, 90, 100]}
          containerClassName="opacity-30 pointer-events-none -z-10" 
        />
      )}

      {/* Brand Logo & Header */}
      <div className="flex items-center space-x-3 px-3 mt-2">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-on-primary-container shadow-premium">
          <Database className="w-4 h-4" />
        </div>
        <div>
          <h1 className="text-base font-bold text-on-surface tracking-tight leading-tight">
            DocuMind
          </h1>
        </div>
      </div>

      {/* Primary CTA Button: New Research */}
      <button
        id="btn-new-research"
        onClick={onNewResearch}
        className="w-full py-2 px-3 bg-surface border border-border hover:bg-surface-container text-on-surface rounded-lg font-medium text-sm flex items-center justify-between cursor-pointer shadow-premium hover:shadow-premium-hover transition-all duration-200"
      >
        <span className="flex items-center space-x-2">
          <Plus className="w-4 h-4 text-muted-foreground" />
          <span>New Thread</span>
        </span>
        <span className="text-[10px] font-mono text-muted-foreground border border-border rounded px-1.5 py-0.5">⌘K</span>
      </button>

      {/* Navigation Links */}
      <nav className="flex-1 space-y-0.5">
        {tabs.map((item) => {
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentTab(item.id)}
              className={cn(
                "relative w-full flex items-center px-3 py-2 text-sm tracking-tight transition-colors rounded-md font-medium cursor-pointer group",
                isActive ? "text-on-surface" : "text-secondary hover:text-on-surface"
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-indicator"
                  className="absolute inset-0 bg-surface-container-high rounded-md -z-10"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <item.icon className={cn("w-4 h-4 mr-3 transition-colors", isActive ? "text-on-surface" : "text-muted-foreground group-hover:text-secondary")} />
              <span>{item.label}</span>
              {item.count !== undefined && (
                <span className={cn(
                  "ml-auto text-[10px] font-bold px-1.5 py-0.5 rounded-md",
                  isActive ? "bg-background text-on-surface border border-border/50" : "bg-transparent text-muted-foreground"
                )}>
                  {item.count}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Profile & Logout */}
      <div className="pt-4 space-y-1">
        <div className="flex items-center justify-between px-2 py-2">
          <div className="flex items-center space-x-3 rounded-lg hover:bg-surface-container-low transition-colors cursor-pointer border border-transparent hover:border-border/50 flex-1 px-1 py-1 min-w-0">
            <img
              alt="User Profile"
              className="w-7 h-7 rounded-full object-cover ring-1 ring-border flex-shrink-0"
              referrerPolicy="no-referrer"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuAjrzIpnxNo6CayfUpEnDDcLmOB3Mtue_fYA_eJmVPUgfmSKFmhyE7VPwkQ4S_1H-c5Mj8LAR1J8T3W93BgNGyhIHe2i1iJEdjHvuwAze4wb5GIauHsFJeNn21B_yWbiVTSdeM2MYXF1MzzkCgheDM8FkDlXo2p9XtQgObxu5D6pEdj_jFqG9yd0agz4_ozoLP3eM4eLjRjBqckWPKhGGHtaY2icEhzvLK_oMfefTPEsjhDO0NOUwfDy46qxu5RdOxTHrXDVbdtMmu8"
            />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-on-surface truncate leading-tight tracking-tight">Alex Sterling</p>
              <span className="text-[10px] text-muted-foreground truncate block">alex@deepforest.intel</span>
            </div>
          </div>
          <button 
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-1.5 text-muted-foreground hover:bg-surface-container hover:text-foreground rounded-md transition-colors cursor-pointer ml-1 flex-shrink-0"
            title="Toggle Theme"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </aside>
  );
}

