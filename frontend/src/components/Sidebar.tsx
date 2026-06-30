import { 
  Database, Plus, FolderOpen, MessageSquare, 
  LayoutDashboard, Search, List, Settings, LogOut, Sun, Moon
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "../lib/utils";
import { useStore } from "../store/useStore";
import { UserButton } from "@clerk/react";
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
  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "chats", label: "Chats", icon: MessageSquare, count: conversationsCount },
    { id: "documents", label: "Documents", icon: FolderOpen },
    { id: "settings", label: "Settings", icon: Settings },
  ];

  return (
    <aside
      id="rag-sidebar"
      className="relative hidden md:flex h-full w-[260px] flex-col p-5 space-y-6 border-r border-border select-none overflow-hidden bg-surface-container-lowest dark:bg-transparent"
    >
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
        type="button"
        id="btn-new-research"
        onClick={onNewResearch}
        className="w-full py-2 px-3 bg-surface border border-border hover:bg-surface-container text-on-surface rounded-lg font-medium text-sm flex items-center justify-between cursor-pointer shadow-premium hover:shadow-premium-hover transition-all duration-200"
      >
        <span className="flex items-center space-x-2">
          <Plus className="w-4 h-4 text-muted-foreground" />
          <span>New Chat</span>
        </span>
        <span className="text-[10px] font-mono text-muted-foreground border border-border rounded px-1.5 py-0.5">⌘K</span>
      </button>

      {/* Navigation Links */}
      <nav className="flex-1 space-y-0.5">
        {tabs.map((item) => {
          const isActive = currentTab === item.id || (item.id === "chats" && currentTab === "conversations");
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setCurrentTab(item.id)}
              className={cn(
                "relative w-full flex items-center px-3 py-2 text-sm tracking-tight transition-colors rounded-md cursor-pointer group",
                isActive ? "text-foreground font-semibold" : "text-muted-foreground hover:text-foreground hover:bg-surface-container/50 font-medium"
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-indicator"
                  className="absolute inset-0 bg-surface-container-high rounded-md"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <item.icon className={cn("relative z-10 w-4 h-4 mr-3 transition-colors", isActive ? "text-foreground" : "text-muted-foreground group-hover:text-foreground")} />
              <span className="relative z-10">{item.label}</span>
              {item.count !== undefined && (
                <span className={cn(
                  "relative z-10 ml-auto text-[10px] font-bold px-1.5 py-0.5 rounded-md",
                  isActive ? "bg-background text-foreground shadow-sm" : "bg-transparent text-muted-foreground group-hover:bg-surface-container-high"
                )}>
                  {item.count}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Profile & Logout */}
      <div className="pt-4 space-y-3.5 relative">
        {/* Segmented Theme Switcher */}
        <div className="p-1 flex rounded-xl border border-border bg-surface-container-lowest/80 backdrop-blur-md">
          <button
            type="button"
            onClick={() => setTheme("light")}
            className={cn(
              "flex-1 py-1.5 flex items-center justify-center space-x-1.5 text-xs font-semibold rounded-lg transition-all duration-200 cursor-pointer btn-press",
              theme === "light"
                ? "bg-surface text-foreground shadow-sm border border-border/40"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Sun className="w-3.5 h-3.5" />
            <span>Light</span>
          </button>
          <button
            type="button"
            onClick={() => setTheme("dark")}
            className={cn(
              "flex-1 py-1.5 flex items-center justify-center space-x-1.5 text-xs font-semibold rounded-lg transition-all duration-200 cursor-pointer btn-press",
              theme === "dark"
                ? "bg-surface text-foreground shadow-sm border border-border/40"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Moon className="w-3.5 h-3.5" />
            <span>Dark</span>
          </button>
        </div>

        <div className="flex items-center justify-between px-2 py-1.5 border border-border/40 rounded-xl bg-surface-container-low/40">
          <div className="flex items-center space-x-3 min-w-0 flex-1">
            <UserButton />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-on-surface truncate leading-tight tracking-tight">{user?.name || "User"}</p>
              <span className="text-[10px] text-muted-foreground truncate block">{user?.email || "No email"}</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

