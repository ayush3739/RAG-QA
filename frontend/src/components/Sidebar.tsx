import React, { useState, useRef, useEffect } from "react";
import { 
  Database, Plus, FolderOpen, MessageSquare, 
  LayoutDashboard, Settings, LogOut, Sun, Moon, PanelLeftClose, PanelLeftOpen
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "../lib/utils";
import { useStore } from "../store/useStore";
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
  const logout = useStore(s => s.logout);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isEdgeHovering, setIsEdgeHovering] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "chats", label: "Chats", icon: MessageSquare, count: conversationsCount },
    { id: "documents", label: "Documents", icon: FolderOpen },
    { id: "settings", label: "Settings", icon: Settings },
  ];

  return (
    <aside
      id="rag-sidebar"
      onMouseEnter={() => isCollapsed && setIsEdgeHovering(true)}
      onMouseLeave={() => setIsEdgeHovering(false)}
      className={cn(
        "group/sidebar relative hidden md:flex h-full flex-col border-r border-border select-none overflow-visible bg-surface-container-lowest dark:bg-transparent transition-[width,padding] duration-200",
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
              "group/sidebar-toggle relative flex h-10 w-10 items-center justify-center rounded-xl border shadow-premium transition-colors",
              isEdgeHovering
                ? "border-border bg-surface text-muted-foreground"
                : "border-transparent bg-primary text-on-primary-container hover:bg-surface-container-high hover:text-foreground"
            )}
            title="Open sidebar"
            aria-label="Open sidebar"
          >
            <Database
              className={cn(
                "absolute h-4 w-4 transition-all duration-150 group-hover/sidebar-toggle:scale-75 group-hover/sidebar-toggle:opacity-0",
                isEdgeHovering && "scale-75 opacity-0"
              )}
            />
            <PanelLeftOpen
              className={cn(
                "absolute h-4 w-4 scale-75 opacity-0 transition-all duration-150 group-hover/sidebar-toggle:scale-100 group-hover/sidebar-toggle:opacity-100",
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
                DocuMind
              </h1>
            </div>
            <button
              type="button"
              onClick={() => setIsCollapsed(true)}
              className="ml-auto rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
              title="Collapse sidebar"
              aria-label="Collapse sidebar"
            >
              <PanelLeftClose className="h-4 w-4" />
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
          "bg-surface border border-border hover:bg-surface-container text-on-surface font-medium text-sm flex items-center cursor-pointer shadow-premium hover:shadow-premium-hover transition-all duration-200",
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
                "relative flex items-center text-sm tracking-tight transition-all duration-200 ease-out cursor-pointer group",
                isCollapsed ? "mx-auto h-10 w-10 justify-center rounded-xl px-0" : "w-full rounded-md px-3 py-2",
                isActive ? "text-foreground font-semibold" : cn("text-muted-foreground hover:text-foreground font-medium", !isCollapsed && "hover:translate-x-1")
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
              {item.count !== undefined && !isCollapsed && (
                <span className={cn(
                  "relative z-10 ml-auto text-[10px] font-bold px-1.5 py-0.5 rounded-md",
                  isActive ? "bg-background text-foreground shadow-sm" : "bg-transparent text-muted-foreground group-hover:bg-surface-container-high"
                )}>
                  {item.count}
                </span>
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
              className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl text-muted-foreground hover:bg-red-500/10 hover:text-red-500 transition-colors cursor-pointer"
              title="Log Out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </>
        )}

        {/* Dropdown Menu (expanded mode) */}
        {showUserMenu && !isCollapsed && (
          <div className="absolute bottom-full left-2 right-2 mb-2 bg-surface border border-border rounded-xl shadow-xl overflow-hidden z-50">
            <div className="p-2">
              <button 
                onClick={logout}
                className="w-full flex items-center px-3 py-2 text-sm text-red-500 font-medium hover:bg-red-500/10 rounded-lg transition-colors cursor-pointer"
              >
                <LogOut className="w-4 h-4 mr-2" />
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
              "flex items-center rounded-lg hover:bg-surface-container-low transition-colors cursor-pointer border border-transparent hover:border-border/50 px-1 py-1 min-w-0 text-left",
              isCollapsed ? "justify-center" : "space-x-3 flex-1"
            )}
          >
            <div className="w-7 h-7 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-xs ring-1 ring-border flex-shrink-0">
              {user?.name?.charAt(0)?.toUpperCase() || 'U'}
            </div>
            {!isCollapsed && (
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-on-surface truncate leading-tight tracking-tight">{user?.name || "User"}</p>
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

