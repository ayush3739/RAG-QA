import React from "react";
import Sidebar from "../Sidebar";
import { Menu, Database } from "lucide-react";
import { Toaster } from "sonner";
import NoInternetBanner from "../NoInternetBanner";

interface AppShellProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  isMobileSidebarOpen: boolean;
  setIsMobileSidebarOpen: (open: boolean) => void;
  onNewResearch: () => void;
  conversationsCount: number;
  children: React.ReactNode;
}

export function AppShell({
  currentTab,
  setCurrentTab,
  isMobileSidebarOpen,
  setIsMobileSidebarOpen,
  onNewResearch,
  conversationsCount,
  children,
}: AppShellProps) {
  return (
    <div className="bg-background text-foreground flex h-screen overflow-hidden font-sans relative">
      <Toaster position="top-right" richColors />
      <NoInternetBanner />

      {/* Background Ambient Moving Orbs */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none hidden dark:block">
        <div className="ambient-orb orb-1 bg-primary/5 w-[500px] h-[500px] -top-[200px] -left-[200px]" />
        <div className="ambient-orb orb-2 bg-secondary/5 w-[600px] h-[600px] -bottom-[300px] -right-[100px]" />
        <div className="ambient-orb orb-3 bg-violet-600/5 w-[450px] h-[450px] top-[30%] left-[55%]" />
      </div>

      {/* Mobile Top Navigation rail bar */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-14 bg-background/80 backdrop-blur-md border-b border-border flex items-center justify-between px-6 z-40 select-none">
        <div className="flex items-center space-x-2">
          <Menu
            className="w-6 h-6 text-primary cursor-pointer active:scale-90 transition-transform"
            onClick={() => setIsMobileSidebarOpen(true)}
          />
          <h1 className="text-sm font-black text-primary tracking-tight">KnowVex</h1>
        </div>
        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-primary" />
          <span className="text-[10px] uppercase tracking-wider font-extrabold text-secondary">v4.2.0</span>
        </div>
      </div>

      {isMobileSidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-background/50 backdrop-blur-sm z-40"
          role="button"
          tabIndex={0}
          aria-label="Close sidebar"
          onClick={() => setIsMobileSidebarOpen(false)}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " " ? setIsMobileSidebarOpen(false) : undefined)}
        />
      )}

      {/* Sidebar Rail */}
      <div
        className={`fixed md:relative top-0 bottom-0 left-0 z-50 md:z-auto transition-transform duration-300 md:transform-none ${
          isMobileSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        <Sidebar
          currentTab={currentTab}
          setCurrentTab={(tab) => {
            setCurrentTab(tab);
            setIsMobileSidebarOpen(false);
          }}
          onNewResearch={onNewResearch}
          conversationsCount={conversationsCount}
        />
      </div>

      {/* Content Area */}
      <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden pt-14 md:pt-0 relative z-10">
        {children}
      </main>
    </div>
  );
}
