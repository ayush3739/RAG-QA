import React, { useState, useEffect } from "react";
import { 
  Upload, MessageSquare, Search, Database, Clock, 
  FileText, ChevronRight, Activity, FileSpreadsheet, Globe,
  CheckCircle2, Layers
} from "lucide-react";
import { SourceDocument, Conversation } from "../types";
import { motion, animate } from "framer-motion";
import { cn } from "../lib/utils";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from "recharts";

interface DashboardViewProps {
  documents: SourceDocument[];
  conversations: Conversation[];
  setCurrentTab: (tab: string) => void;
  onNewResearch: () => void;
  onSelectSession: (id: string) => void;
  onTriggerUploadModal: () => void;
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { type: "spring" as const, stiffness: 300, damping: 24 }
  }
};

const mockActivityData = [
  { name: 'Mon', queries: 12 },
  { name: 'Tue', queries: 19 },
  { name: 'Wed', queries: 15 },
  { name: 'Thu', queries: 28 },
  { name: 'Fri', queries: 22 },
  { name: 'Sat', queries: 9 },
  { name: 'Sun', queries: 35 },
];

function AnimatedCounter({ value, duration = 1.2 }: { value: number; duration?: number }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const controls = animate(0, value, {
      duration,
      ease: "easeOut",
      onUpdate: (val) => setCount(Math.floor(val)),
    });
    return () => controls.stop();
  }, [value, duration]);

  return <span>{count}</span>;
}

interface GlowCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  variants?: any;
}

function GlowCard({ children, className, onClick, variants, ...props }: GlowCardProps) {
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setCoords({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  return (
    <motion.div
      variants={variants}
      whileHover={{ y: -6, scale: 1.015 }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onClick}
      className={cn(
        "group relative p-6 premium-card cursor-pointer overflow-hidden transition-all duration-300 hover:border-primary/45 hover:shadow-[0_20px_50px_rgba(0,0,0,0.4)]",
        className
      )}
      {...props}
    >
      {/* Vercel Mouse-Follow Glow */}
      <div
        className="absolute inset-0 pointer-events-none transition-opacity duration-300 z-0"
        style={{
          opacity: isHovered ? 1 : 0,
          background: `radial-gradient(circle 150px at ${coords.x}px ${coords.y}px, rgba(123, 108, 246, 0.15), transparent 80%)`,
        }}
      />
      <div className="relative z-10 w-full h-full flex flex-col">
        {children}
      </div>
    </motion.div>
  );
}

function StatCard({ title, value, icon: Icon, colorClass, variants }: { title: string; value: number; icon: React.ComponentType<any>; colorClass: string; variants?: any }) {
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setCoords({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  return (
    <motion.div
      variants={variants}
      whileHover={{ y: -3, scale: 1.01 }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="relative p-5 premium-card overflow-hidden transition-all duration-300 hover:border-primary/30 shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
    >
      {/* Mouse Follow Glow */}
      <div
        className="absolute inset-0 pointer-events-none transition-opacity duration-300 z-0"
        style={{
          opacity: isHovered ? 1 : 0,
          background: `radial-gradient(circle 120px at ${coords.x}px ${coords.y}px, rgba(123, 108, 246, 0.1), transparent 80%)`,
        }}
      />
      <div className="relative z-10 flex items-center justify-between">
        <div className="space-y-1">
          <p className="text-[10px] font-extrabold text-muted-foreground uppercase tracking-widest">{title}</p>
          <p className="text-3xl font-extrabold text-foreground tracking-tight">
            <AnimatedCounter value={value} />
          </p>
        </div>
        <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center border border-border bg-surface-container/50", colorClass)}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </motion.div>
  );
}

export default function DashboardView({
  documents,
  conversations,
  setCurrentTab,
  onNewResearch,
  onSelectSession,
  onTriggerUploadModal,
}: DashboardViewProps) {
  const { data: healthData, isError } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      return api.getHealth();
    },
    staleTime: Infinity,
    gcTime: Infinity,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
  });

  const healthStatus = isError ? "offline" : (healthData ? "online" : "offline");
  const healthTime = healthData?.time ? new Date(healthData.time).toLocaleTimeString() : new Date().toLocaleTimeString();

  const totalChunks = documents.reduce((sum, doc) => sum + (doc.chunkCount || 18), 0);
  const activeDocsCount = documents.filter(d => d.active).length;

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-12 py-10 space-y-10 bg-transparent selection:bg-white/20 selection:text-white">
      
      {/* Title */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }} 
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl md:text-3xl font-bold text-foreground tracking-tight">
          DocuMind Workspace
        </h2>
        <p className="text-sm text-muted-foreground mt-1.5 max-w-2xl font-medium">
          Access your RAG document indexing pipeline and analyze curated files with the agentic tool-routing layer.
        </p>
      </motion.div>

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-8"
      >
        {/* Stats Grid Row (Four Columns) */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Total Documents"
            value={documents.length}
            icon={FileText}
            colorClass="text-amber-500 border-amber-500/10"
            variants={itemVariants}
          />
          <StatCard
            title="Indexed Chunks"
            value={totalChunks}
            icon={Database}
            colorClass="text-blue-500 border-blue-500/10"
            variants={itemVariants}
          />
          <StatCard
            title="Active Sessions"
            value={conversations.length}
            icon={MessageSquare}
            colorClass="text-purple-500 border-purple-500/10"
            variants={itemVariants}
          />
          <StatCard
            title="Knowledge Coverage"
            value={activeDocsCount}
            icon={CheckCircle2}
            colorClass="text-emerald-500 border-emerald-500/10"
            variants={itemVariants}
          />
        </div>

        {/* Row 1: Quick Action Cards (Three Columns) */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <GlowCard
            variants={itemVariants}
            onClick={onTriggerUploadModal}
          >
            <div className="w-10 h-10 rounded-xl bg-surface-container border border-border flex items-center justify-center text-foreground mb-4 group-hover:scale-105 group-hover:text-primary group-hover:border-primary/30 transition-all duration-300">
              <Upload className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-foreground tracking-tight mb-1.5 group-hover:text-primary transition-colors">Upload Documents</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Ingest PDFs, spreadsheets, or raw markdown text files to expand your active vector database.
            </p>
          </GlowCard>

          <GlowCard
            variants={itemVariants}
            onClick={onNewResearch}
          >
            <div className="w-10 h-10 rounded-xl bg-surface-container border border-border flex items-center justify-center text-foreground mb-4 group-hover:scale-105 group-hover:text-primary group-hover:border-primary/30 transition-all duration-300">
              <MessageSquare className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-foreground tracking-tight mb-1.5 group-hover:text-primary transition-colors">New Chat Thread</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Launch a streaming conversation, attach specific materials, and ask questions with citations.
            </p>
          </GlowCard>

          <GlowCard
            variants={itemVariants}
            onClick={onNewResearch}
          >
            <div className="w-10 h-10 rounded-xl bg-surface-container border border-border flex items-center justify-center text-foreground mb-4 group-hover:scale-105 group-hover:text-primary group-hover:border-primary/30 transition-all duration-300">
              <Search className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-foreground tracking-tight mb-1.5 group-hover:text-primary transition-colors">Structured Research</h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Execute professional one-shot report generation with multi-source validation and web-search backups.
            </p>
          </GlowCard>
        </div>

        {/* Row 2: Recent Sessions & Recent Documents (Bento Grid) */}
        <div className="grid grid-cols-12 gap-6">
          
          {/* Recent Sessions */}
          <motion.div variants={itemVariants} className="col-span-12 md:col-span-8 flex flex-col p-6 premium-card">
            <div className="flex items-center justify-between mb-5">
              <h4 className="font-semibold text-sm text-foreground flex items-center tracking-tight">
                <Clock className="w-4 h-4 mr-2 text-muted-foreground" />
                Recent Threads
              </h4>
              <button 
                onClick={() => setCurrentTab("chats")}
                className="text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                View all
              </button>
            </div>

            <div className="space-y-2 flex-1">
              {conversations.slice(0, 4).map((conv) => (
                <button
                  key={conv.id}
                  type="button"
                  onClick={() => onSelectSession(conv.id)}
                  className="w-full px-4 py-3 bg-surface-container-high/30 dark:bg-black/40 border border-black/5 dark:border-white/5 hover:border-border rounded-xl transition-all duration-200 cursor-pointer flex items-center justify-between group text-left"
                >
                  <div className="min-w-0 flex-1 pr-4">
                    <h5 className="text-sm font-medium text-foreground truncate">{conv.title}</h5>
                    <p className="text-[12px] text-muted-foreground truncate mt-0.5">
                      {conv.messages[conv.messages.length - 1]?.text || "No queries executed yet."}
                    </p>
                  </div>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-[10px] font-mono text-muted-foreground">
                      {conv.messages.length} msgs
                    </span>
                    <ChevronRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </button>
              ))}
              {conversations.length === 0 && (
                <div className="h-full flex items-center justify-center p-4">
                  <p className="text-sm text-muted-foreground italic">No active sessions found.</p>
                </div>
              )}
            </div>
          </motion.div>

          {/* Recent Documents */}
          <motion.div variants={itemVariants} className="col-span-12 md:col-span-4 flex flex-col p-6 premium-card">
            <div className="flex items-center justify-between mb-5">
              <h4 className="font-semibold text-sm text-foreground flex items-center tracking-tight">
                <Database className="w-4 h-4 mr-2 text-muted-foreground" />
                Library
              </h4>
              <button 
                onClick={() => setCurrentTab("documents")}
                className="text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
              >
                Manage
              </button>
            </div>

            <div className="space-y-2 flex-1">
              {documents.slice(0, 4).map((doc) => (
                <div
                  key={doc.id}
                  className="px-4 py-3 bg-surface-container-high/30 dark:bg-black/40 border border-black/5 dark:border-white/5 hover:border-border rounded-xl flex items-center justify-between transition-colors"
                >
                  <div className="flex items-center space-x-3 min-w-0">
                    <div className={cn(
                      "p-1.5 rounded-md border",
                      doc.type === "spreadsheet" ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-500" :
                      doc.type === "link" ? "bg-sky-500/10 border-sky-500/20 text-sky-500" :
                      doc.type === "pdf" ? "bg-rose-500/10 border-rose-500/20 text-rose-500" :
                      "bg-amber-500/10 border-amber-500/20 text-amber-500"
                    )}>
                      {doc.type === "spreadsheet" && <FileSpreadsheet className="w-3.5 h-3.5" />}
                      {doc.type === "link" && <Globe className="w-3.5 h-3.5" />}
                      {(doc.type === "pdf" || doc.type === "doc") && <FileText className="w-3.5 h-3.5" />}
                    </div>
                    <p className="text-sm font-medium text-foreground truncate">{doc.name}</p>
                  </div>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-[10px] text-muted-foreground font-mono">{doc.size}</span>
                    <span className={cn("w-2 h-2 rounded-full shadow-sm", doc.active ? "bg-emerald-500 shadow-emerald-500/50" : "bg-border")} />
                  </div>
                </div>
              ))}
              {documents.length === 0 && (
                <div className="h-full flex items-center justify-center p-4">
                  <p className="text-sm text-muted-foreground italic">No materials imported.</p>
                </div>
              )}
            </div>
          </motion.div>

        </div>

        {/* Row 3: Activity Chart */}
        <div className="grid grid-cols-1 gap-4">
          <motion.div 
            variants={itemVariants} 
            whileHover={{ y: -4, scale: 1.005 }}
            className="flex flex-col p-6 premium-card h-[320px] overflow-hidden hover:border-primary/30 hover:shadow-[0_0_30px_rgba(123,108,246,0.08)] transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-6">
              <h4 className="font-semibold text-sm text-foreground flex items-center tracking-tight">
                <Activity className="w-4 h-4 mr-2 text-muted-foreground animate-pulse" />
                Workspace Activity (Queries)
              </h4>
            </div>
            <div className="flex-1 w-full min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={mockActivityData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorQueries" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#7B6CF6" stopOpacity={0.25}/>
                      <stop offset="100%" stopColor="#7B6CF6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-muted-foreground)' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--color-muted-foreground)' }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--color-surface)', borderColor: 'var(--color-border)', borderRadius: '8px', fontSize: '12px' }}
                    itemStyle={{ color: 'var(--color-foreground)', fontWeight: 600 }}
                  />
                  <Area type="monotone" dataKey="queries" stroke="var(--color-primary)" strokeWidth={2.5} fillOpacity={1} fill="url(#colorQueries)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Bottom Health Strip */}
      <motion.div 
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        className="pt-6 flex items-center justify-between text-xs text-muted-foreground"
      >
        <div className="flex items-center space-x-2">
          <Activity className="w-3.5 h-3.5" />
          <span>Core RAG Node: <strong className="font-medium">monolith-v4.2.0</strong></span>
        </div>
        <div className="flex items-center space-x-2 bg-surface-container px-3 py-1.5 rounded-md border border-border">
          <span className={cn("w-1.5 h-1.5 rounded-full pulse-online", healthStatus === "online" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]" : "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]")} />
          <span className="font-medium">
            {healthStatus === "online" ? <>Online (Verified <span className="font-mono text-[10px] ml-1">{healthTime}</span>)</> : "Disconnected"}
          </span>
        </div>
      </motion.div>
    </div>
  );
}
