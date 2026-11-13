import React, { useState, useEffect } from "react";
import { 
  Upload, MessageSquare, Database, Clock, 
  FileText, ChevronRight, Activity, FileSpreadsheet, Globe,
  CheckCircle2, ArrowUpRight, Sparkles, AlertCircle
} from "lucide-react";
import { SourceDocument, Conversation } from "../types";
import { motion, animate } from "framer-motion";
import { cn } from "../lib/utils";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
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
    transition: { staggerChildren: 0.05 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { type: "spring" as const, stiffness: 350, damping: 28 }
  }
};

function AnimatedCounter({ value, duration = 1.0 }: { value: number; duration?: number }) {
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

function DocTypeBadgeIcon({ type }: { type: string }) {
  if (type === "spreadsheet") return <FileSpreadsheet className="w-3.5 h-3.5" />;
  if (type === "link") return <Globe className="w-3.5 h-3.5" />;
  if (type === "pdf" || type === "doc") return <FileText className="w-3.5 h-3.5" />;
  return <Database className="w-3.5 h-3.5" />;
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
    queryFn: async () => api.getHealth(),
    staleTime: Infinity,
    gcTime: Infinity,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  const { data: activityResponse } = useQuery({
    queryKey: ['user-activity'],
    queryFn: () => api.getActivity(7),
    staleTime: 10 * 60 * 1000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  const chartData = activityResponse?.activity && activityResponse.activity.length > 0
    ? activityResponse.activity
    : [
        { name: 'Mon', queries: 0, research: 0 },
        { name: 'Tue', queries: 0, research: 0 },
        { name: 'Wed', queries: 0, research: 0 },
        { name: 'Thu', queries: 0, research: 0 },
        { name: 'Fri', queries: 0, research: 0 },
        { name: 'Sat', queries: 0, research: 0 },
        { name: 'Sun', queries: 0, research: 0 },
      ];

  const totalQueriesPast7Days = activityResponse?.activity 
    ? activityResponse.activity.reduce((acc: number, curr: any) => acc + (curr.queries || 0), 0)
    : 0;

  const healthStatus = isError ? "offline" : (healthData ? "online" : "offline");
  const healthTime = healthData?.time ? new Date(healthData.time).toLocaleTimeString() : new Date().toLocaleTimeString();

  const totalChunks = documents.reduce((sum, doc) => sum + (doc.chunkCount || 0), 0);
  const activeDocsCount = documents.filter(d => d.active).length;

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 bg-background">
      {/* Workspace Command Bar & Header */}
      <motion.div 
        initial={{ opacity: 0, y: -8 }} 
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="mb-8 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-border/70 pb-6"
      >
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-foreground tracking-tight">
              KnowVex Workspace
            </h1>
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-surface-container border border-border text-xs text-muted-foreground">
              <span className={cn("w-1.5 h-1.5 rounded-full", healthStatus === "online" ? "bg-success" : "bg-danger")} />
              <span>{healthStatus === "online" ? "Core Node Live" : "Core Node Offline"}</span>
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1 font-medium">
            Active neural RAG retrieval engine on <strong className="text-foreground font-semibold">knowvex-core-v2.0</strong> ({totalQueriesPast7Days} queries past 7 days)
          </p>
        </div>

        {/* Quick Launch Buttons */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <button
            type="button"
            onClick={onTriggerUploadModal}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-surface border border-border text-foreground hover:bg-surface-container hover:border-border/80 text-xs font-semibold cursor-pointer transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transform-none"
          >
            <Upload className="w-3.5 h-3.5 text-muted-foreground" />
            <span>Ingest Document</span>
            <kbd className="text-[10px] font-sans font-medium text-muted-foreground border border-border px-1.5 py-0.5 rounded bg-surface-container">U</kbd>
          </button>
          <button
            type="button"
            onClick={onNewResearch}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 cursor-pointer transition-opacity active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transform-none"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Start Research</span>
            <kbd className="text-[10px] font-sans font-medium opacity-80 border border-primary-foreground/30 px-1.5 py-0.5 rounded">⌘K</kbd>
          </button>
        </div>
      </motion.div>

      {/* Main Workstation Dashboard Grid */}
      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-6"
      >
        {/* Metric Overview (4-Cell High-Density Telemetry Row) */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <motion.div variants={itemVariants} className="p-4 rounded-xl bg-surface-container-lowest border border-border flex flex-col justify-between min-h-[104px]">
            <div className="flex items-center justify-between text-muted-foreground">
              <span className="text-xs font-medium">Documents</span>
              <FileText className="w-4 h-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground font-mono tabular-nums tracking-tight">
                <AnimatedCounter value={documents.length} />
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">Indexed files in storage</p>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="p-4 rounded-xl bg-surface-container-lowest border border-border flex flex-col justify-between min-h-[104px]">
            <div className="flex items-center justify-between text-muted-foreground">
              <span className="text-xs font-medium">Vector Chunks</span>
              <Database className="w-4 h-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground font-mono tabular-nums tracking-tight">
                <AnimatedCounter value={totalChunks} />
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">Dense embeddings</p>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="p-4 rounded-xl bg-surface-container-lowest border border-border flex flex-col justify-between min-h-[104px]">
            <div className="flex items-center justify-between text-muted-foreground">
              <span className="text-xs font-medium">Active Sessions</span>
              <MessageSquare className="w-4 h-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground font-mono tabular-nums tracking-tight">
                <AnimatedCounter value={conversations.length} />
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">Conversational threads</p>
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="p-4 rounded-xl bg-surface-container-lowest border border-border flex flex-col justify-between min-h-[104px]">
            <div className="flex items-center justify-between text-muted-foreground">
              <span className="text-xs font-medium">Context Grounding</span>
              <CheckCircle2 className="w-4 h-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground font-mono tabular-nums tracking-tight">
                <AnimatedCounter value={activeDocsCount} />
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">Active in search router</p>
            </div>
          </motion.div>
        </div>

        {/* Real-time Query Telemetry (Signature Workstation Node) */}
        <motion.div 
          variants={itemVariants}
          className="rounded-xl bg-surface-container-lowest border border-border p-5 overflow-hidden flex flex-col"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-border/60 gap-3 mb-4">
            <div>
              <h2 className="font-semibold text-sm text-foreground flex items-center tracking-tight">
                <Activity className="w-4 h-4 mr-2 text-primary" />
                Query Telemetry & Throughput
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5">Real-time volume distribution across Chat Queries and Deep Research</p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-primary inline-block" />
                <span className="text-[11px] text-muted-foreground">Chat Queries</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-success inline-block" />
                <span className="text-[11px] text-muted-foreground">Deep Research</span>
              </div>
              <span className="text-[10px] font-semibold text-success bg-success-subtle px-2 py-0.5 rounded">Live</span>
            </div>
          </div>
          
          <div className="w-full h-56 min-h-[224px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 12, left: -24, bottom: 0 }} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" strokeOpacity={0.6} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} />
                <YAxis allowDecimals={false} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} />
                <Tooltip 
                  cursor={{ fill: 'var(--surface-container-high)', opacity: 0.3 }}
                  formatter={(value: any, name: string) => [
                    `${value} ${name === 'research' ? 'research queries' : 'queries'}`,
                    name === 'research' ? 'Deep Research' : 'Chat Queries'
                  ]}
                  labelFormatter={(label, payload) => {
                    const item = payload?.[0]?.payload;
                    return item?.date ? `${label} (${item.date})` : label;
                  }}
                  contentStyle={{ 
                    backgroundColor: 'var(--surface-container-lowest)', 
                    borderColor: 'var(--border)', 
                    borderRadius: '8px', 
                    fontSize: '12px', 
                    color: 'var(--foreground)' 
                  }}
                  itemStyle={{ fontWeight: 600 }}
                />
                <Bar 
                  dataKey="queries" 
                  name="queries"
                  fill="var(--primary)" 
                  radius={[3, 3, 0, 0]}
                  barSize={16}
                  animationDuration={1000}
                />
                <Bar 
                  dataKey="research" 
                  name="research"
                  fill="var(--color-success, #10b981)" 
                  radius={[3, 3, 0, 0]}
                  barSize={12}
                  animationDuration={1000}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Dual Workstation Ledgers (Split Layout: Sessions & Indexed Sources) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Recent Sessions Ledger */}
          <motion.div variants={itemVariants} className="rounded-xl bg-surface-container-lowest border border-border flex flex-col overflow-hidden">
            <div className="p-4 border-b border-border/60 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock className="w-3.5 h-3.5 text-primary" />
                <h3 className="font-semibold text-xs text-foreground">Session Log</h3>
                <span className="text-[10px] font-mono tabular-nums text-muted-foreground bg-surface-container px-1.5 py-0.5 rounded">
                  {conversations.length}
                </span>
              </div>
              <button 
                type="button" 
                onClick={() => setCurrentTab("chats")} 
                className="text-[11px] font-medium text-muted-foreground hover:text-foreground active:scale-95 disabled:opacity-50 disabled:pointer-events-none transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 inline-flex items-center gap-1"
              >
                <span>View all</span>
                <ChevronRight className="w-3 h-3" />
              </button>
            </div>
            
            <div className="flex-1 p-2 space-y-1 overflow-y-auto max-h-[320px]">
              {conversations.slice(0, 5).map((conv) => (
                <button
                  key={conv.id}
                  type="button"
                  onClick={() => onSelectSession(conv.id)}
                  className="w-full px-3 py-2.5 bg-transparent hover:bg-surface-container rounded-lg transition-colors cursor-pointer flex items-center group text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none motion-reduce:transform-none"
                >
                  <div className="min-w-0 flex-1 pr-3">
                    <h4 className="text-[13px] font-medium text-foreground truncate group-hover:text-primary transition-colors">
                      {conv.title || "Untitled Research"}
                    </h4>
                    <p className="text-[11px] text-muted-foreground truncate mt-0.5">
                      {conv.messages[conv.messages.length - 1]?.text || "No queries executed in this thread."}
                    </p>
                  </div>
                  <ArrowUpRight className="w-3.5 h-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                </button>
              ))}

              {conversations.length === 0 && (
                <div className="py-12 px-4 flex flex-col items-center justify-center text-center space-y-2">
                  <MessageSquare className="w-7 h-7 text-muted-foreground opacity-40 mb-1" />
                  <p className="text-xs font-medium text-foreground">No active research sessions</p>
                  <p className="text-[11px] text-muted-foreground max-w-xs">Start a research query to create a conversational session.</p>
                  <button
                    type="button"
                    onClick={onNewResearch}
                    className="mt-2 text-xs font-semibold text-primary hover:underline"
                  >
                    Start research session
                  </button>
                </div>
              )}
            </div>
          </motion.div>

          {/* Knowledge Base Index Ledger */}
          <motion.div variants={itemVariants} className="rounded-xl bg-surface-container-lowest border border-border flex flex-col overflow-hidden">
            <div className="p-4 border-b border-border/60 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Database className="w-3.5 h-3.5 text-muted-foreground" />
                <h3 className="font-semibold text-xs text-foreground">Knowledge Base Index</h3>
                <span className="text-[10px] font-mono tabular-nums text-muted-foreground bg-surface-container px-1.5 py-0.5 rounded">
                  {documents.length}
                </span>
              </div>
              <button 
                type="button" 
                onClick={() => setCurrentTab("documents")} 
                className="text-[11px] font-medium text-muted-foreground hover:text-foreground active:scale-95 disabled:opacity-50 disabled:pointer-events-none transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 inline-flex items-center gap-1"
              >
                <span>Manage library</span>
                <ChevronRight className="w-3 h-3" />
              </button>
            </div>

            <div className="flex-1 p-2 space-y-1 overflow-y-auto max-h-[320px]">
              {documents.slice(0, 5).map((doc) => (
                <button
                  key={doc.id}
                  type="button"
                  onClick={() => setCurrentTab("documents")}
                  className="w-full px-3 py-2.5 bg-transparent hover:bg-surface-container rounded-lg transition-colors cursor-pointer flex items-center group text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none motion-reduce:transform-none"
                >
                  <div className="min-w-0 flex-1 flex items-center">
                    <div className="p-1.5 rounded-md bg-surface-container border border-border mr-3 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors">
                      <DocTypeBadgeIcon type={doc.type} />
                    </div>
                    <div className="min-w-0">
                      <h4 className="text-[13px] font-medium text-foreground truncate group-hover:text-primary transition-colors">
                        {doc.name}
                      </h4>
                      <p className="text-[11px] text-muted-foreground mt-0.5">
                        <span className="font-mono tabular-nums">{doc.size}</span> · <span className="font-mono tabular-nums">{doc.chunkCount || 0}</span> chunks
                      </p>
                    </div>
                  </div>
                  <span className={cn(
                    "w-1.5 h-1.5 rounded-full ml-4 shrink-0", 
                    doc.active ? "bg-success" : "bg-border"
                  )} title={doc.active ? "Active in router" : "Standby"} />
                </button>
              ))}

              {documents.length === 0 && (
                <div className="py-12 px-4 flex flex-col items-center justify-center text-center space-y-2">
                  <Database className="w-7 h-7 text-muted-foreground opacity-40 mb-1" />
                  <p className="text-xs font-medium text-foreground">No documents indexed yet</p>
                  <p className="text-[11px] text-muted-foreground max-w-xs">Ingest PDFs, Word files, or Markdown to enable semantic retrieval.</p>
                  <button
                    type="button"
                    onClick={onTriggerUploadModal}
                    className="mt-2 text-xs font-semibold text-primary hover:underline"
                  >
                    Upload first document
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Bottom System Status Bar */}
        <motion.div 
          initial={{ opacity: 0 }} 
          animate={{ opacity: 1 }} 
          transition={{ delay: 0.2 }}
          className="pt-4 border-t border-border/50 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-muted-foreground gap-2"
        >
          <div className="flex items-center gap-2">
            <Activity className="w-3.5 h-3.5" />
            <span>Telemetry channel: <strong className="font-medium text-foreground">knowvex-core-v2.0</strong></span>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn("w-1.5 h-1.5 rounded-full", healthStatus === "online" ? "bg-success" : "bg-danger")} />
            <span>
              {healthStatus === "online" ? `RAG Node Synchronized (${healthTime})` : "Telemetry Disconnected"}
            </span>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
