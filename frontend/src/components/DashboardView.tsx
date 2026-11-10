import React, { useState, useEffect } from "react";
import { 
  Upload, MessageSquare, Search, Database, Clock, 
  FileText, ChevronRight, Activity, FileSpreadsheet, Globe,
  CheckCircle2
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
  { name: '08:00', queries: 12, research: 2 },
  { name: '10:00', queries: 25, research: 5 },
  { name: '12:00', queries: 15, research: 1 },
  { name: '14:00', queries: 32, research: 7 },
  { name: '16:00', queries: 18, research: 3 },
  { name: '18:00', queries: 45, research: 8 },
  { name: '20:00', queries: 22, research: 4 },
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

interface GlowCardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  variants?: any;
}

function GlowCard({ children, className, onClick, variants }: GlowCardProps) {
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
      whileHover={{ y: -2 }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onClick?.()}
      className={cn(
        "group relative p-6 premium-card cursor-pointer overflow-hidden transition-colors duration-200 hover:border-primary/45 hover:shadow-premium-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none motion-reduce:transition-none motion-reduce:transform-none",
        className
      )}
    >
      {/* Vercel Mouse-Follow Glow */}
      <div
        className="absolute inset-0 pointer-events-none transition-opacity duration-300 z-0"
        style={{
          opacity: isHovered ? 1 : 0,
          background: `radial-gradient(circle 200px at ${coords.x}px ${coords.y}px, rgba(123, 108, 246, 0.12), transparent 80%)`,
        }}
      />
      <div className="relative z-10 w-full h-full flex flex-col">
        {children}
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
    staleTime: 10 * 60 * 1000, // 10 min cache
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

  const totalChunks = documents.reduce((sum, doc) => sum + (doc.chunkCount || 18), 0);
  const activeDocsCount = documents.filter(d => d.active).length;

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 bg-transparent selection:bg-white/20 selection:text-white">
      
      {/* Title */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }} 
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-8"
      >
        <h2 className="text-2xl font-bold text-foreground tracking-tight">
          KnowVex Workspace
        </h2>
        <p className="text-sm text-muted-foreground mt-1 font-medium">
          High-density analytics and active agentic tool-routing environment.
        </p>
      </motion.div>

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-12 gap-5 auto-rows-[160px]"
      >
        
        {/* ROW 1: Tall Activity Chart (Spans 8 cols, 2 rows) */}
        <motion.div 
          variants={itemVariants}
          className="col-span-1 md:col-span-8 row-span-2 relative p-6 premium-card overflow-hidden flex flex-col"
        >
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-sm text-foreground flex items-center tracking-tight">
              <Activity className="w-4 h-4 mr-2 text-primary animate-pulse motion-reduce:animate-none" />
              Real-time Query Ticker
            </h4>
            <div className="flex items-center space-x-3 text-xs">
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-primary inline-block" />
                <span className="text-[11px] text-muted-foreground">Queries</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-success inline-block" />
                <span className="text-[11px] text-muted-foreground">Deep Research</span>
              </div>
              <span className="text-[10px] font-semibold text-success bg-success-subtle px-2 py-0.5 rounded-sm ml-1">Live</span>
            </div>
          </div>
          
          <div className="flex-1 w-full min-h-0 relative z-10">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }} barGap={4}>
                <defs>
                  <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--primary)" stopOpacity={1} />
                    <stop offset="100%" stopColor="var(--primary)" stopOpacity={0.4} />
                  </linearGradient>
                  <linearGradient id="researchBarGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={1} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0.4} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" strokeOpacity={0.5} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: 'var(--muted-foreground)', fontFamily: 'JetBrains Mono' }} />
                <YAxis allowDecimals={false} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: 'var(--muted-foreground)', fontFamily: 'JetBrains Mono' }} />
                <Tooltip 
                  cursor={{ fill: 'var(--surface-container-high)', opacity: 0.4 }}
                  formatter={(value: any, name: string) => [
                    `${value} ${name === 'research' ? 'research queries' : 'queries'}`,
                    name === 'research' ? 'Deep Research' : 'Chat Queries'
                  ]}
                  labelFormatter={(label, payload) => {
                    const item = payload?.[0]?.payload;
                    return item?.date ? `${label} (${item.date})` : label;
                  }}
                  contentStyle={{ backgroundColor: 'var(--surface-container-lowest)', borderColor: 'var(--border)', borderRadius: '6px', fontSize: '12px', fontFamily: 'JetBrains Mono', color: 'var(--foreground)' }}
                  itemStyle={{ fontWeight: 600 }}
                />
                <Bar 
                  dataKey="queries" 
                  name="queries"
                  fill="url(#barGradient)" 
                  radius={[4, 4, 0, 0]}
                  barSize={16}
                  animationDuration={1500}
                />
                <Bar 
                  dataKey="research" 
                  name="research"
                  fill="url(#researchBarGradient)" 
                  radius={[4, 4, 0, 0]}
                  barSize={12}
                  animationDuration={1500}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Quick Action: New Chat (Spans 4 cols, 1 row) */}
        <GlowCard variants={itemVariants} onClick={onNewResearch} className="col-span-1 md:col-span-4 row-span-1 flex flex-row items-center p-5">
          <div className="w-12 h-12 rounded-lg bg-surface-container border border-border flex items-center justify-center text-foreground mr-4 group-hover:bg-primary/20 group-hover:text-primary transition-colors duration-200">
            <MessageSquare className="w-6 h-6" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm text-foreground tracking-tight group-hover:text-primary transition-colors">Start Research <kbd className="text-[10px] font-sans font-medium ml-2 text-muted-foreground border border-border px-1.5 py-0.5 rounded">⌘K</kbd></h3>
            <p className="text-[11px] text-muted-foreground leading-snug mt-1 line-clamp-2">Initialize a streaming conversational session with RAG tools.</p>
          </div>
        </GlowCard>

        {/* Quick Action: Upload (Spans 4 cols, 1 row) */}
        <GlowCard variants={itemVariants} onClick={onTriggerUploadModal} className="col-span-1 md:col-span-4 row-span-1 flex flex-row items-center p-5">
          <div className="w-12 h-12 rounded-lg bg-surface-container border border-border flex items-center justify-center text-foreground mr-4 group-hover:bg-secondary/20 group-hover:text-secondary transition-colors duration-200">
            <Upload className="w-6 h-6" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm text-foreground tracking-tight group-hover:text-secondary transition-colors">Upload Knowledge <kbd className="text-[10px] font-sans font-medium ml-2 text-muted-foreground border border-border px-1.5 py-0.5 rounded">U</kbd></h3>
            <p className="text-[11px] text-muted-foreground leading-snug mt-1 line-clamp-2">Ingest new PDFs and datasets into the dense vector database.</p>
          </div>
        </GlowCard>

        {/* Metric Block 1 (Spans 3 cols, 1 row) */}
        <motion.div variants={itemVariants} className="col-span-1 md:col-span-3 row-span-1 premium-card p-5 flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <FileText className="w-4 h-4 text-muted-foreground" />
            <span className="text-[10px] font-semibold text-muted-foreground">Docs</span>
          </div>
          <div>
            <p className="text-3xl font-bold text-foreground tracking-tight font-mono tabular-nums"><AnimatedCounter value={documents.length} /></p>
            <p className="text-[11px] text-muted-foreground mt-0.5 font-medium">Total Indexed Files</p>
          </div>
        </motion.div>

        {/* Metric Block 2 (Spans 3 cols, 1 row) */}
        <motion.div variants={itemVariants} className="col-span-1 md:col-span-3 row-span-1 premium-card p-5 flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <Database className="w-4 h-4 text-muted-foreground" />
            <span className="text-[10px] font-semibold text-muted-foreground">Chunks</span>
          </div>
          <div>
            <p className="text-3xl font-bold text-foreground tracking-tight font-mono tabular-nums"><AnimatedCounter value={totalChunks} /></p>
            <p className="text-[11px] text-muted-foreground mt-0.5 font-medium">Vector Embeddings</p>
          </div>
        </motion.div>

        {/* Metric Block 3 (Spans 3 cols, 1 row) */}
        <motion.div variants={itemVariants} className="col-span-1 md:col-span-3 row-span-1 premium-card p-5 flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <MessageSquare className="w-4 h-4 text-muted-foreground" />
            <span className="text-[10px] font-semibold text-muted-foreground">Sessions</span>
          </div>
          <div>
            <p className="text-3xl font-bold text-foreground tracking-tight font-mono tabular-nums"><AnimatedCounter value={conversations.length} /></p>
            <p className="text-[11px] text-muted-foreground mt-0.5 font-medium">Active Chats</p>
          </div>
        </motion.div>

        {/* Metric Block 4 (Spans 3 cols, 1 row) */}
        <motion.div variants={itemVariants} className="col-span-1 md:col-span-3 row-span-1 premium-card p-5 flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <CheckCircle2 className="w-4 h-4 text-muted-foreground" />
            <span className="text-[10px] font-semibold text-muted-foreground">Coverage</span>
          </div>
          <div>
            <p className="text-3xl font-bold text-foreground tracking-tight font-mono tabular-nums"><AnimatedCounter value={activeDocsCount} /></p>
            <p className="text-[11px] text-muted-foreground mt-0.5 font-medium">Active in Context</p>
          </div>
        </motion.div>

        {/* Recent Chats (Spans 6 cols, 2 rows) */}
        <motion.div variants={itemVariants} className="col-span-1 md:col-span-6 row-span-2 premium-card flex flex-col overflow-hidden">
          <div className="p-4 border-b border-border/50 flex items-center justify-between">
             <h4 className="font-semibold text-xs text-foreground flex items-center">
                <Clock className="w-3.5 h-3.5 mr-2 text-primary" />
                Session Log
              </h4>
              <button type="button" onClick={() => setCurrentTab("chats")} className="text-[11px] font-medium text-muted-foreground hover:text-foreground active:scale-95 disabled:opacity-50 disabled:pointer-events-none transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40">View all</button>
          </div>
          <div className="flex-1 overflow-y-auto no-scrollbar p-2 space-y-1">
             {conversations.slice(0, 5).map((conv) => (
                <button
                  key={conv.id}
                  onClick={() => onSelectSession(conv.id)}
                  className="w-full px-3 py-2.5 bg-transparent hover:bg-surface-container-lowest rounded-lg transition-colors cursor-pointer flex items-center group text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none motion-reduce:transform-none"
                >
                  <div className="min-w-0 flex-1 pr-4">
                    <h5 className="text-[13px] font-medium text-foreground truncate">{conv.title || "Untitled Research"}</h5>
                    <p className="text-[11px] text-muted-foreground truncate mt-0.5">
                      {conv.messages[conv.messages.length - 1]?.text || "No queries executed."}
                    </p>
                  </div>
                  <ChevronRight className="w-3.5 h-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                </button>
              ))}
              {conversations.length === 0 && (
                <div className="h-full flex items-center justify-center p-4">
                  <p className="text-xs text-muted-foreground">No active sessions.</p>
                </div>
              )}
          </div>
        </motion.div>

        {/* Recent Documents (Spans 6 cols, 2 rows) */}
        <motion.div variants={itemVariants} className="col-span-1 md:col-span-6 row-span-2 premium-card flex flex-col overflow-hidden">
          <div className="p-4 border-b border-border/50 flex items-center justify-between">
             <h4 className="font-semibold text-xs text-foreground flex items-center">
                <Database className="w-3.5 h-3.5 mr-2 text-secondary" />
                Knowledge Base Index
              </h4>
              <button type="button" onClick={() => setCurrentTab("documents")} className="text-[11px] font-medium text-muted-foreground hover:text-foreground active:scale-95 disabled:opacity-50 disabled:pointer-events-none transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40">Manage</button>
          </div>
          <div className="flex-1 overflow-y-auto no-scrollbar p-2 space-y-1">
             {documents.slice(0, 5).map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => setCurrentTab("documents")}
                  className="w-full px-3 py-2.5 bg-transparent hover:bg-surface-container-lowest rounded-lg transition-colors cursor-pointer flex items-center group text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none motion-reduce:transform-none"
                >
                  <div className="min-w-0 flex-1 flex items-center">
                    {/* Single neutral doc-type icon — type communicated by shape, not hue */}
                    <div className="p-1 rounded bg-surface-container border border-border mr-3 flex-shrink-0 text-muted-foreground">
                      {doc.type === "spreadsheet" && <FileSpreadsheet className="w-3.5 h-3.5" />}
                      {doc.type === "link" && <Globe className="w-3.5 h-3.5" />}
                      {(doc.type === "pdf" || doc.type === "doc") && <FileText className="w-3.5 h-3.5" />}
                    </div>
                    <div className="min-w-0">
                      <h6 className="text-[13px] font-medium text-foreground truncate">{doc.name}</h6>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        <span className="font-mono tabular-nums">{doc.size}</span> · <span className="font-mono tabular-nums">{doc.chunkCount || 0}</span> chunks
                      </p>
                    </div>
                  </div>
                  <span className={cn("w-1.5 h-1.5 rounded-full ml-4 flex-shrink-0", doc.active ? "bg-success" : "bg-border")} />
                </button>
              ))}
              {documents.length === 0 && (
                <div className="h-full flex items-center justify-center p-4">
                  <p className="text-xs text-muted-foreground">No documents indexed.</p>
                </div>
              )}
          </div>
        </motion.div>

      </motion.div>

      {/* Bottom Health Strip */}
      <motion.div 
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        className="mt-8 flex items-center justify-between text-xs text-muted-foreground"
      >
        <div className="flex items-center space-x-2">
          <Activity className="w-3.5 h-3.5" />
          <span>Core RAG Node: <strong className="font-medium text-foreground font-mono">monolith-v4.2.0</strong></span>
        </div>
        <div className="flex items-center space-x-2 bg-surface-container px-2.5 py-1 rounded-md border border-border">
          <span className={cn("w-1.5 h-1.5 rounded-full", healthStatus === "online" ? "bg-success" : "bg-danger")} />
          <span className="font-medium text-[11px]">
            {healthStatus === "online" ? `Online (${healthTime})` : "Disconnected"}
          </span>
        </div>
      </motion.div>
    </div>
  );
}
