import React, { useState, useEffect } from "react";
import { 
  Upload, MessageSquare, Search, Database, Clock, 
  FileText, ChevronRight, Activity, FileSpreadsheet, Globe
} from "lucide-react";
import { SourceDocument, Conversation } from "../types";
import { motion } from "framer-motion";
import { cn } from "../lib/utils";
import { useQuery } from "@tanstack/react-query";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { api } from "../lib/api";

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
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { type: "spring", stiffness: 300, damping: 24 }
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
        {/* Row 0: Statistics Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div className="bg-surface-container-lowest border border-border p-5 rounded-2xl shadow-sm flex flex-col justify-center">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Documents</span>
            <span className="text-2xl font-bold text-foreground">{(documents || []).length}</span>
          </div>
          <div className="bg-surface-container-lowest border border-border p-5 rounded-2xl shadow-sm flex flex-col justify-center">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Chunks</span>
            <span className="text-2xl font-bold text-foreground">{(documents || []).reduce((acc, doc: any) => acc + (doc.chunkCount || 0), 0)}</span>
          </div>
          <div className="bg-surface-container-lowest border border-border p-5 rounded-2xl shadow-sm flex flex-col justify-center">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Storage</span>
            <span className="text-2xl font-bold text-foreground">
              {(() => {
                const totalSizeKb = (documents || []).reduce((acc, doc: any) => acc + (doc.sizeKb || 0), 0);
                const totalSizeMb = totalSizeKb / 1024;
                return totalSizeMb > 0 ? `${totalSizeMb.toFixed(1)} MB` : "0.0 MB";
              })()}
            </span>
          </div>
          <div className="bg-surface-container-lowest border border-border p-5 rounded-2xl shadow-sm flex flex-col justify-center">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Sessions</span>
            <span className="text-2xl font-bold text-foreground">{(conversations || []).length}</span>
          </div>
        </div>

        {/* Row 1: Quick Action Cards (Bento style) */}
        <div className="grid grid-cols-12 gap-6">
          <motion.div 
            variants={itemVariants}
            onClick={onTriggerUploadModal}
            className="col-span-12 md:col-span-6 group relative p-6 premium-card cursor-pointer overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative z-10">
              <div className="w-10 h-10 rounded-xl bg-surface-container border border-border flex items-center justify-center text-foreground mb-4">
                <Upload className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground tracking-tight mb-1.5">Upload Documents</h3>
              <p className="text-xs text-muted-foreground leading-relaxed mb-4">
                Ingest PDFs, spreadsheets, or raw markdown text files to expand your active vector database.
              </p>
              <span className="text-xs font-semibold text-indigo-500 flex items-center opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all">
                Get Started <ChevronRight className="w-3.5 h-3.5 ml-1" />
              </span>
            </div>
          </motion.div>

          <motion.div 
            variants={itemVariants}
            onClick={onNewResearch}
            className="col-span-12 md:col-span-6 group relative p-6 premium-card cursor-pointer overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-violet-500/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative z-10">
              <div className="w-10 h-10 rounded-xl bg-surface-container border border-border flex items-center justify-center text-foreground mb-4">
                <MessageSquare className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-foreground tracking-tight mb-1.5">New Chat</h3>
              <p className="text-xs text-muted-foreground leading-relaxed mb-4">
                Launch a streaming conversation, attach specific materials, and ask questions with citations.
              </p>
              <span className="text-xs font-semibold text-violet-500 flex items-center opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all">
                Start Conversing <ChevronRight className="w-3.5 h-3.5 ml-1" />
              </span>
            </div>
          </motion.div>
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
                <div
                  key={conv.id}
                  onClick={() => onSelectSession(conv.id)}
                  className="px-4 py-3 bg-background border border-transparent hover:border-border rounded-xl transition-all duration-200 cursor-pointer flex items-center justify-between group"
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
                </div>
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
                  className="px-4 py-3 bg-background border border-transparent hover:border-border rounded-xl flex items-center justify-between transition-colors"
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
        <div className="grid grid-cols-12 gap-6">
          <motion.div variants={itemVariants} className="col-span-12 flex flex-col p-6 premium-card h-[300px]">
            <div className="flex items-center justify-between mb-6">
              <h4 className="font-semibold text-sm text-foreground flex items-center tracking-tight">
                <Activity className="w-4 h-4 mr-2 text-muted-foreground" />
                Workspace Activity (Queries)
              </h4>
            </div>
            <div className="flex-1 w-full min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={mockActivityData} margin={{ top: 15, right: 10, left: -25, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorQueries" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.12}/>
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0}/>
                    </linearGradient>
                    <linearGradient id="lineColor" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#8b5cf6" stopOpacity={1}/>
                      <stop offset="100%" stopColor="#06b6d4" stopOpacity={1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255, 255, 255, 0.05)" />
                  <XAxis 
                    dataKey="name" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fontSize: 10, fill: '#9ca3af', fontWeight: 500 }} 
                  />
                  <YAxis 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fontSize: 10, fill: '#9ca3af', fontWeight: 500 }} 
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(17, 17, 19, 0.95)', 
                      borderColor: 'rgba(255, 255, 255, 0.08)', 
                      borderRadius: '12px', 
                      fontSize: '12px',
                      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3)' 
                    }}
                    itemStyle={{ color: '#ffffff', fontWeight: 600 }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="queries" 
                    stroke="url(#lineColor)" 
                    strokeWidth={3} 
                    fillOpacity={1} 
                    fill="url(#colorQueries)" 
                    dot={{ r: 4, stroke: '#8b5cf6', strokeWidth: 1.5, fill: '#121214' }}
                    activeDot={{ r: 6, stroke: '#06b6d4', strokeWidth: 2, fill: '#ffffff' }}
                  />
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
          <span className={cn("w-1.5 h-1.5 rounded-full", healthStatus === "online" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse" : "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]")} />
          <span className="font-medium">
            {healthStatus === "online" ? <>Online (Verified <span className="font-mono text-[10px] ml-1">{healthTime}</span>)</> : "Disconnected"}
          </span>
        </div>
      </motion.div>
    </div>
  );
}
