import React, { useState, useEffect } from "react";
import { 
  List, Search, Trash2, Edit2, Download, Maximize2, X, Calendar, 
  MessageSquare, FileText, ChevronRight, CheckCircle, Database 
} from "lucide-react";
import { Conversation, SourceDocument, Message } from "../types";
import { cn } from "../lib/utils";
import ConfirmDialog from "./ConfirmDialog";

// Module-scope pure function: no local state used
function formatRelativeTime(isoString: string) {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  if (diffMins < 60) return `${Math.max(1, diffMins)}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return date.toLocaleDateString();
}

interface SessionsViewProps {
  conversations: Conversation[];
  documents: SourceDocument[];
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newName: string) => void;
}

export default function SessionsView({
  conversations,
  documents,
  onSelectSession,
  onDeleteSession,
  onRenameSession,
}: SessionsViewProps) {
  const [search, setSearch] = useState("");
  const [selectedDocFilter, setSelectedDocFilter] = useState("all");
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [tempName, setTempName] = useState("");
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

  const [showSkeleton, setShowSkeleton] = useState(false);

  // Filter conversations
  const filteredConvs = conversations.filter(c => {
    const matchesSearch = c.title.toLowerCase().includes(search.toLowerCase());
    // Filter by attached document if necessary
    const matchesDoc = selectedDocFilter === "all" || (c as any).attachedDocIds?.includes(selectedDocFilter);
    return matchesSearch && matchesDoc;
  });

  const handleStartRename = (id: string, currentTitle: string) => {
    setEditingConvId(id);
    setTempName(currentTitle);
  };

  const handleSaveRename = (id: string) => {
    if (tempName.trim()) {
      onRenameSession(id, tempName.trim());
    }
    setEditingConvId(null);
  };


  const handleExportMarkdown = (conv: Conversation) => {
    const md = `
# DocuMind Conversation Log: ${conv.title}
Created: ${new Date(conv.timestamp).toLocaleString()}

${conv.messages.map((msg) => `
### ${msg.sender === "user" ? "👤 User Query" : "🤖 Assistant Grounded Answer"} (${formatRelativeTime(msg.timestamp)})
${msg.text}

${msg.citations && msg.citations.length > 0 ? `
**Cited Source Proofs:**
${msg.citations.map(cit => `- **${cit.name}** (${cit.fitScore}% Match): "${cit.snippet}"`).join("\n")}
` : ""}
---
`).join("\n")}
`;
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `documind_conversation_${conv.id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 overflow-hidden flex select-none bg-background">
      
      {/* Left panel: Sessions table database lists */}
      <div className="flex-1 overflow-y-auto px-6 md:px-12 py-10 space-y-6">
        
        {/* Title */}
        <div>
          <h2 className="text-2xl font-bold text-foreground flex items-center tracking-tight">
            <List className="w-5 h-5 mr-3 text-muted-foreground" />
            Sessions
          </h2>
          <p className="text-sm text-muted-foreground mt-1.5 font-medium">
            Browse active chat histories, rename research threads, filter by linked documents, and export conversation backups as Markdown logs.
          </p>
        </div>

        {/* Search & Filter Toolbar */}
        <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="relative w-full md:w-72">
            <Search className="absolute left-3 top-2.5 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search conversations..."
              className="w-full text-sm font-medium bg-background border border-border rounded-lg pl-9 pr-4 py-2 text-foreground focus:outline-none focus:ring-1 focus:ring-foreground/20 shadow-sm"
            />
          </div>

          <div className="flex items-center space-x-3 w-full md:w-auto">
            <span className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground">Attached Document:</span>
            <select
              value={selectedDocFilter}
              onChange={(e) => setSelectedDocFilter(e.target.value)}
              className="text-xs font-medium bg-background border border-border px-2.5 py-1.5 rounded-lg text-foreground focus:outline-none shadow-sm cursor-pointer"
            >
              <option value="all">All Documents</option>
              {documents.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Table layout */}
        <div className="bg-surface-container-lowest border border-border rounded-xl overflow-hidden shadow-sm">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-border bg-surface text-[10px] uppercase font-semibold text-muted-foreground tracking-wider">
                <th className="p-3 pl-5">Research Title</th>
                <th className="p-3">Created</th>
                <th className="p-3">Linked Docs</th>
                <th className="p-3">Last Active</th>
                <th className="p-3 pr-5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {showSkeleton ? (
                Array.from({ length: 3 }).map((_, i) => (
                  <tr key={i} className="animate-pulse bg-background">
                    <td className="p-4 pl-5">
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 bg-surface-container rounded" />
                        <div className="h-4 bg-surface-container rounded w-48 skeleton-shimmer-bar" />
                      </div>
                    </td>
                    <td className="p-4">
                      <div className="h-3 bg-surface-container rounded w-16 skeleton-shimmer-bar" />
                    </td>
                    <td className="p-4">
                      <div className="h-4 bg-surface-container rounded w-20 skeleton-shimmer-bar" />
                    </td>
                    <td className="p-4">
                      <div className="h-3 bg-surface-container rounded w-16 skeleton-shimmer-bar" />
                    </td>
                    <td className="p-4 pr-5 text-right">
                      <div className="h-6 bg-surface-container rounded w-24 ml-auto skeleton-shimmer-bar" />
                    </td>
                  </tr>
                ))
              ) : filteredConvs.length === 0 ? (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-xs text-muted-foreground italic bg-background">
                    No matching sessions found.
                  </td>
                </tr>
              ) : (
                filteredConvs.map((conv) => (
                  <tr 
                    key={conv.id}
                    onClick={() => onSelectSession(conv.id)}
                    className="hover:bg-surface/50 cursor-pointer transition-colors bg-background"
                  >
                    {/* Name column */}
                    <td className="p-3 pl-5">
                      {editingConvId === conv.id ? (
                        <div className="flex items-center space-x-2" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="text"
                            value={tempName}
                            onChange={(e) => setTempName(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSaveRename(conv.id)}
                            className="bg-background border border-border px-2 py-1 rounded-md text-xs text-foreground font-medium focus:outline-none"
                          />
                          <button 
                            onClick={() => handleSaveRename(conv.id)}
                            className="p-1 text-primary hover:bg-primary/10 rounded-md transition-colors"
                          >
                            <CheckCircle className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <div 
                          onDoubleClick={() => handleStartRename(conv.id, conv.title)}
                          className="text-sm font-semibold text-foreground hover:text-primary transition-colors flex items-center space-x-2"
                        >
                          <MessageSquare className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                          <span>{conv.title}</span>
                          <span className="text-[9px] font-bold text-primary bg-primary/10 px-1.5 py-0.5 rounded-full uppercase tracking-wider">
                            Auto
                          </span>
                        </div>
                      )}
                    </td>

                    {/* Created Date */}
                    <td className="p-3 text-xs font-mono text-muted-foreground">
                      {new Date(conv.timestamp).toLocaleDateString()}
                    </td>

                    {/* Document Badge Count */}
                    <td className="p-3">
                      <span className="text-[10px] bg-background border border-border text-foreground font-medium px-2 py-0.5 rounded-md inline-flex items-center">
                        <Database className="w-3 h-3 mr-1 text-muted-foreground" />
                        {(conv as any).attachedDocIds?.length || conv.documentCount || 0} attached
                      </span>
                    </td>

                    {/* Last Active relative time */}
                    <td className="p-3 text-xs font-mono text-muted-foreground">
                      {formatRelativeTime(conv.timestamp)}
                    </td>

                    {/* Actions buttons */}
                    <td className="p-3 pr-5 text-right" onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-end space-x-1">
                        <button
                          onClick={() => handleExportMarkdown(conv)}
                          title="Download Markdown Log"
                          className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md cursor-pointer transition-colors"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleStartRename(conv.id, conv.title)}
                          title="Rename Session"
                          className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md cursor-pointer transition-colors"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => onSelectSession(conv.id)}
                          title="Open Conversation Window"
                          className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md cursor-pointer transition-colors"
                        >
                          <Maximize2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setSessionToDelete(conv.id)}
                          title="Delete Session"
                          className="p-1.5 text-muted-foreground hover:text-red-500 hover:bg-red-500/10 rounded-md cursor-pointer transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <ConfirmDialog
        isOpen={!!sessionToDelete}
        title="Delete Session?"
        description="This will permanently delete this chat session and all its messages. This action cannot be undone."
        confirmLabel="Delete Session"
        onConfirm={() => {
          if (sessionToDelete) onDeleteSession(sessionToDelete);
          setSessionToDelete(null);
        }}
        onCancel={() => setSessionToDelete(null)}
      />
    </div>
  );
}
