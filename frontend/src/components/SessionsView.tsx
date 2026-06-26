import React, { useState } from "react";
import { 
  List, Search, Trash2, Edit2, Download, Maximize2, X, Calendar, 
  MessageSquare, FileText, ChevronRight, CheckCircle, Database 
} from "lucide-react";
import { Conversation, SourceDocument, Message } from "../types";
import { cn } from "../lib/utils";

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
  const [selectedConvId, setSelectedConvId] = useState<string | null>(null);
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [tempName, setTempName] = useState("");

  const selectedConv = conversations.find(c => c.id === selectedConvId);

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

  // Helper to format relative dates
  const formatRelativeTime = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    if (diffMins < 60) return `${Math.max(1, diffMins)}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
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
                <th className="p-3">Messages</th>
                <th className="p-3">Linked Docs</th>
                <th className="p-3">Last Active</th>
                <th className="p-3 pr-5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filteredConvs.map((conv) => (
                <tr 
                  key={conv.id}
                  onClick={() => setSelectedConvId(conv.id)}
                  className={cn(
                    "hover:bg-surface/50 cursor-pointer transition-colors",
                    selectedConvId === conv.id ? "bg-surface" : "bg-background"
                  )}
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
                      </div>
                    )}
                  </td>

                  {/* Messages Count */}
                  <td className="p-3 text-xs font-medium text-muted-foreground">
                    {conv.messages.length} messages
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
                        onClick={() => onSelectSession(conv.id)}
                        title="Open Conversation Window"
                        className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md cursor-pointer transition-colors"
                      >
                        <Maximize2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => onDeleteSession(conv.id)}
                        title="Delete Session"
                        className="p-1.5 text-muted-foreground hover:text-red-500 hover:bg-red-500/10 rounded-md cursor-pointer transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {filteredConvs.length === 0 && (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-xs text-muted-foreground italic bg-background">
                    No matching sessions found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right Drawer Panel (320px slide in) */}
      {selectedConv && (
        <div className="w-80 border-l border-border bg-surface-container-lowest h-full p-6 space-y-6 flex flex-col justify-between shadow-premium">
          <div className="space-y-6 overflow-y-auto flex-1 pr-1">
            {/* Drawer Header */}
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Details</span>
                <h3 className="text-sm font-bold text-foreground leading-tight mt-1">{selectedConv.title}</h3>
              </div>
              <button 
                onClick={() => setSelectedConvId(null)}
                className="p-1.5 hover:bg-surface rounded-md cursor-pointer transition-colors"
              >
                <X className="w-4 h-4 text-muted-foreground" />
              </button>
            </div>

            {/* General Info */}
            <div className="space-y-3 bg-background p-4 rounded-lg border border-border">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center font-medium"><Calendar className="w-3.5 h-3.5 mr-1.5" /> Created:</span>
                <span className="font-mono">{new Date(selectedConv.timestamp).toLocaleDateString()}</span>
              </div>
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center font-medium"><MessageSquare className="w-3.5 h-3.5 mr-1.5" /> Size:</span>
                <span className="font-medium text-foreground">{selectedConv.messages.length} Q&As</span>
              </div>
            </div>

            {/* Attached files names checklist */}
            <div className="space-y-2">
              <h4 className="text-[10px] font-semibold uppercase text-muted-foreground tracking-wider">Grounding Scope</h4>
              <div className="space-y-1.5">
                {((selectedConv as any).attachedDocIds || []).map((docId: string) => {
                  const doc = documents.find(d => d.id === docId);
                  return (
                    <div key={docId} className="flex items-center space-x-2 text-[11px] font-medium text-foreground bg-background p-2 rounded-md border border-border">
                      <FileText className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                      <span className="truncate">{doc ? doc.name : docId}</span>
                    </div>
                  );
                })}
                {((selectedConv as any).attachedDocIds || []).length === 0 && (
                  <p className="text-[11px] text-muted-foreground italic">No documents attached.</p>
                )}
              </div>
            </div>

            {/* Message Thread preview */}
            <div className="space-y-3">
              <h4 className="text-[10px] font-semibold uppercase text-muted-foreground tracking-wider">Thread Preview</h4>
              <div className="space-y-3 max-h-60 overflow-y-auto">
                {selectedConv.messages.slice(0, 3).map((msg, mIdx) => (
                  <div key={msg.id || mIdx} className="p-3 bg-background border border-border rounded-lg space-y-1.5">
                    <span className="text-[10px] font-semibold uppercase text-muted-foreground tracking-wider block">
                      {msg.sender === "user" ? "👤 User Query" : "🤖 System"}
                    </span>
                    <p className="text-[11px] text-foreground leading-relaxed line-clamp-3">
                      {msg.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Drawer Actions CTA bottom footer */}
          <div className="pt-4 border-t border-border space-y-2 flex-shrink-0">
            <button
              onClick={() => onSelectSession(selectedConv.id)}
              className="w-full py-2 bg-primary text-primary-foreground text-[11px] font-semibold rounded-md cursor-pointer hover:bg-primary/90 transition-all flex items-center justify-center space-x-1.5 shadow-sm"
            >
              <span>Launch Conversing Screen</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => handleExportMarkdown(selectedConv)}
              className="w-full py-2 bg-background hover:bg-surface text-foreground text-[11px] font-semibold rounded-md border border-border cursor-pointer transition-all flex items-center justify-center space-x-1.5 shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Logbook</span>
            </button>
          </div>
        </div>
      )}

    </div>
  );
}
