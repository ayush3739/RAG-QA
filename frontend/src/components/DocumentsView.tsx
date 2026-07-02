import React, { useState, useRef, useEffect } from "react";
import { SourceDocument } from "../types";
import { 
  FolderOpen, FileText, FileSpreadsheet, Globe, Plus, Trash2, 
  UploadCloud, Loader2, CheckCircle2, AlertTriangle, File
} from "lucide-react";
import { cn } from "../lib/utils";
import { useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "../lib/api";
import { toast } from "sonner";
import ConfirmDialog from "./ConfirmDialog";

// Module-scope pure function: no local state used
function handleDragOver(e: React.DragEvent) {
  e.preventDefault();
}

interface DocumentsViewProps {
  documents: SourceDocument[];
  onToggleActive: (id: string) => void;
  onDeleteDocument: (id: string) => void;
  onAddDocument: (file: File) => void;
  onSelectDocument: (id: string) => void;
}

export function DocumentsView({
  documents,
  onToggleActive,
  onDeleteDocument,
  onAddDocument,
  onSelectDocument,
}: DocumentsViewProps) {
  const [isDragging, setIsDragging] = useState(false);
  const dragCounterRef = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [docToDelete, setDocToDelete] = useState<string | null>(null);
  
  const [showSkeleton, setShowSkeleton] = useState(false);

  const handleUpload = (file: File) => {
    if (file.size === 0) {
      toast.error(`Upload failed: File ${file.name} is empty.`);
      return;
    }
    if (file.size > 30 * 1024 * 1024) {
      toast.error(`Upload failed: File size for ${file.name} exceeds the 30 MB limit.`);
      return;
    }
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const allowed = ['.pdf', '.txt', '.md', '.docx'];
    if (file.name.includes('.') && !allowed.includes(ext)) {
      toast.error(`Upload failed: Unsupported file extension '${ext}'. Allowed extensions: ${allowed.join(', ')}`);
      return;
    }
    onAddDocument(file);
  };

  const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      Array.from(e.target.files).forEach(handleUpload);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current += 1;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) {
      setIsDragging(false);
    }
  };


  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    dragCounterRef.current = 0;
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      Array.from(e.dataTransfer.files).forEach(handleUpload);
    }
  };

  return (
    <div 
      className="flex-1 overflow-hidden flex select-none bg-background relative"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <input 
        type="file" 
        multiple 
        className="hidden" 
        ref={fileInputRef} 
        onChange={onFileSelect}
        accept=".pdf,.txt,.md,.docx"
      />
      
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            className="absolute inset-0 bg-primary/15 backdrop-blur-[4px] z-40 border border-primary m-6 rounded-3xl flex items-center justify-center pointer-events-none shadow-[0_0_50px_rgba(123,108,246,0.3)] transition-all duration-300"
          >
            <div className="flex flex-col items-center space-y-4 bg-[#0d0d0f]/90 border border-white/10 p-10 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)]">
              <motion.div
                animate={{ y: [0, -14, 0] }}
                transition={{ duration: 1.2, ease: "easeInOut", repeat: Infinity }}
              >
                <UploadCloud className="w-16 h-16 text-primary drop-shadow-[0_0_15px_rgba(123,108,246,0.5)]" />
              </motion.div>
              <h2 className="text-2xl font-bold text-foreground tracking-tight">Drop files to ingest</h2>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Left List of Documents */}
      <div className="flex-1 overflow-y-auto px-6 md:px-12 py-10 space-y-8">
        <div>
          <h2 className="text-2xl md:text-3xl font-bold text-foreground tracking-tight flex items-center">
            <FolderOpen className="w-6 h-6 mr-3 text-muted-foreground" />
            Knowledge Library
          </h2>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl font-medium">
            Browse, activate, or delete source materials. Active documents are instantly parsed and queryable by the RAG search router.
          </p>
        </div>

        {/* Upload Zone */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="w-full border-2 border-dashed border-border hover:border-primary/50 transition-colors rounded-2xl bg-surface-container-lowest p-8 flex flex-col items-center justify-center cursor-pointer group text-center"
        >
          <div className="w-12 h-12 rounded-xl bg-surface-container border border-border flex items-center justify-center text-muted-foreground group-hover:text-primary group-hover:bg-primary/10 transition-colors mb-4">
            <UploadCloud className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-semibold text-foreground tracking-tight mb-1">Click or drag documents to ingest</h3>
          <p className="text-xs text-muted-foreground">Supports PDF, DOCX, TXT, and Markdown files (Max 30 MB)</p>
        </button>

        {/* Existing Documents Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {showSkeleton ? (
            Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="p-5 premium-card animate-pulse relative overflow-hidden flex items-start justify-between min-h-[120px]">
                <div className="flex items-center space-x-3 w-full">
                  <div className="p-2.5 rounded-lg bg-surface-container-high w-10 h-10 shrink-0" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-surface-container rounded w-3/4 skeleton-shimmer-bar" />
                    <div className="h-3 bg-surface-container rounded w-1/4 skeleton-shimmer-bar" />
                  </div>
                </div>
              </div>
            ))
          ) : documents.length === 0 ? (
            <div className="col-span-2 text-center p-8 bg-surface-container-lowest/40 rounded-2xl border border-border">
              <p className="text-sm text-muted-foreground italic">No source materials loaded in library.</p>
            </div>
          ) : (
            documents.map((doc) => {
            const isProcessing = doc.status === "queued" || doc.status === "indexing";
            const isFailed = doc.status === "failed";
            const isIndexed = doc.status === "indexed" || !doc.status;

            return (
              <div
                key={doc.id}
                role="button"
                tabIndex={0}
                onClick={() => onSelectDocument(doc.id)}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelectDocument(doc.id)}
                className={cn(
                  "p-5 premium-card relative cursor-pointer glow-hover group overflow-hidden",
                  doc.active && isIndexed
                    ? "border-primary/30 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1)]"
                    : "",
                  isProcessing ? "opacity-80" : ""
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={cn(
                      "p-2.5 rounded-lg border",
                      doc.type === "spreadsheet" ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-500" :
                      doc.type === "link" ? "bg-sky-500/10 border-sky-500/20 text-sky-500" :
                      doc.type === "pdf" ? "bg-rose-500/10 border-rose-500/20 text-rose-500" :
                      "bg-amber-500/10 border-amber-500/20 text-amber-500"
                    )}>
                      {doc.type === "spreadsheet" && <FileSpreadsheet className="w-5 h-5" />}
                      {doc.type === "link" && <Globe className="w-5 h-5" />}
                      {doc.type === "pdf" && <FileText className="w-5 h-5" />}
                      {doc.type === "doc" && <File className="w-5 h-5" />}
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-foreground truncate max-w-[150px]">{doc.name}</h4>
                      <p className="text-[11px] text-muted-foreground font-semibold font-mono mt-0.5">{doc.size}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2" onClick={(e) => e.stopPropagation()}>
                    {!isProcessing && !isFailed && (
                      <button
                        onClick={() => onToggleActive(doc.id)}
                        className={cn(
                          "px-3 py-1 rounded-full text-[10px] font-bold cursor-pointer transition-all shadow-sm",
                          doc.active
                            ? "bg-primary text-primary-foreground"
                            : "bg-surface border border-border text-muted-foreground hover:text-foreground"
                        )}
                      >
                        {doc.active ? "ACTIVE" : "STANDBY"}
                      </button>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDocToDelete(doc.id);
                      }}
                      className="p-1.5 text-muted-foreground hover:bg-red-500/10 hover:text-red-500 rounded-md transition-colors cursor-pointer"
                      title="Remove document"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Status Indicator */}
                <div className="mt-4 pt-3 border-t border-border/50 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {isProcessing && <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />}
                    {isIndexed && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />}
                    {isFailed && <AlertTriangle className="w-3.5 h-3.5 text-red-500" />}
                    
                    <span className={cn(
                      "text-xs font-medium tracking-tight",
                      isProcessing ? "text-primary" : 
                      isFailed ? "text-red-500" : 
                      "text-emerald-500"
                    )}>
                      {isProcessing ? "Indexing..." : isFailed ? "Failed" : "Indexed Successfully"}
                    </span>
                  </div>
                  
                  {isIndexed && (
                    <span className="text-[10px] font-mono font-medium text-muted-foreground">
                      {doc.chunkCount || 0} chunks
                    </span>
                  )}
                </div>
                
                {isProcessing && (
                  <div className="absolute bottom-0 left-0 right-0 h-1 bg-surface-container-high overflow-hidden">
                     <motion.div 
                       className="h-full bg-primary" 
                       initial={{ width: "0%" }}
                       animate={{ width: "100%" }}
                       transition={{ duration: 2, repeat: Infinity }}
                     />
                  </div>
                )}
              </div>
            );
          }))}
        </div>
      </div>

      <ConfirmDialog
        isOpen={!!docToDelete}
        title="Delete Document?"
        description="This will permanently delete the document and all its indexed chunks. This action cannot be undone."
        confirmLabel="Delete Document"
        onConfirm={() => {
          if (docToDelete) onDeleteDocument(docToDelete);
          setDocToDelete(null);
        }}
        onCancel={() => setDocToDelete(null)}
      />
    </div>
  );
}
