import React, { useState, useRef } from "react";
import { SourceDocument } from "../types";
import { 
  FileText, FileSpreadsheet, Globe, 
  UploadCloud, Loader2, CheckCircle2, AlertTriangle, File,
  Database, Layers
} from "lucide-react";
import { cn } from "../lib/utils";
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

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.06 } }
};

const cardVariants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { type: "spring" as const, stiffness: 300, damping: 26 } }
};

function DocTypeIcon({ type }: { type: string }) {
  if (type === "spreadsheet") return <FileSpreadsheet className="w-5 h-5" />;
  if (type === "link") return <Globe className="w-5 h-5" />;
  if (type === "pdf") return <FileText className="w-5 h-5" />;
  return <File className="w-5 h-5" />;
}

// Single neutral color — shape communicates type, not hue
function docTypeColor(_type: string) {
  return "bg-surface-container border-border text-muted-foreground";
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
  const [showSkeleton] = useState(false);

  const handleUpload = (file: File) => {
    if (file.size === 0) { toast.error(`Upload failed: File ${file.name} is empty.`); return; }
    if (file.size > 30 * 1024 * 1024) { toast.error(`Upload failed: File size for ${file.name} exceeds the 30 MB limit.`); return; }
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const allowed = ['.pdf', '.txt', '.md', '.docx'];
    if (file.name.includes('.') && !allowed.includes(ext)) {
      toast.error(`Upload failed: Unsupported file extension '${ext}'. Allowed extensions: ${allowed.join(', ')}`);
      return;
    }
    onAddDocument(file);
  };

  const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) Array.from(e.target.files).forEach(handleUpload);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current += 1;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    dragCounterRef.current = 0;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) Array.from(e.dataTransfer.files).forEach(handleUpload);
  };

  const activeCount = documents.filter(d => d.active).length;
  const totalChunks = documents.reduce((sum, d) => sum + (d.chunkCount || 0), 0);

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

      {/* Drag Overlay */}
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-primary/10 backdrop-blur-sm z-40 border-2 border-dashed border-primary m-4 rounded-2xl flex items-center justify-center pointer-events-none"
          >
            <div className="flex flex-col items-center space-y-4 bg-surface-container-lowest/90 backdrop-blur border border-border p-10 rounded-2xl shadow-2xl">
              <motion.div
                animate={{ y: [0, -12, 0] }}
                transition={{ duration: 1.2, ease: "easeInOut", repeat: Infinity }}
              >
                <UploadCloud className="w-14 h-14 text-primary" />
              </motion.div>
              <h2 className="text-xl font-bold text-foreground tracking-tight">Drop files to ingest</h2>
              <p className="text-sm text-muted-foreground">PDF, DOCX, TXT, Markdown · Max 30 MB</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Scrollable Content */}
      <div className="flex-1 overflow-y-auto px-6 md:px-12 py-10 space-y-8">

        {/* Header */}
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <h2 className="text-2xl md:text-3xl font-bold text-foreground tracking-tight">
            Knowledge Library
          </h2>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-2xl font-medium">
            Manage your source materials. Active documents are instantly queryable by the RAG search router.
          </p>

          {/* Stats Strip */}
          {documents.length > 0 && (
            <div className="flex items-center gap-4 mt-5">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-container-lowest border border-border rounded-lg text-xs font-medium text-muted-foreground">
                <Layers className="w-3.5 h-3.5 text-muted-foreground" />
                <span><strong className="text-foreground font-mono">{documents.length}</strong> documents</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-container-lowest border border-border rounded-lg text-xs font-medium text-muted-foreground">
                <Database className="w-3.5 h-3.5 text-muted-foreground" />
                <span><strong className="text-foreground font-mono">{totalChunks}</strong> chunks indexed</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-container-lowest border border-border rounded-lg text-xs font-medium text-muted-foreground">
                {/* Active dot stays green — it's semantic (live/inactive) */}
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
                <span><strong className="text-foreground font-mono">{activeCount}</strong> active</span>
              </div>
            </div>
          )}
        </motion.div>

        {/* Upload Zone */}
        <motion.button
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.4 }}
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="w-full border border-border hover:border-border/70 hover:bg-surface-container/50 transition-all duration-200 rounded-xl bg-surface p-8 flex flex-col items-center justify-center cursor-pointer group"
        >
          <div className="w-11 h-11 rounded-xl bg-surface-container border border-border flex items-center justify-center text-muted-foreground group-hover:text-foreground group-hover:border-border/70 transition-all duration-200 mb-4">
            <UploadCloud className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-semibold text-foreground tracking-tight mb-1">Click or drag documents to ingest</h3>
          <p className="text-xs text-muted-foreground">PDF, DOCX, TXT, Markdown · Max 30 MB per file</p>
        </motion.button>

        {/* Document Grid */}
        {showSkeleton ? (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="p-5 premium-card animate-pulse flex items-start justify-between min-h-[120px]">
                <div className="flex items-center space-x-3 w-full">
                  <div className="p-2.5 rounded-lg bg-surface-container-high w-10 h-10 shrink-0" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-surface-container rounded w-3/4" />
                    <div className="h-3 bg-surface-container rounded w-1/4" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : documents.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="col-span-2 flex flex-col items-center justify-center py-24 px-8 bg-surface-container-lowest/40 rounded-2xl border border-dashed border-border"
          >
            {/* Animated illustration */}
            <div className="relative mb-8">
              <motion.div
                className="w-20 h-20 rounded-xl bg-surface-container border border-border flex items-center justify-center"
                animate={{ y: [0, -8, 0] }}
                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              >
                <UploadCloud className="w-9 h-9 text-muted-foreground" />
              </motion.div>
            </div>

            <h3 className="text-lg font-semibold text-foreground tracking-tight mb-2">Your library is empty</h3>
            <p className="text-sm text-muted-foreground text-center max-w-xs leading-relaxed">
              Upload your first document to get started. The RAG router will index it and make it instantly queryable.
            </p>
          </motion.div>
        ) : (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-1 xl:grid-cols-2 gap-4"
          >
            {documents.map((doc) => {
              const isProcessing = doc.status === "queued" || doc.status === "indexing";
              const isFailed = doc.status === "failed";
              const isIndexed = doc.status === "indexed" || !doc.status;

              return (
                <motion.div
                  key={doc.id}
                  variants={cardVariants}
                  whileHover={{ y: -2 }}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectDocument(doc.id)}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelectDocument(doc.id)}
                  className={cn(
                    "p-5 premium-card relative cursor-pointer group overflow-hidden transition-all duration-300 hover:border-primary/30 hover:shadow-premium-hover",
                    doc.active && isIndexed ? "border-primary/20" : "",
                    isProcessing ? "opacity-80" : ""
                  )}
                >
                  {/* Subtle top-left glow for active docs */}
                  {doc.active && isIndexed && (
                    <div className="absolute top-0 left-0 w-24 h-24 bg-primary/5 rounded-full blur-2xl pointer-events-none" />
                  )}

                  {/* Card Header */}
                  <div className="flex items-start justify-between relative z-10">
                    <div className="flex items-center space-x-3 min-w-0">
                      <div className={cn("p-2.5 rounded-xl border shrink-0", docTypeColor(doc.type))}>
                        <DocTypeIcon type={doc.type} />
                      </div>
                      <div className="min-w-0">
                        <h4 className="text-sm font-semibold text-foreground truncate max-w-[200px] group-hover:text-primary transition-colors">{doc.name}</h4>
                        <p className="text-[11px] text-muted-foreground font-mono mt-0.5">{doc.size}</p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2 shrink-0 ml-3" onClick={(e) => e.stopPropagation()}>
                      {!isProcessing && !isFailed && (
                        <button
                          onClick={() => onToggleActive(doc.id)}
                          className={cn(
                            "px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wider cursor-pointer transition-all border",
                            doc.active
                              ? "bg-primary/10 border-primary/30 text-primary"
                              : "bg-surface border-border text-muted-foreground hover:text-foreground hover:border-foreground/30"
                          )}
                        >
                          {doc.active ? "ACTIVE" : "STANDBY"}
                        </button>
                      )}
                      <button
                        onClick={(e) => { e.stopPropagation(); setDocToDelete(doc.id); }}
                        className="p-1.5 text-muted-foreground hover:bg-red-500/10 hover:text-red-500 rounded-lg transition-colors cursor-pointer opacity-0 group-hover:opacity-100"
                        title="Remove document"
                      >
                        <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                          <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  {/* Status Footer */}
                  <div className="mt-4 pt-3 border-t border-border/40 flex items-center justify-between relative z-10">
                    <div className="flex items-center space-x-1.5">
                      {isProcessing && <Loader2 className="w-3 h-3 text-primary animate-spin" />}
                      {isIndexed && <CheckCircle2 className="w-3 h-3 text-emerald-500" />}
                      {isFailed && <AlertTriangle className="w-3 h-3 text-red-500" />}
                      <span className={cn(
                        "text-[11px] font-medium",
                        isProcessing ? "text-primary" :
                        isFailed ? "text-red-500" :
                        "text-emerald-600 dark:text-emerald-500"
                      )}>
                        {isProcessing ? "Indexing…" : isFailed ? "Failed to index" : "Indexed Successfully"}
                      </span>
                    </div>
                    {isIndexed && (
                      <span className="text-[10px] font-mono text-muted-foreground bg-surface-container px-2 py-0.5 rounded-md">
                        {doc.chunkCount || 0} chunks
                      </span>
                    )}
                  </div>

                  {/* Processing Progress Bar */}
                  {isProcessing && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-surface-container-high overflow-hidden">
                      <motion.div
                        className="h-full bg-primary"
                        initial={{ x: "-100%" }}
                        animate={{ x: "100%" }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                      />
                    </div>
                  )}
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </div>

      <ConfirmDialog
        isOpen={!!docToDelete}
        title="Delete Document?"
        description="This will permanently delete the document and all its indexed chunks. This action cannot be undone."
        confirmLabel="Delete Document"
        onConfirm={() => { if (docToDelete) onDeleteDocument(docToDelete); setDocToDelete(null); }}
        onCancel={() => setDocToDelete(null)}
      />
    </div>
  );
}
