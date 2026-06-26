import React, { useState } from "react";
import { SourceDocument, DocType } from "../types";
import { 
  FolderOpen, FileText, FileSpreadsheet, Globe, Plus, Trash2, 
  CheckCircle, Database, Clock, Sparkles, BookOpen 
} from "lucide-react";
import { cn } from "../lib/utils";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";

interface LibraryViewProps {
  documents: SourceDocument[];
  onToggleActive: (id: string) => void;
  onDeleteDocument: (id: string) => void;
  onAddDocument: (name: string, content: string, type: DocType, size: string) => void;
}

const uploadSchema = z.object({
  name: z.string().min(1, "Filename is required"),
  type: z.enum(["doc", "spreadsheet", "link", "pdf"]),
  content: z.string().min(1, "Content is required"),
});
type UploadFormValues = z.infer<typeof uploadSchema>;

export function LibraryView({
  documents,
  onToggleActive,
  onDeleteDocument,
  onAddDocument,
}: LibraryViewProps) {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);

  const { register, handleSubmit, reset, formState: { errors } } = useForm<UploadFormValues>({
    resolver: zodResolver(uploadSchema),
    defaultValues: {
      name: "",
      type: "doc",
      content: ""
    }
  });

  const uploadMutation = useMutation({
    mutationFn: async (data: UploadFormValues) => {
      const sizeStr = `${Math.round(data.content.length / 1024) || 1} KB`;
      const res = await fetch("/api/v1/documents/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...data, size: sizeStr })
      });
      if (!res.ok) throw new Error("Upload failed");
      return { ...data, sizeStr };
    },
    onMutate: (data) => {
      const sizeStr = `${Math.round(data.content.length / 1024) || 1} KB`;
      onAddDocument(data.name, data.content, data.type as DocType, sizeStr);
      reset();
    }
  });

  const onSubmit = (data: UploadFormValues) => {
    uploadMutation.mutate(data);
  };

  const selectedDoc = documents.find(d => d.id === selectedDocId) || documents[0];

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    setDragCounter(prev => prev + 1);
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragCounter(prev => prev - 1);
    if (dragCounter - 1 === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setDragCounter(0);
    // In a real implementation, you would process e.dataTransfer.files here
  };

  return (
    <div 
      className="flex-1 overflow-hidden flex select-none bg-background relative"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/40 backdrop-blur-sm z-40 pointer-events-none"
          />
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

        {/* Existing Documents Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {documents.map((doc) => (
            <div
              key={doc.id}
              onClick={() => setSelectedDocId(doc.id)}
              className={cn(
                "p-5 premium-card relative cursor-pointer glow-hover group",
                doc.active
                  ? "border-primary/30 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1)]"
                  : "",
                selectedDocId === doc.id ? "ring-1 ring-primary" : ""
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
                    {(doc.type === "pdf" || doc.type === "doc") && <FileText className="w-5 h-5" />}
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-foreground truncate max-w-[150px]">{doc.name}</h4>
                    <p className="text-[11px] text-muted-foreground font-semibold font-mono mt-0.5">{doc.size}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2" onClick={(e) => e.stopPropagation()}>
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
                  <button
                    onClick={() => {
                      onDeleteDocument(doc.id);
                      if (selectedDocId === doc.id) setSelectedDocId(null);
                    }}
                    className="p-1.5 text-muted-foreground hover:bg-red-500/10 hover:text-red-500 rounded-md transition-colors cursor-pointer"
                    title="Remove document"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Content excerpt preview */}
              <div className="mt-4 bg-background p-3 rounded-lg border border-border text-xs text-muted-foreground leading-relaxed font-mono max-h-20 overflow-hidden text-ellipsis line-clamp-3">
                {doc.content}
              </div>
            </div>
          ))}
        </div>

        {/* Ingest New Source Form */}
        <motion.div 
          animate={isDragging ? { scale: 1.02 } : { scale: 1 }}
          whileHover={!isDragging ? { scale: 1.005 } : {}}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className={cn(
            "premium-card p-6 border-2 transition-colors relative z-50",
            isDragging 
              ? "border-primary border-solid shadow-[0_0_30px_rgba(99,102,241,0.2)] bg-surface-container-high" 
              : "border-dashed border-border hover:border-primary/50"
          )}
        >
          <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-5">Ingest Knowledge Text</h3>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div className="md:col-span-2 space-y-1.5">
                <label className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold block">Document Filename</label>
                <input
                  type="text"
                  placeholder="e.g. cloud_architecture_spec.md"
                  {...register("name")}
                  className={cn(
                    "w-full bg-background border border-border rounded-lg px-4 py-2.5 text-sm focus:ring-1 focus:outline-none transition-all shadow-sm font-medium",
                    errors.name ? "border-red-500 focus:ring-red-500/20" : "focus:ring-foreground/20 text-foreground"
                  )}
                />
                {errors.name && <span className="text-[10px] text-red-500 font-medium">{errors.name.message}</span>}
              </div>
              <div className="space-y-1.5">
                <label className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold block">Format Classification</label>
                <select
                  {...register("type")}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2.5 text-sm focus:ring-1 focus:ring-foreground/20 text-foreground font-medium focus:outline-none transition-all shadow-sm cursor-pointer"
                >
                  <option value="doc">Text Document (.md, .txt)</option>
                  <option value="spreadsheet">Spreadsheet (.xlsx, .csv)</option>
                  <option value="link">Internal Web Link</option>
                  <option value="pdf">Academic PDF</option>
                </select>
                {errors.type && <span className="text-[10px] text-red-500 font-medium">{errors.type.message}</span>}
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold block">Document Content Text</label>
              <textarea
                rows={4}
                placeholder="Paste the ground truth text content here... When the researcher queries about this topic, the RAG engine will match these facts."
                {...register("content")}
                className={cn(
                  "w-full bg-background border border-border rounded-lg px-4 py-3 text-sm focus:ring-1 focus:outline-none transition-all shadow-sm font-medium",
                  errors.content ? "border-red-500 focus:ring-red-500/20" : "focus:ring-foreground/20 text-foreground"
                )}
              />
              {errors.content && <span className="text-[10px] text-red-500 font-medium">{errors.content.message}</span>}
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={uploadMutation.isPending}
                className="px-6 py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg font-semibold text-sm cursor-pointer transition-all flex items-center space-x-2 shadow-sm disabled:opacity-70"
              >
                <Plus className="w-4 h-4" />
                <span>{uploadMutation.isPending ? "Ingesting..." : "Ingest into Library"}</span>
              </button>
            </div>
          </form>
        </motion.div>
      </div>

      {/* Right Drawer Panel: Auto-Summary Card Detail */}
      {selectedDoc && (
        <div className="w-80 border-l border-border bg-surface-container-lowest h-full p-6 space-y-6 overflow-y-auto shadow-premium">
          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center">
              <Sparkles className="w-3.5 h-3.5 mr-1.5 text-primary" />
              Intelligence Auto-Summary
            </span>
            <h3 className="text-sm font-bold text-foreground leading-tight mt-1">{selectedDoc.name}</h3>
          </div>

          {/* Core summary bullets generated by Gemini */}
          <div className="bg-background p-4 rounded-xl border border-border shadow-sm space-y-4">
            <h4 className="text-[10px] uppercase font-semibold tracking-wider text-muted-foreground flex items-center">
              <BookOpen className="w-3.5 h-3.5 mr-1.5" />
              Key Findings
            </h4>
            <div className="text-[11px] text-foreground leading-relaxed space-y-2 whitespace-pre-line font-medium">
              {selectedDoc.summary || (selectedDoc as any).summary || (
                "1. Document registered in the cache list.\n2. Ingesting content words...\n3. Standby for auto-summary generation."
              )}
            </div>
          </div>

          {/* Indexing details */}
          <div className="space-y-3">
            <h4 className="text-[10px] font-semibold uppercase text-muted-foreground tracking-wider">Pipeline Metadata</h4>
            
            <div className="space-y-2 bg-background p-4 rounded-xl border border-border shadow-sm">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="font-medium flex items-center"><Database className="w-3.5 h-3.5 mr-1" /> Chunk Count:</span>
                <span className="font-mono font-medium text-foreground">{(selectedDoc as any).chunkCount || 12}</span>
              </div>
              <div className="flex items-center justify-between text-xs text-muted-foreground pt-3 border-t border-border mt-2">
                <span className="font-medium flex items-center"><Clock className="w-3.5 h-3.5 mr-1" /> Estimated Read:</span>
                <span className="font-medium text-foreground">3 mins</span>
              </div>
              <div className="flex items-center justify-between text-xs text-muted-foreground pt-3 border-t border-border mt-2">
                <span className="font-medium flex items-center"><CheckCircle className="w-3.5 h-3.5 mr-1" /> Vector Model:</span>
                <span className="font-mono text-[10px] text-foreground bg-surface border border-border px-2 py-0.5 rounded-md font-medium">
                  {(selectedDoc as any).embeddingModel || "text-embedding-3-large"}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export function SourcesView() { return null; }
export function ModelsView() { return null; }
export function AccountView() { return null; }
