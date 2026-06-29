import React from "react";
import { SourceDocument } from "../types";
import { ArrowLeft, FileText, Database, Clock, CheckCircle, FileSpreadsheet, Globe, Sparkles, Layers, Settings2, Box, AlertTriangle, Loader2 } from "lucide-react";
import { cn } from "../lib/utils";

interface DocumentDetailsViewProps {
  document: SourceDocument | undefined;
  onClose: () => void;
}

export default function DocumentDetailsView({ document, onClose }: DocumentDetailsViewProps) {
  if (!document) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-background">
        <p className="text-muted-foreground mb-4 text-sm font-medium">Document not found.</p>
        <button 
          onClick={onClose}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg shadow-sm text-sm font-semibold"
        >
          Go Back
        </button>
      </div>
    );
  }

  const isProcessing = document.status === "queued" || document.status === "indexing";
  const isFailed = document.status === "failed";
  const isIndexed = document.status === "indexed" || !document.status;

  return (
    <div className="flex-1 flex flex-col w-full h-full bg-background overflow-hidden relative">
      
      {/* Header */}
      <header className="h-14 flex items-center px-6 bg-background/80 backdrop-blur-md border-b border-border z-20 shrink-0">
        <button
          onClick={onClose}
          className="mr-4 p-1.5 rounded-md hover:bg-surface text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex items-center space-x-3">
          <div className="p-1.5 rounded bg-surface border border-border">
            {document.type === "spreadsheet" ? <FileSpreadsheet className="w-4 h-4 text-emerald-500" /> :
             document.type === "link" ? <Globe className="w-4 h-4 text-sky-500" /> :
             document.type === "pdf" ? <FileText className="w-4 h-4 text-rose-500" /> :
             <FileText className="w-4 h-4 text-amber-500" />}
          </div>
          <h2 className="text-sm font-semibold text-foreground tracking-tight">{document.name}</h2>
          
          <div className="flex items-center space-x-2 ml-4 pl-4 border-l border-border/50">
            {isProcessing && <Loader2 className="w-3.5 h-3.5 text-primary animate-spin" />}
            {isIndexed && <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />}
            {isFailed && <AlertTriangle className="w-3.5 h-3.5 text-red-500" />}
            <span className={cn(
              "text-[10px] px-2 py-0.5 rounded-full font-bold tracking-wider uppercase",
              isProcessing ? "bg-primary/10 text-primary" : 
              isFailed ? "bg-red-500/10 text-red-500" : 
              "bg-emerald-500/10 text-emerald-500"
            )}>
              {isProcessing ? "Indexing..." : isFailed ? "Failed" : "Indexed"}
            </span>
          </div>

        </div>
      </header>

      {/* Main Content: Focused Metadata Dashboard */}
      <div className="flex-1 overflow-y-auto p-6 md:p-12 flex justify-center">
        <div className="w-full max-w-4xl space-y-10">
          
          <div className="text-center space-y-2 mb-12 mt-4">
            <h1 className="text-3xl font-bold tracking-tight text-foreground">Pipeline Metadata Dashboard</h1>
            <p className="text-muted-foreground text-sm font-medium max-w-lg mx-auto">
              Detailed breakdown of ingestion, chunking, and vector embedding statistics for this document.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Metadata Card */}
            <div className="premium-card p-8 bg-surface-container-lowest">
              <h3 className="text-xs uppercase font-bold tracking-wider text-muted-foreground mb-6 flex items-center">
                <Settings2 className="w-4 h-4 mr-2" />
                Ingestion Details
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between text-sm pt-4 border-t border-border">
                  <span className="font-semibold text-muted-foreground flex items-center"><FileText className="w-4 h-4 mr-2" /> File Size</span>
                  <span className="font-mono font-bold text-foreground">{document.size}</span>
                </div>
                
                <div className="flex items-center justify-between text-sm pt-4 border-t border-border">
                  <span className="font-semibold text-muted-foreground flex items-center"><Layers className="w-4 h-4 mr-2" /> Extractor</span>
                  <span className="font-bold text-foreground">{document.type === "pdf" ? "PyPDF2" : "TextLoader"}</span>
                </div>

                <div className="flex items-center justify-between text-sm pt-4 border-t border-border">
                  <span className="font-semibold text-muted-foreground flex items-center"><CheckCircle className="w-4 h-4 mr-2" /> Embedding Model</span>
                  <span className="font-mono text-xs text-primary bg-primary/10 px-2 py-0.5 rounded font-bold">
                    {document.embeddingModel || "text-embedding-3-large"}
                  </span>
                </div>

                <div className="flex items-center justify-between text-sm pt-4 border-t border-border">
                  <span className="font-semibold text-muted-foreground flex items-center"><Database className="w-4 h-4 mr-2" /> Total Chunks</span>
                  <span className="font-mono font-bold text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded">
                    {document.chunkCount || 0}
                  </span>
                </div>
              </div>
            </div>

            {/* Chunk Viewer Summary */}
            <div className="premium-card p-8 bg-surface-container-lowest flex flex-col">
              <h3 className="text-xs uppercase font-bold tracking-wider text-muted-foreground mb-6 flex items-center">
                <Sparkles className="w-4 h-4 mr-2" />
                Intelligence Ready
              </h3>
              
              {isProcessing ? (
                <div className="flex-1 flex flex-col items-center justify-center space-y-4">
                  <Loader2 className="w-10 h-10 text-primary animate-spin" />
                  <p className="text-sm font-semibold text-muted-foreground">Generating embeddings & chunks...</p>
                </div>
              ) : isFailed ? (
                <div className="flex-1 flex flex-col items-center justify-center space-y-4">
                  <AlertTriangle className="w-10 h-10 text-red-500" />
                  <p className="text-sm font-semibold text-muted-foreground">Indexing failed.</p>
                </div>
              ) : (
                <div className="flex-1 flex flex-col justify-center">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-background p-4 rounded-xl border border-border shadow-sm text-center">
                      <Database className="w-6 h-6 text-emerald-500 mx-auto mb-2" />
                      <p className="text-2xl font-black text-foreground">{document.chunkCount || 0}</p>
                      <p className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">Vector Blocks</p>
                    </div>
                    <div className="bg-background p-4 rounded-xl border border-border shadow-sm text-center">
                      <Box className="w-6 h-6 text-sky-500 mx-auto mb-2" />
                      <p className="text-2xl font-black text-foreground">Yes</p>
                      <p className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">BM25 Indexed</p>
                    </div>
                  </div>
                  <div className="mt-6 bg-primary/5 border border-primary/20 rounded-xl p-4 flex items-start space-x-3">
                    <Sparkles className="w-5 h-5 text-primary shrink-0" />
                    <p className="text-xs text-foreground font-medium leading-relaxed">
                      This document is now fully parsed and loaded into the vector database. It is ready for semantic and keyword queries by the AI researcher.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}