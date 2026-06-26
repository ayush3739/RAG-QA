import React, { useState } from "react";
import { 
  Search, Sliders, Sparkles, FileText, ArrowRight, Copy, Share, Send, ChevronRight, Globe
} from "lucide-react";
import { SourceDocument, Citation } from "../types";
import { motion, Variants } from "framer-motion";
import { cn } from "../lib/utils";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const reportContainerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.05 }
  }
};

const reportItemVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { type: "spring", stiffness: 300, damping: 25 } 
  }
};

interface ResearchViewProps {
  documents: SourceDocument[];
  onTakeToChat: (topic: string, selectedDocIds: string[]) => void;
}

interface ResearchReport {
  summary: string;
  key_findings: string[];
  sources: Citation[];
  confidence: number;
  tool_trace: string[];
  follow_up_questions: string[];
}

const researchSchema = z.object({
  topic: z.string().min(1, "Topic is required"),
  includeWeb: z.boolean(),
  outputFormat: z.enum(["structured", "bullet"]),
  selectedDocIds: z.array(z.string()).min(1, "Select at least one document for grounding")
});
type ResearchFormValues = z.infer<typeof researchSchema>;

export default function ResearchView({ documents, onTakeToChat }: ResearchViewProps) {
  const [report, setReport] = useState<ResearchReport | null>(null);

  const { register, handleSubmit, setValue, watch, formState: { errors } } = useForm<ResearchFormValues>({
    resolver: zodResolver(researchSchema),
    defaultValues: {
      topic: "",
      includeWeb: true,
      outputFormat: "structured",
      selectedDocIds: []
    }
  });

  const selectedDocIds = watch("selectedDocIds");
  const topic = watch("topic");
  const outputFormat = watch("outputFormat");
  const includeWeb = watch("includeWeb");

  const handleToggleDoc = (id: string) => {
    const next = selectedDocIds.includes(id) ? selectedDocIds.filter(i => i !== id) : [...selectedDocIds, id];
    setValue("selectedDocIds", next, { shouldValidate: true });
  };

  const researchMutation = useMutation({
    mutationFn: async (data: ResearchFormValues) => {
      const collection = data.selectedDocIds.length > 0 ? data.selectedDocIds[0] : undefined;
      const res = await fetch("/api/v1/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: data.topic.trim(),
          collection,
          include_web: data.includeWeb,
          output_format: data.outputFormat
        })
      });
      if (!res.ok) {
        throw new Error(`Failed to generate report: status ${res.status}`);
      }
      return res.json();
    },
    onSuccess: (data) => {
      setReport(data);
    },
    onError: (err) => {
      console.error(err);
      alert("Workspace Research Refused: Please ensure API is healthy.");
    }
  });

  const onSubmit = (data: ResearchFormValues) => {
    setReport(null);
    researchMutation.mutate(data);
  };

  const handleCopyJson = () => {
    if (!report) return;
    navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    alert("JSON payload copied to clipboard.");
  };

  const handleExportMarkdown = () => {
    if (!report) return;
    const md = `
# DocuMind Synthesis Report: ${topic}
Confidence Score: ${report.confidence * 100}%
Tool Trace: ${report.tool_trace.join(" -> ")}

## Executive Summary
${report.summary}

## Key Grounding Findings
${report.key_findings.map((f, i) => `${i + 1}. ${f}`).join("\n")}

## Follow-up Questions
${report.follow_up_questions.map(q => `- ${q}`).join("\n")}
`;
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `documind_report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const isResearching = researchMutation.isPending;

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-12 py-10 space-y-10 bg-background selection:bg-primary/10">
      
      {/* Title */}
      <div className="max-w-5xl mx-auto">
        <h2 className="text-2xl md:text-3xl font-bold text-foreground flex items-center tracking-tight">
          <Search className="w-5 h-5 mr-3 text-muted-foreground" />
          Structured Research
        </h2>
        <p className="text-sm text-muted-foreground mt-1.5 max-w-2xl font-medium">
          Perform a one-shot vector-grounded analysis across your custom knowledge library. The system synthesizes cross-referenced documents with full citation mapping.
        </p>
      </div>

      <div className="max-w-[1200px] mx-auto flex flex-col lg:flex-row gap-8">
        
        {/* Left Column: Form Setup (1/3) */}
        <div className="w-full lg:w-80 flex-shrink-0 bg-surface-container-lowest p-6 rounded-2xl border border-border shadow-premium h-fit space-y-6">
          <h3 className="text-xs uppercase font-semibold tracking-wider text-muted-foreground flex items-center">
            <Sliders className="w-3.5 h-3.5 mr-2" />
            Parameters
          </h3>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            
            {/* Topic Input */}
            <div className="space-y-2">
              <label className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground block">
                Inquiry Topic
              </label>
              <textarea
                {...register("topic")}
                rows={3}
                placeholder="E.g., What are the main performance specifications of the system?"
                className={cn(
                  "w-full text-sm font-medium bg-background border rounded-lg px-3 py-2 text-foreground focus:outline-none transition-all shadow-sm resize-none",
                  errors.topic ? "border-red-500 focus:ring-1 focus:ring-red-500/20" : "border-border focus:ring-1 focus:ring-foreground/20"
                )}
              />
              {errors.topic && <span className="text-[10px] text-red-500 font-medium">{errors.topic.message}</span>}
            </div>

            {/* Document Multi-Selector Checklist */}
            <div className="space-y-2">
              <label className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground block">
                Library Grounds
              </label>
              <div className="max-h-48 overflow-y-auto space-y-2 pr-1">
                {documents.map((doc) => (
                  <div 
                    key={doc.id}
                    onClick={() => handleToggleDoc(doc.id)}
                    className={cn(
                      "p-2.5 rounded-lg border text-sm font-medium flex items-center justify-between cursor-pointer transition-colors shadow-sm",
                      selectedDocIds.includes(doc.id)
                        ? "bg-primary/5 border-primary text-foreground"
                        : "bg-surface border-border text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <span className="truncate max-w-[160px]">{doc.name}</span>
                    <input 
                      type="checkbox" 
                      checked={selectedDocIds.includes(doc.id)}
                      readOnly
                      className="accent-foreground w-3.5 h-3.5 rounded cursor-pointer"
                    />
                  </div>
                ))}
                {documents.length === 0 && (
                  <p className="text-[11px] text-muted-foreground italic">No documents available.</p>
                )}
              </div>
              {errors.selectedDocIds && <span className="text-[10px] text-red-500 font-medium">{errors.selectedDocIds.message}</span>}
            </div>

            {/* Web Search backup toggle */}
            <div className="flex items-center justify-between pt-4 border-t border-border mt-2">
              <div className="flex items-center space-x-2">
                <Globe className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs font-semibold text-foreground">Web Search Backup</span>
              </div>
              <input
                type="checkbox"
                {...register("includeWeb")}
                className="accent-foreground w-3.5 h-3.5 cursor-pointer"
              />
            </div>

            {/* Output format segmented selectors */}
            <div className="space-y-2 pt-2">
              <label className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground block">
                Format
              </label>
              <div className="grid grid-cols-2 gap-1 bg-surface p-1 rounded-lg border border-border shadow-sm">
                <button
                  type="button"
                  onClick={() => setValue("outputFormat", "structured")}
                  className={cn(
                    "py-1.5 rounded-md text-[11px] font-semibold transition-all",
                    outputFormat === "structured" ? "bg-background text-foreground shadow-sm border border-border" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Prose
                </button>
                <button
                  type="button"
                  onClick={() => setValue("outputFormat", "bullet")}
                  className={cn(
                    "py-1.5 rounded-md text-[11px] font-semibold transition-all",
                    outputFormat === "bullet" ? "bg-background text-foreground shadow-sm border border-border" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Bullets
                </button>
              </div>
            </div>

            {/* Trigger Button */}
            <button
              type="submit"
              disabled={isResearching}
              className={cn(
                "w-full py-2.5 rounded-lg flex items-center justify-center space-x-2 text-xs font-semibold transition-all shadow-sm",
                isResearching ? "bg-surface text-muted-foreground border border-border cursor-not-allowed" : "bg-primary text-primary-foreground cursor-pointer hover:bg-primary/90"
              )}
            >
              <span>{isResearching ? "Synthesizing..." : "Execute Synthesis"}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>

          </form>
        </div>

        {/* Right Column: Dynamic Report output results or loading skeleton (2/3) */}
        <div className="flex-1 space-y-6 max-w-[840px] mx-auto w-full">
          
          {/* Skeleton state */}
          {isResearching && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-surface-container-lowest border border-border rounded-2xl p-8 space-y-6 shadow-sm">
              <div className="h-4 skeleton-shimmer-bar rounded-md w-1/3"></div>
              <div className="h-24 skeleton-shimmer-bar rounded-md w-full"></div>
              <div className="h-6 skeleton-shimmer-bar rounded-md w-1/4"></div>
              <div className="space-y-3">
                <div className="h-10 skeleton-shimmer-bar rounded-md w-full"></div>
                <div className="h-10 skeleton-shimmer-bar rounded-md w-full"></div>
                <div className="h-10 skeleton-shimmer-bar rounded-md w-full"></div>
              </div>
            </motion.div>
          )}

          {/* Idle Empty Welcome State */}
          {!report && !isResearching && (
            <div className="p-12 flex flex-col items-center justify-center text-center border border-dashed border-border rounded-2xl bg-surface-container-lowest/50">
              <Sparkles className="w-6 h-6 text-muted-foreground mb-3" />
              <h4 className="font-semibold text-foreground text-sm tracking-tight">Synthesis Output Standby</h4>
              <p className="text-xs text-muted-foreground mt-2 max-w-xs">
                Configure your prompt settings on the left panel, and click "Execute Synthesis" to generate a report.
              </p>
            </div>
          )}

          {/* Realized Report Panel */}
          {report && !isResearching && (
            <motion.div 
              variants={reportContainerVariants} 
              initial="hidden" 
              animate="visible" 
              className="bg-surface-container-lowest border border-border rounded-2xl p-8 space-y-8 shadow-premium"
            >
              
              {/* Header metadata row */}
              <motion.div variants={reportItemVariants} className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-border">
                <div>
                  <span className="text-[10px] uppercase font-semibold tracking-wider text-muted-foreground block">Topic</span>
                  <h4 className="text-lg font-bold text-foreground mt-1 truncate max-w-md tracking-tight">{topic}</h4>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-[10px] bg-background border border-border text-foreground font-semibold px-2 py-1 rounded-md uppercase">
                    {Math.round(report.confidence * 100)}% Confidence
                  </span>
                  <span className="text-[10px] font-mono text-muted-foreground font-semibold bg-surface border border-border px-2 py-1 rounded-md">
                    {report.tool_trace.join(" + ")}
                  </span>
                </div>
              </motion.div>

              {/* Prose Report Summary */}
              <motion.div variants={reportItemVariants} className="space-y-2">
                <h4 className="text-[11px] font-semibold uppercase text-muted-foreground tracking-wider">Executive Summary</h4>
                <div className="text-[14px] text-foreground leading-relaxed prose prose-sm max-w-none dark:prose-invert prose-p:leading-relaxed prose-pre:bg-surface prose-pre:border prose-pre:border-border">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{report.summary}</ReactMarkdown>
                </div>
              </motion.div>

              {/* Key Findings List */}
              <motion.div variants={reportItemVariants} className="space-y-3">
                <h4 className="text-[11px] font-semibold uppercase text-muted-foreground tracking-wider">Grounded Takeaways</h4>
                <div className="grid grid-cols-1 gap-2">
                  {report.key_findings.map((finding, idx) => (
                    <div key={idx} className="p-3 bg-surface border border-border rounded-lg flex items-start space-x-3">
                      <span className="text-[10px] font-mono font-semibold text-muted-foreground bg-background border border-border w-5 h-5 rounded flex items-center justify-center flex-shrink-0 mt-0.5">
                        {idx + 1}
                      </span>
                      <p className="text-[13px] font-medium text-foreground leading-relaxed">{finding}</p>
                    </div>
                  ))}
                </div>
              </motion.div>

              {/* Grounding Citations */}
              {report.sources && report.sources.length > 0 && (
                <motion.div variants={reportItemVariants} className="space-y-3">
                  <h4 className="text-[11px] font-semibold uppercase text-muted-foreground tracking-wider">Sources</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {report.sources.map((src, sIdx) => (
                      <div key={sIdx} className="p-3 bg-background border border-border rounded-lg space-y-2 glow-hover group cursor-default">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-foreground truncate max-w-[150px] uppercase flex items-center space-x-2 relative z-10">
                            <FileText className="w-3 h-3 mr-1 text-muted-foreground" />
                            {src.name}
                          </span>
                          <span className="text-[10px] bg-surface border border-border text-foreground font-semibold px-1.5 py-0.5 rounded-md relative z-10">
                            {src.fitScore}% Match
                          </span>
                        </div>
                        <p className="text-[11px] text-muted-foreground font-mono leading-relaxed truncate relative z-10">
                          {src.snippet}
                        </p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Follow up questions chips */}
              {report.follow_up_questions && report.follow_up_questions.length > 0 && (
                <motion.div variants={reportItemVariants} className="space-y-3 pt-6 border-t border-border">
                  <h4 className="text-[11px] font-semibold uppercase text-muted-foreground tracking-wider">Recommended Next Inquiries</h4>
                  <div className="flex flex-wrap gap-2">
                    {report.follow_up_questions.map((q, qIdx) => (
                      <button
                        key={qIdx}
                        onClick={() => setValue("topic", q)}
                        className="bg-background hover:bg-surface border border-border px-2.5 py-1.5 rounded-md text-[11px] font-medium text-foreground cursor-pointer transition-colors flex items-center"
                      >
                        <ChevronRight className="w-3 h-3 mr-1 text-muted-foreground" />
                        <span>{q}</span>
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Action Rows */}
              <motion.div variants={reportItemVariants} className="flex flex-wrap items-center justify-between gap-4 pt-6 border-t border-border">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleCopyJson}
                    className="px-3 py-1.5 bg-surface hover:bg-surface-container-low text-foreground rounded-md font-medium text-[11px] cursor-pointer transition-colors flex items-center space-x-1.5 border border-border"
                  >
                    <Copy className="w-3.5 h-3.5 text-muted-foreground" />
                    <span>Copy JSON</span>
                  </button>
                  <button
                    onClick={handleExportMarkdown}
                    className="px-3 py-1.5 bg-surface hover:bg-surface-container-low text-foreground rounded-md font-medium text-[11px] cursor-pointer transition-colors flex items-center space-x-1.5 border border-border"
                  >
                    <Share className="w-3.5 h-3.5 text-muted-foreground" />
                    <span>Export MD</span>
                  </button>
                </div>

                <button
                  onClick={() => onTakeToChat(topic, selectedDocIds)}
                  className="px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-md font-medium text-xs cursor-pointer transition-colors flex items-center space-x-2 shadow-sm"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>Migrate to Chat</span>
                </button>
              </motion.div>

            </motion.div>
          )}

        </div>

      </div>

    </div>
  );
}
