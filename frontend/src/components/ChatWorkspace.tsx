import React, { useState, useRef, useEffect } from "react";
import { Message, SourceDocument, Citation } from "../types";
import { 
  Sparkles, Paperclip, FileText, FileSpreadsheet, 
  PanelRightClose, PanelRightOpen, BrainCircuit, Layers, Settings2, 
  CheckCircle2, Search, ArrowRight, ArrowDown, Plus, Loader2, AlertTriangle, X,
  Copy, ThumbsUp, ThumbsDown, Check, RotateCcw, ChevronDown, Database
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../lib/utils";
import { api } from "../lib/api";
import { toast } from "sonner";
import FeedbackDialog from "./FeedbackDialog";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const EMPTY_DOCS: SourceDocument[] = [];
const STARTING_GRAPH_STATUS = [{ node: "main_router", label: "Deciding route", status: "running" }];

function CodeBlock({ code, language }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard?.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };

  return (
    <div className="my-4 w-full max-w-full overflow-hidden rounded-xl border border-border bg-surface-container-lowest">
      <div className="flex items-center justify-between border-b border-border px-3.5 py-1.5 bg-surface-container">
        <span className="text-xs font-medium text-muted-foreground">
          {language || "code"}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
          title="Copy code"
          aria-label="Copy code"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 text-xs font-mono leading-6 text-foreground bg-surface-container-lowest">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export const cleanMessageText = (text: string): string => {
  if (!text) return "";
  return text
    // Strip bracketed hex/hash citations e.g. 【67bb9d91487d4bbe】
    .replace(/【[a-zA-Z0-9_-]{6,}】/g, "")
    // Strip [chunk_id=...]
    .replace(/\[chunk_id=[^\]]+\]/g, "")
    // Strip raw 32-char hex filenames if output literally in text e.g. 27c46b7e13c349d18a158d7b50e05bc7.pdf
    .replace(/\b[a-f0-9]{32}\.(?:pdf|docx|txt|md)\b/gi, "")
    // Clean up spaces before punctuation
    .replace(/ +([.,;:?!])/g, "$1");
};

const markdownCache = new Map<string, string>();

const formatAssistantMarkdown = (text: string) => {
  if (!text) return "";
  if (markdownCache.has(text)) {
    return markdownCache.get(text)!;
  }

  let inFence = false;
  let sectionIndex = 0;
  const cleanedText = cleanMessageText(text);

  const formatted = cleanedText
    .split("\n")
    .map((line) => {
      const trimmed = line.trim();
      if (trimmed.startsWith("```")) {
        inFence = !inFence;
        return line;
      }
      if (inFence || !trimmed) return line;

      const boldLeadIn = trimmed.match(/^\*\*([^*:]{3,90}):\*\*\s*(.*)$/);
      if (boldLeadIn) {
        sectionIndex += 1;
        return boldLeadIn[2]
          ? `\n### ${sectionIndex}. ${boldLeadIn[1]}\n\n${boldLeadIn[2]}`
          : `\n### ${sectionIndex}. ${boldLeadIn[1]}`;
      }

      const labelLine = trimmed.match(/^([A-Z][A-Za-z0-9/() ,.&'"-]{2,90}):\s*(.*)$/);
      if (labelLine && labelLine[1].includes(" ")) {
        sectionIndex += 1;
        return labelLine[2]
          ? `\n### ${sectionIndex}. ${labelLine[1]}\n\n${labelLine[2]}`
          : `\n### ${sectionIndex}. ${labelLine[1]}`;
      }

      if (/^(conclusion|summary|overview)$/i.test(trimmed)) {
        return `\n### ${trimmed}`;
      }
      return line;
    })
    .join("\n");

  if (markdownCache.size > 200) {
    markdownCache.clear();
  }
  markdownCache.set(text, formatted);
  return formatted;
};

interface ChatWorkspaceProps {
  messages: Message[];
  isProcessing: boolean;
  onSendMessage: (text: string, mode?: "auto" | "research") => void;
  documents: SourceDocument[];
  allDocuments?: SourceDocument[];
  activeConvId?: string;
  onAddDocument: (file: File) => void;
  onLinkDocument?: (documentId: string) => void;
  onUnlinkDocument?: (documentId: string) => void;
  onRetryMessage?: (messageId: string, mode?: "auto" | "research") => void;
  selectedModel: string;
  sessionTitle?: string;
  graphStatuses?: any[];
}

export default function ChatWorkspace({
  messages,
  isProcessing,
  onSendMessage,
  documents,
  allDocuments = EMPTY_DOCS,
  activeConvId,
  onAddDocument,
  onLinkDocument,
  onUnlinkDocument,
  onRetryMessage,
  selectedModel,
  sessionTitle = "New Research Session",
  graphStatuses = [],
}: ChatWorkspaceProps) {
  const [inputText, setInputText] = useState("");
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(true);
  const [chatMode, setChatMode] = useState<"auto" | "research">("auto");
  const [isLinkDropdownOpen, setIsLinkDropdownOpen] = useState(false);
  
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [feedbacks, setFeedbacks] = useState<Record<string, 'up' | 'down'>>({});
  const [activeFeedbackMsg, setActiveFeedbackMsg] = useState<{ id: string; dbId: number; rating: number; type: 'up' | 'down' } | null>(null);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});

  const toggleMessageSources = (msgId: string) => {
    setExpandedSources((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  };

  const resolveCitationName = (cite: any): string => {
    if (cite.type === "web") return cite.title || cite.url || "Web Source";
    const rawName = cite.name || cite.title || "";
    if (/^[a-f0-9]{20,}\./i.test(rawName)) {
      const doc = documents.find((d) => d.id === cite.id || rawName.includes(d.id)) || (documents.length === 1 ? documents[0] : null);
      if (doc?.name) return doc.name;
      const allDoc = allDocuments.find((d) => d.id === cite.id || rawName.includes(d.id));
      if (allDoc?.name) return allDoc.name;
    }
    return rawName || "Document";
  };
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldStickToBottomRef = useRef(true);

  const visibleGraphStatuses = graphStatuses.length > 0
    ? graphStatuses.reduce((items: any[], status: any) => {
        const key = status.node || status.label || status.message;
        const existingIdx = items.findIndex((item) => (item.node || item.label || item.message) === key);
        if (existingIdx === -1) return [...items, status];
        return items.map((item, idx) => idx === existingIdx ? { ...item, ...status } : item);
      }, [])
    : STARTING_GRAPH_STATUS;

  const handleChatScroll = () => {
    const scroller = chatScrollRef.current;
    if (!scroller) return;
    const distanceFromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    const isNearBottom = distanceFromBottom < 140;
    shouldStickToBottomRef.current = isNearBottom;
    setShowJumpToLatest(!isNearBottom);
  };

  const scrollToLatest = () => {
    shouldStickToBottomRef.current = true;
    setShowJumpToLatest(false);
    scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  };

  useEffect(() => {
    if (shouldStickToBottomRef.current) {
      requestAnimationFrame(() => {
        scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      });
      setShowJumpToLatest(false);
    } else {
      setShowJumpToLatest(true);
    }
  }, [messages, isProcessing, graphStatuses]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!inputText.trim() || isProcessing) return;
    onSendMessage(inputText.trim(), chatMode);
    setInputText("");
  };

  const fallbackCopyText = (id: string, text: string) => {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      const successful = document.execCommand('copy');
      if (successful) {
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 1000);
      }
    } catch (err) {
      console.error("Fallback copy failed: ", err);
    }
    document.body.removeChild(textArea);
  };

  const handleCopy = (id: string, text: string) => {
    const cleaned = cleanMessageText(text);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(cleaned)
        .then(() => {
          setCopiedId(id);
          setTimeout(() => setCopiedId(null), 1000);
        })
        .catch(() => {
          fallbackCopyText(id, cleaned);
        });
    } else {
      fallbackCopyText(id, cleaned);
    }
  };

  const handleFeedback = (id: string, type: 'up' | 'down') => {
    const msg = messages.find(m => m.id === id);
    if (!msg) return;
    if (!msg.dbId) {
      return;
    }
    setActiveFeedbackMsg({
      id: msg.id,
      dbId: msg.dbId,
      rating: type === 'up' ? 5 : 2,
      type
    });
  };

  const handleFeedbackSubmit = async (rating: number, comment: string) => {
    if (!activeFeedbackMsg) return;
    const msgIndex = messages.findIndex(m => m.id === activeFeedbackMsg.id);
    const prevMsg = msgIndex > 0 ? messages[msgIndex - 1] : null;
    const queryText = prevMsg?.text || "Workspace query";
    const answerText = messages.find(m => m.id === activeFeedbackMsg.id)?.text || "";

    try {
      await api.submitFeedback({
        message_id: activeFeedbackMsg.dbId,
        query: queryText,
        answer: answerText,
        rating,
        comment,
        confidence: 0.9,
        tool_used: "RAG Pipeline"
      });
      setFeedbacks(prev => ({ ...prev, [activeFeedbackMsg.id]: activeFeedbackMsg.type }));
    } catch (err) {
      console.error("Failed to submit feedback", err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size === 0) {
      toast.error(`Upload failed: File ${file.name} is empty.`);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    if (file.size > 30 * 1024 * 1024) {
      toast.error(`Upload failed: File size for ${file.name} exceeds the 30 MB limit.`);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const allowed = ['.pdf', '.txt', '.md', '.docx'];
    if (file.name.includes('.') && !allowed.includes(ext)) {
      toast.error(`Upload failed: Unsupported file extension '${ext}'. Allowed extensions: ${allowed.join(', ')}`);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    onAddDocument(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="flex-1 flex w-full h-full overflow-hidden relative bg-background">
      
      {/* Main Chat Workstation Stage */}
      <div
        className={cn(
          "flex flex-col min-w-0 min-h-0 relative",
          isInspectorOpen ? "w-[calc(100%-20rem)] md:w-[calc(100%-24rem)] shrink-0" : "flex-1"
        )}
      >
        {/* Workstation Top Navigation Bar */}
        <header className="h-14 shrink-0 flex justify-between items-center px-6 bg-background border-b border-border z-20">
          <div className="flex items-center gap-3 min-w-0">
            <span className="text-sm font-semibold text-foreground tracking-tight truncate max-w-[280px] sm:max-w-md">
              {sessionTitle}
            </span>
            <span className="hidden sm:inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-surface-container border border-border text-xs text-muted-foreground shrink-0">
              <Paperclip className="w-3 h-3" />
              <span>{documents.length} sources in context</span>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsInspectorOpen(!isInspectorOpen)}
              className={cn(
                "px-2.5 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 text-xs font-semibold border active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transform-none",
                isInspectorOpen 
                  ? "bg-surface-container border-border text-foreground" 
                  : "bg-surface border-border/70 text-muted-foreground hover:text-foreground hover:border-border"
              )}
              title={isInspectorOpen ? "Close source inspector" : "Open source inspector"}
              aria-label={isInspectorOpen ? "Close source inspector" : "Open source inspector"}
            >
              <span>Inspector</span>
              {isInspectorOpen ? <PanelRightClose className="w-3.5 h-3.5" /> : <PanelRightOpen className="w-3.5 h-3.5" />}
            </button>
          </div>
        </header>

        {/* Chat Feed */}
        <div
          ref={chatScrollRef}
          onScroll={handleChatScroll}
          className="flex-1 overflow-y-auto overflow-x-hidden px-4 md:px-10 pt-6 pb-4 flex flex-col items-center min-h-0"
        >
          <div className={cn("w-full min-w-0 space-y-10", isInspectorOpen ? "max-w-3xl" : "max-w-4xl")}>
            
            {messages.length === 0 && (
              <motion.div 
                initial={{ opacity: 0, y: 8 }} 
                animate={{ opacity: 1, y: 0 }} 
                transition={{ duration: 0.3 }}
                className="select-none my-8 text-center flex flex-col items-center w-full"
              >
                <div className="w-12 h-12 rounded-xl bg-surface-container border border-border flex items-center justify-center mb-4">
                  <BrainCircuit className="w-6 h-6 text-primary" />
                </div>
                <h2 className="text-xl font-bold text-foreground tracking-tight mb-1.5">
                  KnowVex Research Workstation
                </h2>
                <p className="text-muted-foreground text-xs max-w-md text-center mb-8 leading-relaxed">
                  Query your indexed documents, synthesize evidence, and execute multi-hop RAG retrievals.
                </p>

                {/* Technical Workstation Prompt Launchers */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-xl">
                  {[
                    {
                      title: "Summarize active context",
                      desc: "Extract key findings and generate an executive brief",
                      prompt: "Please write a comprehensive executive summary of the primary topics and key findings in the active documents.",
                      icon: FileText,
                    },
                    {
                      title: "Verify grounding claims",
                      desc: "Cross-reference facts against cited source passages",
                      prompt: "Identify the critical evidence in the indexed documents that supports the following claim: ",
                      icon: Search,
                    },
                    {
                      title: "Compare source materials",
                      desc: "Extract side-by-side contrasts between indexed files",
                      prompt: "Compare the core methodologies and identify any contrasting points between the active documents.",
                      icon: Layers,
                    },
                    {
                      title: "Draft verification script",
                      desc: "Generate verification logic based on retrieved rules",
                      prompt: "Write a python script to run verification assertions on retrieved RAG contexts.",
                      icon: Settings2,
                    }
                  ].map((item, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setInputText(item.prompt)}
                      className="flex items-start p-3.5 rounded-xl border border-border bg-surface-container-lowest hover:border-primary/40 hover:bg-surface-container/60 transition-colors duration-150 text-left group cursor-pointer active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 motion-reduce:transform-none"
                    >
                      <div className="w-7 h-7 rounded-lg bg-surface-container border border-border flex items-center justify-center shrink-0 mr-3 text-muted-foreground group-hover:text-foreground transition-colors">
                        <item.icon className="w-3.5 h-3.5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="text-xs font-semibold text-foreground truncate">{item.title}</h3>
                        <p className="text-[11px] text-muted-foreground mt-0.5 line-clamp-1">{item.desc}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Conversation Flow */}
            <div className="space-y-8">
              {messages.map((msg, index) => {
                const isLastMessage = index === messages.length - 1;
                const isStreaming = isProcessing && isLastMessage && msg.sender === "assistant";

                return (
                  <div key={msg.id} className="w-full flex flex-col">
                    {msg.sender === "user" ? (
                      <div className="flex flex-col items-end w-full">
                        <div className="max-w-[85%] rounded-2xl bg-surface-container border border-border/80 px-4 py-3 text-sm text-foreground">
                          <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                        </div>
                        <div className="flex items-center gap-1 mt-1 mr-1">
                          <button
                            type="button"
                            onClick={() => handleCopy(msg.id, msg.text)}
                            className="p-1 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
                            title="Copy message"
                            aria-label="Copy user message"
                          >
                            {copiedId === msg.id ? <Check className="w-3.5 h-3.5 text-success" /> : <Copy className="w-3.5 h-3.5" />}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start gap-3.5 w-full min-w-0">
                        <div className="w-7 h-7 rounded-lg bg-surface-container border border-border flex items-center justify-center text-primary shrink-0 mt-0.5">
                          <Sparkles className="w-3.5 h-3.5" />
                        </div>
                        
                        <div className={cn(
                          "min-w-0 flex-1 flex flex-col space-y-2",
                          isStreaming && "border-l-2 border-l-primary/40 pl-3"
                        )}>
                          <div className="max-w-none min-w-0 text-foreground">
                            {msg.text ? (
                              <>
                                <ReactMarkdown
                                  remarkPlugins={[remarkGfm]}
                                  components={{
                                    a: ({ href, children }) => (
                                      <a href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2 break-words">
                                        {children}
                                      </a>
                                    ),
                                    p: ({ children }) => (
                                      <p className="my-3.5 break-words text-sm leading-7 text-foreground">
                                        {children}
                                      </p>
                                    ),
                                    h2: ({ children }) => (
                                      <h2 className="mt-7 mb-3 text-lg font-bold tracking-tight text-foreground">
                                        {children}
                                      </h2>
                                    ),
                                    h3: ({ children }) => (
                                      <h3 className="mt-5 mb-2 text-sm font-semibold tracking-tight text-foreground">
                                        {children}
                                      </h3>
                                    ),
                                    ul: ({ children }) => (
                                      <ul className="my-3.5 list-disc space-y-1.5 pl-6 text-sm">
                                        {children}
                                      </ul>
                                    ),
                                    ol: ({ children }) => (
                                      <ol className="my-3.5 list-decimal space-y-1.5 pl-6 text-sm">
                                        {children}
                                      </ol>
                                    ),
                                    li: ({ children }) => (
                                      <li className="break-words pl-1 text-sm leading-6 text-foreground">
                                        {children}
                                      </li>
                                    ),
                                    strong: ({ children }) => (
                                      <strong className="font-semibold text-foreground">
                                        {children}
                                      </strong>
                                    ),
                                    code: ({ className, children, ...props }: any) => {
                                      const code = String(children).replace(/\n$/, "");
                                      const language = /language-(\w+)/.exec(className || "")?.[1];
                                      const isBlock = Boolean(className) || code.includes("\n");
                                      if (isBlock) {
                                        return <CodeBlock code={code} language={language} />;
                                      }
                                      return (
                                        <code className="rounded-md bg-surface-container border border-border px-1.5 py-0.5 text-xs font-mono font-medium text-foreground" {...props}>
                                          {children}
                                        </code>
                                      );
                                    },
                                    pre: ({ children }) => <>{children}</>,
                                  }}
                                >
                                  {formatAssistantMarkdown(msg.text)}
                                </ReactMarkdown>
                                {isStreaming && <span className="streaming-cursor">▋</span>}
                              </>
                            ) : (
                              <div className="flex items-center space-x-1 mt-1 dot-wave text-muted-foreground">
                                <span className="bg-current"></span>
                                <span className="bg-current"></span>
                                <span className="bg-current"></span>
                              </div>
                            )}
                          </div>
                        
                          {/* Grounding Citations Drawer */}
                          {msg.citations && msg.citations.length > 0 && (
                            <div className="mt-3 pt-3 border-t border-border/50">
                              <button
                                type="button"
                                onClick={() => toggleMessageSources(msg.id)}
                                className={cn(
                                  "inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer border active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 motion-reduce:transform-none",
                                  expandedSources[msg.id]
                                    ? "bg-primary/10 text-primary border-primary/30"
                                    : "bg-surface hover:bg-surface-container text-muted-foreground hover:text-foreground border-border/70 hover:border-border"
                                )}
                                title={expandedSources[msg.id] ? "Collapse sources" : "View cited sources"}
                              >
                                <Layers className="w-3.5 h-3.5 text-primary" />
                                <span className="font-semibold text-foreground">
                                  {msg.citations.length} {msg.citations.length === 1 ? "Source" : "Sources"}
                                </span>
                                <span className="text-[11px] text-muted-foreground">
                                  {expandedSources[msg.id] ? "Click to hide" : "Click to view"}
                                </span>
                                <ChevronDown
                                  className={cn(
                                    "w-3.5 h-3.5 transition-transform duration-200 text-muted-foreground",
                                    expandedSources[msg.id] && "rotate-180 text-primary"
                                  )}
                                />
                              </button>

                              <AnimatePresence>
                                {expandedSources[msg.id] && (
                                  <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                    transition={{ duration: 0.18, ease: "easeOut" }}
                                    className="overflow-hidden"
                                  >
                                    <div className="flex flex-wrap gap-2 pt-2.5">
                                      {msg.citations.map((cite: any, cIdx) => {
                                        const isWeb = cite.type === "web";
                                        const name = resolveCitationName(cite);
                                        const badge = isWeb ? "Web" : cite.page != null ? `Page ${cite.page}` : "Doc";
                                        return (
                                          <button
                                            key={cIdx}
                                            type="button"
                                            onClick={() => setActiveCitation({ ...cite, name })}
                                            className="bg-surface border border-border hover:border-primary/40 hover:bg-surface-container px-2.5 py-1 rounded-md flex items-center space-x-2 transition-colors cursor-pointer text-left group active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 motion-reduce:transform-none"
                                          >
                                            {isWeb ? (
                                              <Search className="w-3 h-3 text-muted-foreground group-hover:text-foreground" />
                                            ) : name.endsWith(".xlsx") || name.endsWith(".csv") ? (
                                              <FileSpreadsheet className="w-3 h-3 text-muted-foreground group-hover:text-foreground" />
                                            ) : (
                                              <FileText className="w-3 h-3 text-muted-foreground group-hover:text-foreground" />
                                            )}
                                            <span className="text-xs font-medium text-foreground truncate max-w-[140px]">
                                              {name}
                                            </span>
                                            <span className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-semibold">
                                              {badge}
                                            </span>
                                          </button>
                                        );
                                      })}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </div>
                          )}

                          {/* Assistant Message Actions Toolbar */}
                          {msg.text && (
                            <div className="flex items-center space-x-1 mt-2">
                              <button
                                type="button"
                                onClick={() => handleCopy(msg.id, msg.text)}
                                className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
                                title="Copy answer"
                                aria-label="Copy assistant answer"
                              >
                                {copiedId === msg.id ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
                              </button>
                              <button
                                type="button"
                                onClick={() => handleFeedback(msg.id, 'up')}
                                className={cn(
                                  "p-1.5 rounded-md transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40",
                                  feedbacks[msg.id] === 'up' ? "text-primary bg-primary/10" : "text-muted-foreground hover:text-foreground hover:bg-surface"
                                )}
                                title="Accurate response"
                                aria-label="Mark accurate response"
                              >
                                <ThumbsUp className="w-4 h-4" />
                              </button>
                              <button
                                type="button"
                                onClick={() => handleFeedback(msg.id, 'down')}
                                className={cn(
                                  "p-1.5 rounded-md transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40",
                                  feedbacks[msg.id] === 'down' ? "text-danger bg-danger-subtle" : "text-muted-foreground hover:text-foreground hover:bg-surface"
                                )}
                                title="Needs improvement"
                                aria-label="Mark response needs improvement"
                              >
                                <ThumbsDown className="w-4 h-4" />
                              </button>
                              {onRetryMessage && (
                                <button
                                  type="button"
                                  onClick={() => onRetryMessage(msg.id, chatMode)}
                                  className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-surface transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
                                  title="Regenerate response"
                                  aria-label="Regenerate response"
                                >
                                  <RotateCcw className="w-4 h-4" />
                                </button>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Research Execution Pipeline Node */}
              {isProcessing && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col space-y-3">
                  <div className="flex items-center gap-2.5 mb-1">
                    <div className="w-6 h-6 rounded-md bg-surface-container border border-border flex items-center justify-center text-primary">
                      <Search className="w-3.5 h-3.5 animate-pulse motion-reduce:animate-none" />
                    </div>
                    <span className="text-xs font-semibold text-foreground tracking-tight">
                      Executing Research Pipeline…
                    </span>
                  </div>
                   
                  <div className="pl-8 space-y-2.5">
                    {visibleGraphStatuses.map((step: any, idx) => {
                      const isPast = step.status === "completed";
                      const isActive = step.status === "running";
                      const isFailed = step.status === "failed" || step.status === "error";
                      const label = step.label || step.message || step.node || String(step);
                       
                      return (
                        <div key={`${label}-${idx}`} className={cn(
                          "flex items-center gap-2.5 transition-opacity duration-200", 
                          isFailed ? "opacity-100" : isPast ? "opacity-100" : isActive ? "opacity-100" : "opacity-40"
                        )}>
                          {isFailed ? (
                            <AlertTriangle className="w-4 h-4 text-destructive shrink-0" />
                          ) : isPast ? (
                            <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
                          ) : isActive ? (
                            <div className="w-4 h-4 flex items-center justify-center shrink-0">
                              <span className="w-2 h-2 bg-primary rounded-full animate-ping motion-reduce:animate-none" />
                            </div>
                          ) : (
                            <div className="w-4 h-4 rounded-full border border-border shrink-0" />
                          )}
                          <span className={cn(
                            "text-xs font-medium",
                            isFailed ? "text-destructive font-semibold" : isPast ? "text-muted-foreground" : isActive ? "text-primary font-semibold" : "text-muted-foreground"
                          )}>
                            {label}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </motion.div>
              )}
              
              <div ref={scrollRef} />
            </div>
          </div>
        </div>

        {/* Jump to Latest Floating Button */}
        <AnimatePresence>
          {showJumpToLatest && (
            <motion.button
              type="button"
              onClick={scrollToLatest}
              initial={{ opacity: 0, y: 8, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.95 }}
              transition={{ duration: 0.15, ease: "easeOut" }}
              className="absolute bottom-[140px] left-1/2 z-40 -translate-x-1/2 rounded-full border border-border bg-surface-container-lowest p-2 text-foreground shadow-sm transition-colors hover:bg-surface-container focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 active:scale-95 disabled:opacity-50 disabled:pointer-events-none motion-reduce:transform-none"
              title="Jump to latest message"
              aria-label="Jump to latest message"
            >
              <ArrowDown className="h-4 w-4" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* Docked Command Composer Dock */}
        <div className="shrink-0 px-4 md:px-10 pb-5 pt-2 flex justify-center relative z-30">
          <div className={cn(
            "w-full relative z-10",
            isInspectorOpen ? "max-w-3xl" : "max-w-4xl"
          )}>
            <form 
              onSubmit={handleSubmit} 
              className="w-full bg-surface-container-lowest border border-border rounded-xl p-2.5 flex flex-col transition-colors duration-150 shadow-sm focus-within:border-primary/50 focus-within:ring-2 focus-within:ring-primary/15"
            >
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                className="w-full max-h-36 min-h-[52px] resize-none overflow-y-auto overflow-x-hidden bg-transparent border-none px-3.5 pt-2 text-sm font-medium text-foreground placeholder:text-muted-foreground outline-none focus:outline-none focus:ring-0"
                placeholder="Ask anything or research your documents…"
                rows={1}
              />
            
              <div className="flex items-center justify-between px-2 pb-1 pt-1.5 border-t border-border/40 mt-1">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span className="hidden sm:inline">Press</span>
                  <kbd className="text-[10px] font-sans font-medium text-muted-foreground border border-border px-1.5 py-0.5 rounded bg-surface">Enter</kbd>
                  <span className="hidden sm:inline">to send</span>
                </div>
              
                <div className="flex items-center gap-2.5">
                  {/* Segmented Mode Selector */}
                  <div className="flex items-center bg-surface-container border border-border rounded-lg p-0.5 text-xs select-none gap-0.5">
                    <button
                      type="button"
                      onClick={() => setChatMode("auto")}
                      className={cn(
                        "px-2.5 py-1 rounded-md transition-colors duration-150 cursor-pointer font-medium flex items-center justify-center focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 active:scale-95 disabled:opacity-50 disabled:pointer-events-none motion-reduce:transform-none",
                        chatMode === "auto"
                          ? "bg-surface-container-lowest text-foreground border-border shadow-xs font-semibold border"
                          : "text-muted-foreground hover:text-foreground hover:bg-surface border border-transparent"
                      )}
                      title="Auto: Dynamic tool routing (Direct, Documents, Web)"
                    >
                      Auto
                    </button>
                    <button
                      type="button"
                      onClick={() => setChatMode("research")}
                      className={cn(
                        "px-2.5 py-1 rounded-md transition-colors duration-150 cursor-pointer font-medium flex items-center justify-center focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 active:scale-95 disabled:opacity-50 disabled:pointer-events-none motion-reduce:transform-none",
                        chatMode === "research"
                          ? "bg-surface-container-lowest text-foreground border-border shadow-xs font-semibold border"
                          : "text-muted-foreground hover:text-foreground hover:bg-surface border border-transparent"
                      )}
                      title="Research: Multi-step evidence synthesis & deep research"
                    >
                      Research
                    </button>
                  </div>

                  {/* Send Button */}
                  <button
                    type="submit"
                    disabled={!inputText.trim() || isProcessing}
                    className={cn(
                      "p-1.5 rounded-lg transition-colors flex items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 active:scale-95 disabled:opacity-40 disabled:pointer-events-none motion-reduce:transform-none",
                      !inputText.trim() || isProcessing 
                        ? "text-muted-foreground cursor-not-allowed" 
                        : "text-primary border border-primary/30 hover:bg-primary/10 cursor-pointer"
                    )}
                    title="Send message"
                    aria-label="Send message"
                  >
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>

      {/* Right Workstation Inspector Panel */}
      <AnimatePresence>
        {isInspectorOpen && (
          <motion.div
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ type: "spring", stiffness: 350, damping: 32 }}
            className="absolute right-0 top-0 bottom-0 w-80 md:w-96 border-l border-border bg-background z-40 flex flex-col"
          >
            <div className="h-14 border-b border-border flex items-center px-4 bg-surface-container-lowest shrink-0">
              <h3 className="font-semibold text-xs text-foreground flex items-center">
                <BrainCircuit className="w-4 h-4 mr-2 text-primary" />
                Knowledge Context Inspector
              </h3>
              <button 
                type="button"
                onClick={() => setIsInspectorOpen(false)}
                className="ml-auto p-1.5 hover:bg-surface rounded-md text-muted-foreground hover:text-foreground transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
                title="Close inspector"
                aria-label="Close inspector"
              >
                <PanelRightClose className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-5">
              {/* Add Sources Section */}
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground mb-2.5">Add Sources to Context</h4>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="flex flex-col items-center justify-center p-3 rounded-xl border border-border bg-surface-container-lowest hover:border-primary/40 hover:bg-surface-container transition-colors duration-150 active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 group motion-reduce:transform-none"
                  >
                    <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center mb-1.5">
                      <Plus className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-medium text-foreground">Upload File</span>
                  </button>
                  
                  {allDocuments.length > 0 && onLinkDocument ? (
                    <div className="relative">
                      <button
                        type="button"
                        onClick={() => setIsLinkDropdownOpen(!isLinkDropdownOpen)}
                        className="w-full flex flex-col items-center justify-center p-3 border border-border bg-surface-container-lowest hover:border-primary/40 hover:bg-surface-container rounded-xl transition-colors duration-150 active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 group motion-reduce:transform-none"
                      >
                        <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center mb-1.5">
                          <Paperclip className="w-4 h-4" />
                        </div>
                        <span className="text-xs font-medium text-foreground">Link Existing</span>
                      </button>
                      
                      <AnimatePresence>
                        {isLinkDropdownOpen && (
                          <>
                            <div className="fixed inset-0 z-40" onClick={() => setIsLinkDropdownOpen(false)} />
                            <motion.div
                              initial={{ opacity: 0, y: 4, scale: 0.98 }}
                              animate={{ opacity: 1, y: 0, scale: 1 }}
                              exit={{ opacity: 0, y: 4, scale: 0.98 }}
                              transition={{ duration: 0.14, ease: "easeOut" }}
                              className="absolute top-full left-0 right-0 mt-1.5 p-1 bg-surface-container-lowest border border-border rounded-xl shadow-md z-50 max-h-48 overflow-y-auto"
                            >
                              {allDocuments.reduce<React.ReactNode[]>((acc, d) => {
                                if (!documents.find(sd => sd.id === d.id)) {
                                  acc.push(
                                    <button
                                      key={d.id}
                                      type="button"
                                      onClick={() => {
                                        onLinkDocument(d.id);
                                        setIsLinkDropdownOpen(false);
                                      }}
                                      className="w-full text-left px-2.5 py-2 text-xs font-medium text-foreground hover:bg-surface hover:text-primary rounded-lg transition-colors truncate focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
                                    >
                                      {d.name}
                                    </button>
                                  );
                                }
                                return acc;
                              }, [])}
                              {allDocuments.every(d => documents.some(sd => sd.id === d.id)) && (
                                <div className="px-3 py-3 text-center text-xs text-muted-foreground">
                                  All indexed documents linked
                                </div>
                              )}
                            </motion.div>
                          </>
                        )}
                      </AnimatePresence>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center p-3 rounded-xl border border-border bg-surface-container-lowest opacity-50 cursor-not-allowed">
                      <div className="w-8 h-8 rounded-full bg-surface-container text-muted-foreground flex items-center justify-center mb-1.5">
                        <Paperclip className="w-4 h-4" />
                      </div>
                      <span className="text-xs font-medium text-muted-foreground">Link Existing</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Active Documents Section */}
              <div>
                <div className="flex items-center justify-between mb-2.5">
                  <h4 className="text-xs font-semibold text-muted-foreground">Active Documents ({documents.length})</h4>
                </div>
                
                {documents.length === 0 ? (
                  <div className="text-center p-5 bg-surface border border-dashed border-border rounded-xl">
                    <Database className="w-6 h-6 text-muted-foreground opacity-40 mx-auto mb-1.5" />
                    <p className="text-xs text-muted-foreground">No documents in context.</p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">Upload or link sources above.</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {documents.map((doc, i) => {
                      const isProcessingDoc = doc.status === "queued" || doc.status === "indexing";
                      const isFailed = doc.status === "failed";
                      const isIndexed = doc.status === "indexed" || !doc.status;
                      
                      return (
                        <div key={i} className={cn("p-2.5 bg-surface-container-lowest border border-border rounded-lg flex items-start space-x-2.5 relative overflow-hidden", isProcessingDoc ? "opacity-80" : "")}>
                          {isProcessingDoc ? (
                            <Loader2 className="w-4 h-4 text-primary animate-spin shrink-0 mt-0.5 motion-reduce:animate-none" />
                          ) : isFailed ? (
                            <AlertTriangle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
                          ) : (
                            <CheckCircle2 className="w-4 h-4 text-success shrink-0 mt-0.5" />
                          )}
                          <div className="flex-1 min-w-0">
                            <h5 className="text-xs font-semibold text-foreground truncate">{doc.name}</h5>
                            <p className="text-[10px] text-muted-foreground mt-0.5 truncate">
                              Status: {isProcessingDoc ? "Indexing…" : isFailed ? "Failed" : "Indexed"} · Size: {doc.size || (doc as any).file_size_kb + " KB"}
                            </p>
                          </div>
                          {onUnlinkDocument && (
                            <button
                              type="button"
                              onClick={() => onUnlinkDocument(doc.id)}
                              className="p-1 hover:bg-surface-container text-muted-foreground hover:text-foreground rounded-md transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40"
                              title="Unlink document"
                              aria-label="Unlink document"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          )}
                          {isProcessingDoc && (
                            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-surface-container-high overflow-hidden">
                              <motion.div 
                                className="h-full bg-primary" 
                                initial={{ x: "-100%" }}
                                animate={{ x: "100%" }}
                                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                              />
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileUpload}
        className="hidden"
        accept=".pdf,.txt,.md,.docx"
      />

      {/* Citation Detail Modal */}
      <AnimatePresence>
        {activeCitation && (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-background/80 flex items-center justify-center p-4 z-50"
          >
            <motion.div 
              initial={{ scale: 0.96 }} 
              animate={{ scale: 1 }} 
              exit={{ scale: 0.96 }}
              className="bg-surface-container-lowest max-w-xl w-full rounded-2xl p-6 border border-border shadow-lg space-y-4"
            >
              <div className="flex items-center justify-between border-b border-border pb-3.5">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-surface rounded-md border border-border text-foreground">
                    {activeCitation.type === 'web' ? <Search className="w-4 h-4 text-muted-foreground" /> : <FileText className="w-4 h-4 text-muted-foreground" />}
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm tracking-tight truncate max-w-[320px]">
                      {activeCitation.name || activeCitation.title || "Grounding Source"}
                    </h4>
                    {activeCitation.page != null && (
                      <p className="text-[11px] text-muted-foreground mt-0.5">Page {activeCitation.page}</p>
                    )}
                  </div>
                </div>
                {activeCitation.fitScore != null && (
                  <span className="text-[11px] font-semibold text-foreground bg-surface px-2 py-1 rounded-md border border-border">
                    {activeCitation.fitScore}% Match
                  </span>
                )}
              </div>
              
              <div className="bg-surface p-4 rounded-xl text-xs text-muted-foreground leading-relaxed font-mono max-h-72 overflow-y-auto whitespace-pre-line border border-border/50">
                {activeCitation.type === 'web' ? (
                  <a href={activeCitation.url} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2 break-all">
                    {activeCitation.url}
                  </a>
                ) : (
                  activeCitation.snippet || "No snippet available."
                )}
              </div>

              <div className="flex justify-end pt-1">
                <button
                  type="button"
                  onClick={() => setActiveCitation(null)}
                  className="px-4 py-2 bg-surface border border-border hover:bg-surface-container text-foreground rounded-lg font-medium text-xs cursor-pointer transition-colors active:scale-95 disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 shadow-xs motion-reduce:transform-none"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <FeedbackDialog
        isOpen={!!activeFeedbackMsg}
        onClose={() => setActiveFeedbackMsg(null)}
        onSubmit={handleFeedbackSubmit}
        defaultRating={activeFeedbackMsg?.rating || 5}
      />
    </div>
  );
}
