import React, { useState, useRef, useEffect } from "react";
import { Message, SourceDocument, Citation, DocType } from "../types";
import { 
  Sparkles, Send, Paperclip, Mic, FileText, FileSpreadsheet, 
  PanelRightClose, PanelRightOpen, BrainCircuit, LayoutList, Layers, Settings2, 
  CheckCircle2, Search, ArrowRight, ArrowDown, Plus, Loader2, AlertTriangle, X,
  Copy, ThumbsUp, ThumbsDown, Check, RotateCcw
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
    <div className="my-5 w-full max-w-full overflow-hidden rounded-2xl border border-border bg-surface-container-lowest shadow-sm dark:border-[#2a2f3a] dark:bg-[#11141a] dark:shadow-[0_18px_48px_rgba(0,0,0,0.35)]">
      <div className="flex items-center justify-between border-b border-border px-4 py-2 dark:border-[#2a2f3a] dark:bg-[#171b23]">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          {language || "code"}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground dark:hover:bg-white/8"
          title="Copy code"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-4 text-[14px] leading-6 text-foreground dark:bg-[#0c0f14] dark:text-[#e7eaf0]">
        <code>{code}</code>
      </pre>
    </div>
  );
}

const formatAssistantMarkdown = (text: string) => {
  let inFence = false;
  let sectionIndex = 0;
  return text
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
};

interface ChatWorkspaceProps {
  messages: Message[];
  isProcessing: boolean;
  onSendMessage: (text: string) => void;
  documents: SourceDocument[];
  allDocuments?: SourceDocument[];
  activeConvId?: string;
  onAddDocument: (file: File) => void;
  onLinkDocument?: (documentId: string) => void;
  onUnlinkDocument?: (documentId: string) => void;
  onRetryMessage?: (messageId: string) => void;
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
  sessionTitle = "New Chat",
  graphStatuses = [],
}: ChatWorkspaceProps) {
  const [inputText, setInputText] = useState("");
  const [isMicActive, setIsMicActive] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(true);
  const [inspectorTab, setInspectorTab] = useState<"sources">("sources");
  const [chatMode, setChatMode] = useState<"auto" | "research">("auto");
  const [isLinkDropdownOpen, setIsLinkDropdownOpen] = useState(false);
  
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [feedbacks, setFeedbacks] = useState<Record<string, 'up' | 'down'>>({});
  const [activeFeedbackMsg, setActiveFeedbackMsg] = useState<{ id: string; dbId: number; rating: number; type: 'up' | 'down' } | null>(null);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  
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
    onSendMessage(inputText.trim());
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
      } else {
        console.error("Fallback copy unsuccessful");
      }
    } catch (err) {
      console.error("Fallback copy failed: ", err);
    }
    document.body.removeChild(textArea);
  };

  const handleCopy = (id: string, text: string) => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text)
        .then(() => {
          setCopiedId(id);
          setTimeout(() => setCopiedId(null), 1000);
        })
        .catch((err) => {
          console.error("Clipboard copy failed: ", err);
          fallbackCopyText(id, text);
        });
    } else {
      fallbackCopyText(id, text);
    }
  };

  const handleFeedback = (id: string, type: 'up' | 'down') => {
    const msg = messages.find(m => m.id === id);
    if (!msg) return;
    if (!msg.dbId) {
      console.warn("Cannot submit feedback: message database ID not available yet.");
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

  const handleMicToggle = () => {
    if (!isMicActive) {
      setIsMicActive(true);
      const suggestions = [
        "Synthesize latency metrics from monolith blueprint.",
        "Compare operational efficiency projections in Q3 revised report."
      ];
      const randomPrompt = suggestions[Math.floor(Math.random() * suggestions.length)];
      
      let typed = "";
      let index = 0;
      
      const interval = setInterval(() => {
        if (index < randomPrompt.length) {
          typed += randomPrompt[index];
          setInputText(typed);
          index++;
        } else {
          clearInterval(interval);
          setIsMicActive(false);
        }
      }, 35);
    } else {
      setIsMicActive(false);
    }
  };

  return (
    <div className="flex-1 flex w-full h-full overflow-hidden relative bg-background">
      
      {/* Main Chat Area */}
      <div
        className={cn(
          "flex flex-col min-w-0 min-h-0 transition-all duration-300 relative",
          isInspectorOpen ? "w-[calc(100%-20rem)] md:w-[calc(100%-24rem)] shrink-0" : "flex-1"
        )}
      >
        {/* Header */}
        <header className="h-16 shrink-0 flex justify-between items-center px-6 pt-1 bg-background/80 backdrop-blur-md border-b border-border z-20">
          <div className="flex items-center space-x-3">
            <span className="text-sm font-semibold text-foreground tracking-tight leading-none">{sessionTitle}</span>
            <span className="text-xs text-muted-foreground flex items-center leading-none">
              <Paperclip className="w-3.5 h-3.5 mr-1" />
              {documents.length} Docs
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsInspectorOpen(!isInspectorOpen)}
              className={cn(
                "p-1.5 rounded-md transition-colors flex items-center space-x-2 text-xs font-semibold btn-press",
                isInspectorOpen ? "bg-surface-container-high text-foreground" : "text-muted-foreground hover:bg-surface hover:text-foreground"
              )}
            >
              <span>Inspector</span>
              {isInspectorOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
            </button>
          </div>
        </header>

        {/* Chat Feed */}
        <div
          ref={chatScrollRef}
          onScroll={handleChatScroll}
          className="flex-1 overflow-y-auto overflow-x-hidden px-6 md:px-12 pt-6 pb-4 flex flex-col items-center min-h-0"
        >
          <div className={cn("w-full min-w-0 space-y-12", isInspectorOpen ? "max-w-4xl" : "max-w-5xl")}>
            
            {messages.length === 0 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-16 select-none mt-10 text-center flex flex-col items-center w-full">
                <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-6 shadow-glow-primary float-icon">
                   <BrainCircuit className="w-8 h-8 text-primary" />
                </div>
                <h2 className="text-3xl font-semibold text-foreground tracking-tight mb-3">
                  Welcome to DocuMind
                </h2>
                <p className="text-muted-foreground text-sm max-w-xl text-center mb-8">
                  Start a new conversation to research your documents. The system will automatically provide quick answers or execute deep semantic search across your attached context.
                </p>

                {/* Prompt Suggestions Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl px-4">
                  {[
                    {
                      title: "Summarize latest document",
                      desc: "Extract key findings and generate an executive summary",
                      prompt: "Please write a detailed summary of the main topics and key findings in my document.",
                      icon: FileText,
                      color: "text-amber-500 bg-amber-500/10 border-amber-500/20"
                    },
                    {
                      title: "Verify grounding context",
                      desc: "Cross-reference facts and check for correct context bounds",
                      prompt: "Identify the critical evidence in the document that supports the claim: ",
                      icon: Search,
                      color: "text-blue-500 bg-blue-500/10 border-blue-500/20"
                    },
                    {
                      title: "Compare documents",
                      desc: "Extract side-by-side differences between multiple sources",
                      prompt: "Compare the main themes and identify any contrasting information between the documents.",
                      icon: Layers,
                      color: "text-emerald-500 bg-emerald-500/10 border-emerald-500/20"
                    },
                    {
                      title: "Draft QA verification script",
                      desc: "Write a verification logic script based on standard models",
                      prompt: "Write a python script to run verification rules on retrieved contexts.",
                      icon: Settings2,
                      color: "text-pink-500 bg-pink-500/10 border-pink-500/20"
                    }
                  ].map((item, idx) => (
                    <motion.button
                      key={idx}
                      whileHover={{ y: -3, scale: 1.01 }}
                      onClick={() => setInputText(item.prompt)}
                      className="flex items-start p-4 rounded-2xl border border-border bg-surface-container-lowest hover:border-primary/45 hover:bg-surface/50 transition-all text-left group cursor-pointer btn-press"
                    >
                      <div className={cn("w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 mr-4 group-hover:scale-105 group-hover:rotate-6 transition-all duration-300", item.color)}>
                        <item.icon className="w-5 h-5" />
                      </div>
                      <div className="flex-1 min-w-0 pr-2">
                        <h4 className="text-sm font-semibold text-foreground truncate group-hover:text-primary transition-colors">{item.title}</h4>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-1">{item.desc}</p>
                      </div>
                      <div className="w-5 h-5 rounded-full border border-border flex items-center justify-center text-muted-foreground group-hover:scale-110 group-hover:bg-primary group-hover:border-primary group-hover:text-primary-foreground transition-all duration-300 flex-shrink-0 self-center">
                        <Plus className="w-3.5 h-3.5" />
                      </div>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}

            <div className="space-y-10">
              {messages.map((msg, index) => {
                const isLastMessage = index === messages.length - 1;
                const isStreaming = isProcessing && isLastMessage && msg.sender === "assistant";

                return (
                  <motion.div
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={msg.id}
                    className={cn("flex w-full", msg.sender === "user" ? "justify-end" : "justify-start")}
                  >
                    {msg.sender === "user" ? (
                      <div className="max-w-[80%] flex flex-col items-end space-y-1 group">
                        <div className="bg-surface-container-high text-foreground px-5 py-3.5 rounded-2xl rounded-tr-sm shadow-sm border border-border">
                          <div className="text-[15px] leading-relaxed whitespace-pre-wrap font-normal">
                            {msg.text}
                          </div>
                        </div>
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center pr-1 h-7">
                          <button
                            onClick={() => handleCopy(msg.id, msg.text)}
                            className="p-1 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md transition-colors"
                            title="Copy message"
                          >
                            {copiedId === msg.id ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start space-x-4 w-full min-w-0">
                        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center text-white flex-shrink-0 mt-0.5 shadow-sm">
                          <Sparkles className="w-4 h-4" />
                        </div>
                        <div className={cn(
                          "min-w-0 flex-1 flex flex-col space-y-2 p-3 rounded-2xl border border-transparent transition-all duration-300",
                          isStreaming && "bg-primary/[0.02] border-primary/10 shadow-[0_0_20px_rgba(123,108,246,0.06)]"
                        )}>
                          <div className="max-w-none min-w-0 text-foreground">
                            {msg.text ? (
                              <>
                                <ReactMarkdown
                                  remarkPlugins={[remarkGfm]}
                                  components={{
                                    a: ({ href, children }) => (
                                      <a href={href} target="_blank" rel="noreferrer" className="text-sky-500 underline underline-offset-2 break-words">
                                        {children}
                                      </a>
                                    ),
                                    p: ({ children }) => (
                                      <p className="my-5 break-words text-[16px] leading-8 font-normal text-foreground">
                                        {children}
                                      </p>
                                    ),
                                    h2: ({ children }) => (
                                      <h2 className="mt-10 mb-5 text-[22px] font-semibold tracking-tight text-foreground">
                                        {children}
                                      </h2>
                                    ),
                                    h3: ({ children }) => (
                                      <h3 className="mt-8 mb-4 text-[17px] font-semibold tracking-tight text-foreground">
                                        {children}
                                      </h3>
                                    ),
                                    ul: ({ children }) => (
                                      <ul className="my-5 list-disc space-y-2 pl-7">
                                        {children}
                                      </ul>
                                    ),
                                    ol: ({ children }) => (
                                      <ol className="my-5 list-decimal space-y-2 pl-7">
                                        {children}
                                      </ol>
                                    ),
                                    li: ({ children }) => (
                                      <li className="break-words pl-1 text-[16px] leading-8 text-foreground">
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
                                        <code className="rounded-md bg-surface px-1.5 py-0.5 text-[14px] font-semibold text-foreground" {...props}>
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
                        
                        {/* Citations */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="mt-4 pt-4 border-t border-border/50">
                            <div className="flex flex-wrap gap-2">
                              {msg.citations.map((cite: any, cIdx) => {
                                const isWeb = cite.type === "web";
                                const name = cite.name || cite.title || "Source";
                                const badge = isWeb ? "Web" : cite.page != null ? `P.${cite.page}` : "Doc";
                                return (
                                  <button
                                    key={cIdx}
                                    onClick={() => setActiveCitation(cite)}
                                    className="bg-surface border border-border hover:border-muted-foreground/30 px-2.5 py-1 rounded-md flex items-center space-x-2 transition-all cursor-pointer text-left shadow-sm"
                                  >
                                    {isWeb ? (
                                      <Search className="w-3 h-3 text-sky-500" />
                                    ) : name.endsWith(".xlsx") || name.endsWith(".csv") ? (
                                      <FileSpreadsheet className="w-3 h-3 text-emerald-500" />
                                    ) : name.endsWith(".pdf") ? (
                                      <FileText className="w-3 h-3 text-rose-500" />
                                    ) : (
                                      <FileText className="w-3 h-3 text-amber-500" />
                                    )}
                                    <span className="text-xs font-medium text-foreground truncate max-w-[120px]">
                                      {name}
                                    </span>
                                    <span className="text-[10px] bg-primary/10 text-primary px-1 rounded-full font-semibold">
                                      {badge}
                                    </span>
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Actions Toolbar (Copy & Feedback) */}
                        {msg.text && (
                          <div className="flex items-center space-x-1 mt-2">
                            <button
                              onClick={() => handleCopy(msg.id, msg.text)}
                              className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-surface rounded-md transition-colors"
                              title="Copy message"
                            >
                              {copiedId === msg.id ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                            </button>
                            <button
                              onClick={() => handleFeedback(msg.id, 'up')}
                              className={cn(
                                "p-1.5 rounded-md transition-colors",
                                feedbacks[msg.id] === 'up' ? "text-primary bg-primary/10" : "text-muted-foreground hover:text-foreground hover:bg-surface"
                              )}
                              title="Good response"
                            >
                              <ThumbsUp className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleFeedback(msg.id, 'down')}
                              className={cn(
                                "p-1.5 rounded-md transition-colors",
                                feedbacks[msg.id] === 'down' ? "text-red-500 bg-red-500/10" : "text-muted-foreground hover:text-foreground hover:bg-surface"
                              )}
                              title="Bad response"
                            >
                              <ThumbsDown className="w-4 h-4" />
                            </button>
                            {onRetryMessage && (
                              <button
                                onClick={() => onRetryMessage(msg.id)}
                                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-surface transition-colors btn-press"
                                title="Regenerate response"
                              >
                                <RotateCcw className="w-4 h-4" />
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </motion.div>
              )})}

              {/* Research Execution UI */}
              {isProcessing && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col space-y-4">
                   <div className="flex items-center space-x-3 mb-2">
                     <div className="w-6 h-6 rounded-md bg-surface-container-high border border-border flex items-center justify-center text-muted-foreground shadow-sm">
                       <Search className="w-3.5 h-3.5 animate-pulse text-primary" />
                     </div>
                     <span className="text-sm font-semibold text-foreground tracking-tight">
                       Building Answer...
                     </span>
                   </div>
                   
                   <div className="pl-9 space-y-3">
                     {visibleGraphStatuses.map((step: any, idx) => {
                       const isPast = step.status === "completed";
                       const isActive = step.status === "running";
                       const label = step.label || step.message || step.node || String(step);
                       
                       return (
                         <div key={`${label}-${idx}`} className={cn("flex items-center space-x-3 transition-opacity duration-300", 
                           isPast ? "opacity-100" : isActive ? "opacity-100" : "opacity-30"
                         )}>
                           {isPast ? (
                             <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                           ) : isActive ? (
                             <div className="w-4 h-4 flex items-center justify-center">
                               <div className="w-2 h-2 bg-primary rounded-full animate-ping" />
                             </div>
                           ) : (
                             <div className="w-4 h-4 rounded-full border-2 border-border" />
                           )}
                           <span className={cn(
                             "text-xs font-medium",
                             isPast ? "text-muted-foreground" : isActive ? "text-primary" : "text-muted-foreground"
                           )}>
                             {label}
                           </span>
                         </div>
                       )
                     })}
                   </div>
                </motion.div>
              )}
              
              <div ref={scrollRef} />
            </div>
          </div>
        </div>

        <AnimatePresence>
          {showJumpToLatest && (
            <motion.button
              type="button"
              onClick={scrollToLatest}
              initial={{ opacity: 0, y: 10, scale: 0.94 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.94 }}
              transition={{ duration: 0.16, ease: "easeOut" }}
              className="absolute bottom-[160px] left-1/2 z-40 -translate-x-1/2 rounded-full border border-border bg-surface-container-lowest p-2.5 text-foreground shadow-premium-panel transition-colors hover:bg-surface-container-high dark:border-white/15"
              title="Jump to latest"
              aria-label="Jump to latest message"
            >
              <ArrowDown className="h-4 w-4" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* Input Bar */}
        <div className="shrink-0 px-6 md:px-12 pb-6 pt-2 flex justify-center relative z-30">
          <div className="absolute inset-0 bg-primary/20 blur-[100px] ambient-glow max-w-2xl mx-auto rounded-full h-24 bottom-0 top-auto translate-y-6 pointer-events-none" />
          <form 
            onSubmit={handleSubmit} 
            className={cn(
              "w-full bg-surface-container-lowest backdrop-blur-xl rounded-2xl p-2 flex flex-col relative z-10 input-glow-ring",
              isInspectorOpen ? "max-w-4xl" : "max-w-5xl"
            )}
          >
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full max-h-32 min-h-[56px] resize-none overflow-y-auto overflow-x-hidden bg-transparent border-none px-4 pt-3 text-[15px] font-medium text-foreground placeholder:text-muted-foreground outline-none focus:outline-none focus:ring-0"
              placeholder="Ask anything..."
              rows={1}
            />
            
            <div className="flex items-center justify-between px-2 pb-1 pt-2">
              <div className="flex items-center space-x-1">
                {/* File upload removed from chat input */}
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={handleMicToggle}
                  className={cn(
                    "p-1.5 rounded-md transition-colors cursor-pointer flex items-center justify-center btn-press",
                    isMicActive ? "bg-red-500/10 text-red-500" : "text-muted-foreground hover:bg-surface hover:text-foreground"
                  )}
                  title="Voice input"
                >
                  <Mic className="w-4 h-4" />
                </button>

                <button
                  type="button"
                  onClick={() => setChatMode(prev => prev === "auto" ? "research" : "auto")}
                  className={cn(
                    "px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded-md transition-colors border btn-press",
                    chatMode === "auto" 
                      ? "bg-surface border-border text-foreground hover:bg-surface-container-high" 
                      : "bg-primary/10 border-primary/20 text-primary hover:bg-primary/20"
                  )}
                  title="Toggle Mode"
                >
                  {chatMode === "auto" ? "Auto" : "Research"}
                </button>

                <button
                  type="submit"
                  disabled={!inputText.trim() || isProcessing}
                  className={cn(
                    "p-2 rounded-xl transition-all flex items-center justify-center btn-press",
                    !inputText.trim() || isProcessing 
                      ? "opacity-50 bg-surface-container-high text-muted-foreground cursor-not-allowed" 
                      : "bg-primary text-primary-foreground cursor-pointer hover:bg-primary/90 shadow-sm"
                  )}
                  title="Send message"
                >
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* Right Inspector Panel */}
      <AnimatePresence>
        {isInspectorOpen && (
          <motion.div
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute right-0 top-0 bottom-0 w-80 md:w-96 border-l border-border bg-background z-40 flex flex-col shadow-2xl"
          >
            <div className="h-16 border-b border-border flex items-center px-4 pt-1 bg-background/50 backdrop-blur-md shrink-0">
              <h3 className="font-semibold text-sm text-foreground flex items-center leading-none">
                <BrainCircuit className="w-4 h-4 mr-2 text-primary" />
                Inspector
              </h3>
              <button 
                onClick={() => setIsInspectorOpen(false)}
                className="ml-auto p-1.5 hover:bg-surface rounded-md text-muted-foreground transition-colors"
              >
                <PanelRightClose className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5">
              <div className="space-y-4">
                <div className="mb-6">
                  <h4 className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Add Sources</h4>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="flex flex-col items-center justify-center p-3 rounded-xl border border-border bg-surface-container-lowest hover:border-primary/40 hover:bg-surface/50 transition-all btn-press group"
                    >
                      <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
                        <Plus className="w-4 h-4" />
                      </div>
                      <span className="text-[11px] font-semibold text-foreground">Upload File</span>
                    </button>
                    
                    {allDocuments.length > 0 && onLinkDocument ? (
                      <div className="relative">
                        <button
                          type="button"
                          onClick={() => setIsLinkDropdownOpen(!isLinkDropdownOpen)}
                          className="w-full flex flex-col items-center justify-center p-3 border border-border bg-surface-container-lowest hover:border-primary/40 hover:bg-surface/50 rounded-xl transition-all btn-press group"
                        >
                          <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
                            <Paperclip className="w-4 h-4" />
                          </div>
                          <span className="text-[11px] font-semibold text-foreground">Link Existing</span>
                        </button>
                        
                        <AnimatePresence>
                          {isLinkDropdownOpen && (
                            <>
                              <div className="fixed inset-0 z-40" onClick={() => setIsLinkDropdownOpen(false)} />
                              <motion.div
                                initial={{ opacity: 0, y: 5, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, y: 5, scale: 0.95 }}
                                transition={{ duration: 0.15, ease: "easeOut" }}
                                className="absolute top-full left-0 right-0 mt-2 p-1 bg-surface-container-lowest border border-border rounded-xl shadow-premium-panel z-50 max-h-48 overflow-y-auto"
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
                                        className="w-full text-left px-3 py-2 text-xs font-medium text-foreground hover:bg-surface hover:text-primary rounded-lg transition-colors truncate"
                                      >
                                        {d.name}
                                      </button>
                                    );
                                  }
                                  return acc;
                                }, [])}
                                {allDocuments.every(d => documents.some(sd => sd.id === d.id)) && (
                                  <div className="px-3 py-4 text-center text-xs text-muted-foreground">
                                    All documents linked
                                  </div>
                                )}
                              </motion.div>
                            </>
                          )}
                        </AnimatePresence>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center p-3 rounded-xl border border-border bg-surface-container-lowest opacity-50 cursor-not-allowed">
                        <div className="w-8 h-8 rounded-full bg-surface-container-high text-muted-foreground flex items-center justify-center mb-2">
                          <Paperclip className="w-4 h-4" />
                        </div>
                        <span className="text-[11px] font-semibold text-muted-foreground">Link Existing</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Active Documents ({documents.length})</h4>
                </div>
                
                {documents.length === 0 ? (
                  <div className="text-center p-4 bg-surface border border-dashed border-border rounded-lg">
                    <p className="text-xs text-muted-foreground">No documents uploaded.</p>
                  </div>
                ) : (
                  documents.map((doc, i) => {
                    const isProcessing = doc.status === "queued" || doc.status === "indexing";
                    const isFailed = doc.status === "failed";
                    const isIndexed = doc.status === "indexed" || !doc.status;
                    
                    return (
                      <div key={i} className={cn("p-3 bg-surface border border-border rounded-lg flex items-start space-x-3 relative overflow-hidden", isProcessing ? "opacity-80" : "")}>
                        {isProcessing ? (
                           <Loader2 className="w-4 h-4 text-primary animate-spin shrink-0 mt-0.5" />
                        ) : isFailed ? (
                           <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                        ) : (
                           <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                        )}
                        <div className="flex-1 min-w-0">
                          <h5 className="text-xs font-semibold text-foreground truncate">{doc.name}</h5>
                          <p className="text-[10px] text-muted-foreground mt-1 truncate">Status: {isProcessing ? "Indexing..." : isFailed ? "Failed" : "Indexed"} • Size: {doc.size || (doc as any).file_size_kb + " KB"}</p>
                        </div>
                        {onUnlinkDocument && (
                          <button
                            onClick={() => onUnlinkDocument(doc.id)}
                            className="p-1 hover:bg-border rounded-md transition-colors"
                            title="Unlink document"
                          >
                            <X className="w-3.5 h-3.5 text-muted-foreground" />
                          </button>
                        )}
                        {isProcessing && (
                          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary/20">
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
                  })
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

      <AnimatePresence>
        {activeCitation && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-background/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
          >
            <motion.div 
              initial={{ scale: 0.95 }} animate={{ scale: 1 }} exit={{ scale: 0.95 }}
              className="bg-surface-container-lowest max-w-xl w-full rounded-2xl p-6 border border-border shadow-premium space-y-4"
            >
              <div className="flex items-center justify-between border-b border-border pb-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-surface rounded-md border border-border text-foreground">
                    {activeCitation.type === 'web' ? <Search className="w-4 h-4 text-sky-500" /> : <FileText className="w-4 h-4 text-amber-500" />}
                  </div>
                  <div>
                    <h5 className="font-semibold text-foreground text-sm tracking-tight truncate max-w-[300px]">
                      {activeCitation.name || activeCitation.title || "Source"}
                    </h5>
                    {activeCitation.page != null && (
                      <p className="text-[10px] text-muted-foreground mt-0.5">Page {activeCitation.page}</p>
                    )}
                  </div>
                </div>
                {activeCitation.fitScore != null && (
                  <span className="text-[10px] font-semibold text-foreground bg-surface px-2 py-1 rounded-md border border-border">
                    {activeCitation.fitScore}% Match
                  </span>
                )}
              </div>
              
              <div className="bg-surface p-4 rounded-xl text-sm text-muted-foreground leading-relaxed font-mono max-h-72 overflow-y-auto whitespace-pre-line border border-border/50">
                {activeCitation.type === 'web' ? (
                  <a href={activeCitation.url} target="_blank" rel="noreferrer" className="text-sky-500 hover:underline">
                    {activeCitation.url}
                  </a>
                ) : (
                  activeCitation.snippet || "No snippet available."
                )}
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={() => setActiveCitation(null)}
                  className="px-4 py-2 bg-surface border border-border hover:bg-surface-container text-foreground rounded-lg font-medium text-sm cursor-pointer transition-colors shadow-sm"
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
