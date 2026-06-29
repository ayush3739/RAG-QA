import React, { useState, useRef, useEffect } from "react";
import { Message, SourceDocument, Citation, DocType } from "../types";
import { 
  Sparkles, Send, Paperclip, Mic, FileText, FileSpreadsheet, 
  PanelRightClose, PanelRightOpen, BrainCircuit, LayoutList, Layers, Settings2, 
  CheckCircle2, Search, ArrowRight, Plus, Loader2, AlertTriangle, X,
  Copy, ThumbsUp, ThumbsDown, Check
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
  selectedModel: string;
}

export default function ChatWorkspace({
  messages,
  isProcessing,
  onSendMessage,
  documents,
  allDocuments = [],
  activeConvId,
  onAddDocument,
  onLinkDocument,
  onUnlinkDocument,
  selectedModel,
}: ChatWorkspaceProps) {
  const [inputText, setInputText] = useState("");
  const [isMicActive, setIsMicActive] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(true);
  const [inspectorTab, setInspectorTab] = useState<"sources">("sources");
  const [chatMode, setChatMode] = useState<"auto" | "research">("auto");
  
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [feedbacks, setFeedbacks] = useState<Record<string, 'up' | 'down'>>({});
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Fake research execution steps
  const [executionStep, setExecutionStep] = useState(0);
  const researchSteps = ["Planning", "Searching Documents", "Searching Web", "Analyzing Evidence", "Writing", "Reviewing"];

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isProcessing, executionStep]);

  useEffect(() => {
    if (isProcessing) {
      setExecutionStep(0);
      const interval = setInterval(() => {
        setExecutionStep(prev => (prev < researchSteps.length - 1 ? prev + 1 : prev));
      }, 1200);
      return () => clearInterval(interval);
    }
  }, [isProcessing]);

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
    setFeedbacks(prev => ({ ...prev, [id]: type }));
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
      <div className={cn("flex-1 flex flex-col min-w-0 transition-all duration-300 relative", isInspectorOpen ? "mr-80 md:mr-96" : "")}>
        {/* Header */}
        <header className="absolute top-0 left-0 right-0 h-14 flex justify-between items-center px-6 bg-background/80 backdrop-blur-md border-b border-border z-20">
          <div className="flex items-center space-x-3">
            <span className="text-sm font-medium text-foreground tracking-tight">Active Session</span>
            <span className="text-[10px] font-bold text-primary bg-primary/10 px-2 py-0.5 rounded-full uppercase tracking-wider">
              Research Auto
            </span>
            <span className="text-xs text-muted-foreground flex items-center">
              <Paperclip className="w-3.5 h-3.5 mr-1" />
              {documents.length} Docs
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsInspectorOpen(!isInspectorOpen)}
              className={cn(
                "p-1.5 rounded-md transition-colors flex items-center space-x-2 text-xs font-semibold",
                isInspectorOpen ? "bg-surface-container-high text-foreground" : "text-muted-foreground hover:bg-surface hover:text-foreground"
              )}
            >
              <span>Inspector</span>
              {isInspectorOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
            </button>
          </div>
        </header>

        {/* Chat Feed */}
        <div className="flex-1 overflow-y-auto px-6 md:px-12 pt-20 pb-36 flex flex-col items-center">
          <div className="w-full max-w-3xl space-y-12">
            
            {messages.length === 0 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-16 select-none mt-10 text-center flex flex-col items-center">
                <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-6 shadow-lg shadow-primary/5">
                   <BrainCircuit className="w-8 h-8 text-primary" />
                </div>
                <h2 className="text-3xl font-semibold text-foreground tracking-tight mb-3">
                  Welcome to DocuMind
                </h2>
                <p className="text-muted-foreground text-sm max-w-xl text-center">
                  Start a new conversation to research your documents. The system will automatically provide quick answers or execute deep semantic search across your attached context.
                </p>
              </motion.div>
            )}

            <div className="space-y-10">
              {messages.map((msg) => (
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
                    <div className="flex items-start space-x-4 max-w-[85%]">
                      <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center text-white flex-shrink-0 mt-0.5 shadow-sm">
                        <Sparkles className="w-4 h-4" />
                      </div>
                      <div className="flex-1 flex flex-col space-y-2">
                        <div className="prose prose-sm dark:prose-invert prose-p:text-[15px] prose-p:leading-relaxed prose-p:font-normal prose-li:text-[15px] prose-li:leading-relaxed prose-li:font-normal prose-pre:bg-surface prose-pre:border prose-pre:border-border text-foreground">
                          {msg.text ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown> : <span className="text-muted-foreground animate-pulse text-[15px]">Thinking...</span>}
                        </div>
                        
                        {/* Citations */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="mt-4 pt-4 border-t border-border/50">
                            <div className="flex flex-wrap gap-2">
                              {msg.citations.map((cite: any, cIdx) => {
                                const isWeb = cite.type === "web";
                                const name = cite.name || cite.title || "Source";
                                const badge = isWeb ? "Web" : cite.page != null ? `P.${cite.page}` : (cite.fitScore ? `${cite.fitScore}%` : "Doc");
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
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}

              {/* Research Execution UI */}
              {isProcessing && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col space-y-4">
                   <div className="flex items-center space-x-3 mb-2">
                     <div className="w-6 h-6 rounded-md bg-surface-container-high border border-border flex items-center justify-center text-muted-foreground shadow-sm">
                       <Search className="w-3.5 h-3.5 animate-pulse text-primary" />
                     </div>
                     <span className="text-sm font-semibold text-foreground tracking-tight">Executing Research...</span>
                   </div>
                   
                   <div className="pl-9 space-y-3">
                     {researchSteps.map((step, idx) => {
                       const isPast = idx < executionStep;
                       const isActive = idx === executionStep;
                       
                       return (
                         <div key={step} className={cn("flex items-center space-x-3 transition-opacity duration-300", 
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
                             {step}
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

        {/* Input Bar */}
        <div className="absolute bottom-6 left-0 right-0 px-6 md:px-12 flex justify-center z-30">
          <form 
            onSubmit={handleSubmit} 
            className="w-full max-w-3xl bg-surface-container-lowest backdrop-blur-xl border border-border shadow-premium rounded-2xl p-2 flex flex-col transition-all focus-within:ring-1 focus-within:ring-primary/30"
          >
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full bg-transparent border-none focus:outline-none focus:ring-0 resize-none min-h-[56px] px-4 pt-3 text-foreground placeholder:text-muted-foreground text-sm font-medium outline-none"
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
                    "p-1.5 rounded-md transition-colors cursor-pointer flex items-center justify-center",
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
                    "px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded-md transition-colors border",
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
                    "p-2 rounded-xl transition-all flex items-center justify-center",
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
            className="absolute right-0 top-0 bottom-0 w-80 md:w-96 border-l border-border bg-surface-container-lowest z-40 flex flex-col shadow-2xl"
          >
            <div className="h-14 border-b border-border flex items-center px-4 bg-background/50 backdrop-blur-md shrink-0">
              <h3 className="font-semibold text-sm text-foreground flex items-center">
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
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Grounding Sources</h4>
                  <div className="flex items-center space-x-2">
                    {allDocuments.length > 0 && onLinkDocument && (
                      <select 
                        className="text-[10px] p-1 bg-surface border border-border rounded-md text-foreground max-w-[90px] outline-none"
                        onChange={(e) => {
                          if (e.target.value) {
                            onLinkDocument(e.target.value);
                            e.target.value = "";
                          }
                        }}
                        defaultValue=""
                      >
                        <option value="" disabled>Link...</option>
                        {allDocuments.filter(d => !documents.find(sd => sd.id === d.id)).map(d => (
                          <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                      </select>
                    )}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="flex items-center space-x-1 px-2 py-1 bg-primary text-primary-foreground text-[10px] font-semibold rounded-md shadow-sm transition-colors cursor-pointer hover:bg-primary/90 whitespace-nowrap"
                    >
                      <Plus className="w-3 h-3" />
                      <span>Upload</span>
                    </button>
                  </div>
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
        accept=".txt,.md,.xlsx,.csv,.pdf"
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
    </div>
  );
}
