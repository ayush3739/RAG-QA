import React, { useState, useRef, useEffect } from "react";
import { Message, SourceDocument, Citation, DocType } from "../types";
import { 
  Sparkles, Send, Paperclip, Mic,
  FileText, Database, FileSpreadsheet
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
  onAddDocument: (name: string, content: string, type: DocType, size: string) => void;
  selectedModel: string;
}

export default function ChatWorkspace({
  messages,
  isProcessing,
  onSendMessage,
  documents,
  onAddDocument,
  selectedModel,
}: ChatWorkspaceProps) {
  const [inputText, setInputText] = useState("");
  const [isMicActive, setIsMicActive] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isProcessing]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!inputText.trim() || isProcessing) return;
    onSendMessage(inputText.trim());
    setInputText("");
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

    const reader = new FileReader();
    reader.onload = (event) => {
      const textContent = event.target?.result as string;
      if (textContent) {
        let type: DocType = "doc";
        if (file.name.endsWith(".xlsx") || file.name.endsWith(".csv")) type = "spreadsheet";
        else if (file.name.endsWith(".pdf")) type = "pdf";
        
        const sizeKB = Math.round(file.size / 1024);
        const sizeStr = sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(1)} MB` : `${sizeKB} KB`;

        onAddDocument(file.name, textContent, type, sizeStr);
        alert(`Added "${file.name}" to workspace.`);
      }
    };
    reader.readAsText(file);
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
    <div className="flex-1 flex flex-col min-w-0 bg-background relative h-full">
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 h-14 flex justify-between items-center px-6 bg-background/80 backdrop-blur-md border-b border-border z-20">
        <div className="text-sm font-medium text-foreground tracking-tight">Research Thread</div>
        <div className="flex items-center space-x-3">
          <span className="text-[11px] font-mono text-muted-foreground bg-surface px-2 py-0.5 rounded border border-border">
            {selectedModel}
          </span>
        </div>
      </header>

      {/* Main Area */}
      <div className="flex-1 overflow-y-auto px-6 md:px-12 pt-20 pb-32 flex flex-col items-center">
        <div className="w-full max-w-3xl space-y-12">
          
          {messages.length === 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} 
              className="mb-16 select-none mt-10"
            >
              <h2 className="text-3xl font-semibold text-foreground tracking-tight mb-3">
                What do you want to know?
              </h2>
              <p className="text-muted-foreground text-sm max-w-xl">
                Search your curated vector library, synthesize reports, or ask cross-disciplinary questions.
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-8">
                <button
                  onClick={() => onSendMessage("What is the operational efficiency gain of applying the Modular RAG Framework?")}
                  className="p-4 bg-surface-container-low border border-border hover:border-primary/50 text-left rounded-xl transition-all cursor-pointer group"
                >
                  <p className="text-sm font-medium text-foreground mb-1">Synthesize Efficiency Metrics</p>
                  <p className="text-xs text-muted-foreground">Analyze efficiency parameters.</p>
                </button>
                <button
                  onClick={() => onSendMessage("Cross reference the efficiency gains with the Q3 fiscal projections for the Monolith project.")}
                  className="p-4 bg-surface-container-low border border-border hover:border-primary/50 text-left rounded-xl transition-all cursor-pointer group"
                >
                  <p className="text-sm font-medium text-foreground mb-1">Q3 Fiscal Projections</p>
                  <p className="text-xs text-muted-foreground">Cross-reference budget figures.</p>
                </button>
              </div>
            </motion.div>
          )}

          <div className="space-y-10">
            {messages.map((msg) => (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                key={msg.id}
                className="flex flex-col space-y-3"
              >
                {msg.sender === "user" ? (
                  // Perplexity style: User query is large and clear
                  <div className="text-2xl font-semibold text-foreground tracking-tight mt-6">
                    {msg.text}
                  </div>
                ) : (
                  <div className="flex flex-col space-y-4">
                    {/* Assistant Response */}
                    <div className="flex items-start space-x-4">
                      <div className="w-6 h-6 rounded-md bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center text-white flex-shrink-0 mt-1 shadow-sm">
                        <Sparkles className="w-3.5 h-3.5" />
                      </div>
                      <div className="flex-1">
                        <div className="prose prose-sm max-w-none dark:prose-invert prose-p:leading-relaxed prose-pre:bg-surface prose-pre:border prose-pre:border-border text-foreground">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                        </div>
                        
                        {/* Citations */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="mt-6 pt-4 border-t border-border/50">
                            <div className="flex flex-wrap gap-2">
                              {msg.citations.map((cite, cIdx) => (
                                <button
                                  key={cIdx}
                                  onClick={() => setActiveCitation(cite)}
                                  className="bg-surface border border-border hover:border-muted-foreground/30 px-2.5 py-1 rounded-md flex items-center space-x-2 transition-all cursor-pointer text-left"
                                >
                                  {cite.name.endsWith(".xlsx") || cite.name.endsWith(".csv") ? (
                                    <FileSpreadsheet className="w-3 h-3 text-emerald-500" />
                                  ) : cite.name.endsWith(".pdf") ? (
                                    <FileText className="w-3 h-3 text-rose-500" />
                                  ) : (
                                    <FileText className="w-3 h-3 text-amber-500" />
                                  )}
                                  <span className="text-xs font-medium text-foreground truncate max-w-[120px]">
                                    {cite.name}
                                  </span>
                                  <span className="text-[10px] bg-primary/10 text-primary px-1 rounded-full font-semibold">
                                    {cite.fitScore}%
                                  </span>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            ))}

            {isProcessing && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-start space-x-4">
                <div className="w-6 h-6 rounded-md bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center text-white flex-shrink-0 shadow-sm">
                  <Sparkles className="w-3.5 h-3.5 animate-pulse" />
                </div>
                <div className="flex space-x-1 items-center h-6">
                  <div className="w-1.5 h-1.5 rounded-full bg-border animate-bounce" />
                  <div className="w-1.5 h-1.5 rounded-full bg-border animate-bounce [animation-delay:0.2s]" />
                  <div className="w-1.5 h-1.5 rounded-full bg-border animate-bounce [animation-delay:0.4s]" />
                </div>
              </motion.div>
            )}
            
            <div ref={scrollRef} />
          </div>
        </div>
      </div>

      {/* Floating Input Bar */}
      <div className="absolute bottom-6 left-0 right-0 px-6 md:px-12 flex justify-center z-30">
        <form 
          onSubmit={handleSubmit} 
          className="w-full max-w-3xl bg-background/70 backdrop-blur-xl border border-border shadow-premium-panel rounded-2xl p-2 flex flex-col transition-all focus-within:ring-2 focus-within:ring-primary/20"
        >
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full bg-transparent border-none focus:outline-none focus:ring-0 resize-none min-h-[56px] px-3 pt-3 text-foreground placeholder:text-muted-foreground text-[15px] outline-none"
            placeholder="Ask anything..."
            rows={1}
          />
          
          <div className="flex items-center justify-between px-2 pb-1 pt-2">
            <div className="flex items-center space-x-1">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="p-1.5 text-muted-foreground hover:bg-surface hover:text-foreground rounded-md transition-colors cursor-pointer"
              >
                <Paperclip className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={handleMicToggle}
                className={cn(
                  "p-1.5 rounded-md transition-colors cursor-pointer",
                  isMicActive ? "bg-red-500/10 text-red-500" : "text-muted-foreground hover:bg-surface hover:text-foreground"
                )}
              >
                <Mic className="w-4 h-4" />
              </button>
            </div>
            
            <button
              type="submit"
              disabled={!inputText.trim() || isProcessing}
              className={cn(
                "p-1.5 rounded-md transition-all flex items-center justify-center",
                !inputText.trim() || isProcessing 
                  ? "opacity-50 text-muted-foreground cursor-not-allowed" 
                  : "bg-primary text-background cursor-pointer hover:bg-primary/90 shadow-sm"
              )}
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileUpload}
        className="hidden"
        accept=".txt,.md,.xlsx,.csv"
      />

      <AnimatePresence>
        {activeCitation && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-background/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
          >
            <motion.div 
              initial={{ scale: 0.95 }} animate={{ scale: 1 }} exit={{ scale: 0.95 }}
              className="bg-background max-w-xl w-full rounded-2xl p-6 border border-border shadow-premium-panel space-y-4"
            >
              <div className="flex items-center justify-between border-b border-border pb-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-surface rounded-md border border-border text-foreground">
                    <FileText className="w-4 h-4" />
                  </div>
                  <h5 className="font-semibold text-foreground text-sm tracking-tight">
                    {activeCitation.name}
                  </h5>
                </div>
                <span className="text-[10px] font-semibold text-foreground bg-surface px-2 py-1 rounded-md border border-border">
                  {activeCitation.fitScore}% Match
                </span>
              </div>
              
              <div className="bg-surface p-4 rounded-xl text-sm text-muted-foreground leading-relaxed font-mono max-h-72 overflow-y-auto whitespace-pre-line border border-border/50">
                {activeCitation.snippet}
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
