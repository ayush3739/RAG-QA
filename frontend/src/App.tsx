import React, { useEffect, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Sidebar from "./components/Sidebar";
import ChatWorkspace from "./components/ChatWorkspace";
import DashboardView from "./components/DashboardView";
import SessionsView from "./components/SessionsView";
import SettingsView from "./components/SettingsView";
import AuthView from "./components/AuthView";
import { DocumentsView } from "./components/DocumentsView";
import DocumentDetailsView from "./components/DocumentDetailsView";
import { Message, Conversation, DocType, SourceDocument, Citation } from "./types";
import { Menu, Database, Sparkles } from "lucide-react";
import { useStore } from "./store/useStore";
import { api } from "./lib/api";

export default function App() {
  const queryClient = useQueryClient();
  
  const currentTab = useStore(s => s.currentTab);
  const setCurrentTab = useStore(s => s.setCurrentTab);
  const isMobileSidebarOpen = useStore(s => s.isMobileSidebarOpen);
  const setIsMobileSidebarOpen = useStore(s => s.setIsMobileSidebarOpen);
  const theme = useStore(s => s.theme);
  const isAuthenticated = useStore(s => s.isAuthenticated);

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);
  
  const documents = useStore(s => s.documents);
  const setDocuments = useStore(s => s.setDocuments);
  const toggleDocumentActive = useStore(s => s.toggleDocumentActive);
  
  const conversations = useStore(s => s.conversations);
  const setConversations = useStore(s => s.setConversations);
  
  const activeConvId = useStore(s => s.activeConvId);
  const setActiveConvId = useStore(s => s.setActiveConvId);
  
  const params = useStore(s => s.params);
  const setParams = useStore(s => s.setParams);
  
  const isProcessing = useStore(s => s.isProcessing);
  const setIsProcessing = useStore(s => s.setIsProcessing);
  
  const setUser = useStore(s => s.setUser);

  // Queries to sync data from backend
  useQuery({
    queryKey: ['me'],
    queryFn: async () => {
      const data = await api.getMe();
      setUser(data);
      return data;
    },
    enabled: isAuthenticated,
    retry: 1,
    refetchOnWindowFocus: false,
  });
  useQuery({
    queryKey: ['documents'],
    queryFn: async () => {
      const data = await api.getDocuments();
      const docs = data.documents || [];
      const mappedDocs = docs.map((d: any) => ({
        id: d.public_id,
        name: d.name,
        type: d.mime_type?.includes("pdf") ? "pdf" : d.mime_type?.includes("spreadsheet") || d.mime_type?.includes("csv") ? "spreadsheet" : "doc",
        size: Math.round(d.file_size_kb / 1024) + " MB",
        sizeKb: d.file_size_kb,
        addedAt: "Just now",
        summary: d.status === "indexed" ? "Indexed successfully" : d.status === "failed" ? "Indexing failed" : "Processing...",
        active: true,
        chunkCount: d.chunk_count || 0,
        status: d.status
      }));
      setDocuments(mappedDocs);
      return mappedDocs;
    },
    enabled: isAuthenticated,
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const { data: sessionDocsData } = useQuery({
    queryKey: ['sessionDocuments', activeConvId],
    queryFn: async () => {
      if (!activeConvId) return [];
      const data = await api.getSessionDocuments(activeConvId);
      const docs = data.documents || [];
      return docs.map((d: any) => ({
        id: d.public_id,
        name: d.name,
        type: d.mime_type?.includes("pdf") ? "pdf" : d.mime_type?.includes("spreadsheet") || d.mime_type?.includes("csv") ? "spreadsheet" : "doc",
        size: Math.round(d.file_size_kb / 1024) + " MB",
        sizeKb: d.file_size_kb,
        addedAt: "Just now",
        summary: d.status === "indexed" ? "Indexed successfully" : d.status === "failed" ? "Indexing failed" : "Processing...",
        active: true,
        chunkCount: d.chunk_count || 0,
        status: d.status,
        public_id: d.public_id
      }));
    },
    enabled: isAuthenticated && !!activeConvId && currentTab === "conversations",
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const sessionDocuments = sessionDocsData || [];

  const [activeMessages, setActiveMessages] = useState<Message[]>([]);
  const [draftLinkedDocIds, setDraftLinkedDocIds] = useState<string[]>([]);

  // Sync URL hash to state and vice versa
  useEffect(() => {
    const handleHashChange = () => {
      if (!isAuthenticated) return;
      const hash = window.location.hash;
      if (!hash) {
        setCurrentTab("dashboard");
        setActiveConvId(null);
        return;
      }

      const parts = hash.replace("#/", "").split("/");
      const route = parts[0];
      const param = parts[1];

      if (route === "dashboard") {
        setCurrentTab("dashboard");
      } else if (route === "documents") {
        if (param === "details") {
          setCurrentTab("document_details");
        } else {
          setCurrentTab("documents");
        }
      } else if (route === "chats") {
        setCurrentTab("chats");
      } else if (route === "settings") {
        setCurrentTab("settings");
      } else if (route === "chat") {
        setCurrentTab("conversations");
        if (param === "new") {
          setActiveConvId("");
        } else if (param) {
          setActiveConvId(param);
        } else {
          setActiveConvId("");
        }
      }
    };

    window.addEventListener("hashchange", handleHashChange);
    if (isAuthenticated) {
      handleHashChange();
    }
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, [setCurrentTab, setActiveConvId, isAuthenticated]);

  // Sync state to URL hash
  useEffect(() => {
    if (!isAuthenticated) return;

    let targetHash = "";
    if (currentTab === "dashboard") {
      targetHash = "#/dashboard";
    } else if (currentTab === "documents") {
      targetHash = "#/documents";
    } else if (currentTab === "document_details") {
      targetHash = "#/documents/details";
    } else if (currentTab === "chats") {
      targetHash = "#/chats";
    } else if (currentTab === "settings") {
      targetHash = "#/settings";
    } else if (currentTab === "conversations") {
      if (activeConvId === "") {
        targetHash = "#/chat/new";
      } else if (activeConvId) {
        targetHash = `#/chat/${activeConvId}`;
      } else {
        targetHash = "#/chat/new";
      }
    }

    if (targetHash && window.location.hash !== targetHash) {
      window.history.pushState(null, "", targetHash);
    }
  }, [currentTab, activeConvId, isAuthenticated]);

  const { data: historyMessages } = useQuery({
    queryKey: ['sessionHistory', activeConvId],
    queryFn: async () => {
      if (!activeConvId) return [];
      const histRes = await api.getSessionHistory(activeConvId);
      const messages = histRes.messages || [];
      return messages.map((m: any, idx: number) => ({
        id: m.message_id ? String(m.message_id) : `msg-${m.created_at}-${idx}`,
        dbId: m.message_id || undefined,
        sender: m.role,
        text: m.content,
        timestamp: m.created_at,
        citations: m.citations || [],
        chunks: m.chunks || undefined
      }));
    },
    enabled: isAuthenticated && !!activeConvId && currentTab === "conversations",
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const lastLoadedConvIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (activeConvId !== lastLoadedConvIdRef.current) {
      if (historyMessages) {
        setActiveMessages(historyMessages);
        lastLoadedConvIdRef.current = activeConvId ?? null;
      } else if (activeConvId === "") {
        setActiveMessages([]);
        lastLoadedConvIdRef.current = "";
      }
    } else if (historyMessages && activeMessages.length === 0 && historyMessages.length > 0) {
      setActiveMessages(historyMessages);
    }
  }, [activeConvId, historyMessages, activeMessages.length]);

  useQuery({
    queryKey: ['sessions'],
    queryFn: async () => {
      const sessData = await api.getSessions();
      const sessions = sessData.sessions || [];
      
      const mappedSessions = sessions.map((s: any) => ({
        id: s.session_id,
        title: s.title,
        timestamp: s.updated_at || new Date().toISOString(),
        messages: []
      }));
      if (mappedSessions.length > 0) {
        setConversations(mappedSessions);
        if (activeConvId === null) setActiveConvId(mappedSessions[0].id);
      }
      return mappedSessions;
    },
    enabled: isAuthenticated,
    retry: 1,
    refetchOnWindowFocus: false,
  });

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isProcessing) return;

    let targetConvId = activeConvId;

    if (!targetConvId) {
      setIsProcessing(true);
      try {
        const createRes = await api.createSession();
        targetConvId = createRes.session_id;
        
        const words = text.trim().split(/\s+/);
        const title = words.length <= 6 
          ? text.trim() 
          : words.slice(0, 6).join(" ") + "...";
        
        const newConv = {
          id: targetConvId,
          title,
          timestamp: new Date().toISOString(),
          messages: []
        };
        setConversations(prev => [newConv, ...prev]);
        setActiveConvId(targetConvId);
        setActiveMessages([]);
        
        // Save the first query title immediately in the database
        await api.renameSession(targetConvId, title);
        
        for (const docId of draftLinkedDocIds) {
          await api.linkDocumentToSession(targetConvId, docId);
        }
        setDraftLinkedDocIds([]);
      } catch (err) {
        setIsProcessing(false);
        return;
      }
    }

    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      sender: "user",
      text,
      timestamp: new Date().toISOString(),
    };

    const finalConvId = targetConvId;

    setConversations((prev) => prev.map((conv) => {
      if (conv.id === finalConvId) {
        return {
          ...conv,
          timestamp: new Date().toISOString(),
        };
      }
      return conv;
    }));
    setActiveMessages(prev => [...prev, userMsg]);
    setIsProcessing(true);

    const botMsgId = `msg-${Date.now() + 1}`;
    setActiveMessages(prev => [...prev, {
      id: botMsgId,
      sender: "assistant",
      text: "",
      timestamp: new Date().toISOString(),
    }]);

    try {
      const res = await fetch(api.getChatEndpoint(finalConvId), {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${useStore.getState().accessToken}`
        },
        body: JSON.stringify({ question: text })
      });

      if (!res.ok || !res.body) throw new Error("Chat failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      var currentText = "";
      var finalCitations: Citation[] = [];
      var finalChunks: any = undefined;
      var finalMessageId: number | undefined = undefined;
      let currentEvent = "message";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (let line of lines) {
          if (line.endsWith('\r')) {
            line = line.slice(0, -1);
          }
          
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7);
          } else if (line.startsWith('data: ') || line === 'data:') {
            const dataStr = line.startsWith('data: ') ? line.slice(6) : '';
            if (dataStr === '[DONE]') continue;
            
            if (currentEvent === 'token') {
              const tokenText = dataStr === '' ? '\n' : dataStr;
              currentText += tokenText;
              setActiveMessages((prev) => prev.map(m => m.id === botMsgId ? { ...m, text: currentText } : m));
            } else if (currentEvent === 'metadata') {
              try {
                const data = JSON.parse(dataStr);
                // Map backend source fields to frontend Citation shape
                const mappedSources = (data.sources || []).map((s: any) => {
                  if (s.type === 'web') {
                    return {
                      type: 'web',
                      name: s.title || s.url || 'Web',
                      title: s.title,
                      url: s.url,
                      snippet: s.content || s.excerpt || '',
                      fitScore: null,
                    };
                  }
                  // Document source
                  const rawSource: string = s.source || '';
                  const fileName = rawSource.split('/').pop()?.split('_').slice(1).join('_') || rawSource.split('/').pop() || 'Source';
                  return {
                    type: 'document',
                    name: fileName,
                    snippet: s.excerpt || s.text || '',
                    fitScore: s.reranker_score != null ? Math.round(s.reranker_score * 100) : null,
                    page: s.page,
                    chunk_id: s.chunk_id,
                  };
                });
                finalCitations = mappedSources;
                finalChunks = data.chunks || undefined;
                finalMessageId = data.message_id || undefined;
                setActiveMessages((prev) => prev.map(m => m.id === botMsgId ? { 
                  ...m, 
                  dbId: data.message_id || undefined,
                  citations: mappedSources,
                  chunks: data.chunks || undefined
                } : m));
              } catch (e) {}
            }
          }
        }
      }
    } catch (error) {
       const errorMsgText = `⚠️ **Workspace Query Refusal**: Connection to backend failed.`;
       currentText = errorMsgText;
       const errorMsg: Message = {
        id: `msg-error-${Date.now()}`,
        sender: "assistant",
        text: errorMsgText,
        timestamp: new Date().toISOString(),
      };
      setActiveMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);

      const finalAssistantMsg: Message = {
        id: botMsgId,
        dbId: finalMessageId,
        sender: "assistant",
        text: currentText || "⚠️ **Workspace Query Refusal**: Connection to backend failed.",
        timestamp: new Date().toISOString(),
        citations: finalCitations,
        chunks: finalChunks
      };

      queryClient.setQueryData(['sessionHistory', finalConvId], (old: any) => {
        const filtered = (old || []).filter((m: any) => m.id !== botMsgId && m.id !== userMsg.id && !m.id.startsWith("msg-error-"));
        return [...filtered, userMsg, finalAssistantMsg];
      });

      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    }
  };

  const createSessionMutation = useMutation({
    mutationFn: async ({ title, docIds }: { title: string, docIds: string[] }) => {
      const createRes = await api.createSession();
      const sessionId = createRes.session_id;
      
      // Link active documents to this session
      for (const docId of docIds) {
        await api.linkDocumentToSession(sessionId, docId);
      }
      
      // If we wanted to rename it right away we could call api.renameSession here
      return sessionId;
    },
    onSuccess: (newSessionId) => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setActiveConvId(newSessionId);
    }
  });

  const handleNewResearch = () => {
    setCurrentTab("conversations");
    setIsMobileSidebarOpen(false);
    setActiveConvId("");
    setDraftLinkedDocIds([]);
  };

  const handleTakeToChat = (topic: string, selectedDocIds: string[]) => {
    setCurrentTab("conversations");
    
    createSessionMutation.mutate({ title: topic, docIds: selectedDocIds }, {
      onSuccess: (newSessionId) => {
         setTimeout(() => {
           // Temporarily set active id so handleSendMessage can work
           useStore.getState().setActiveConvId(newSessionId);
           handleSendMessage(topic);
         }, 100);
      }
    });
  };

  const uploadDocMutation = useMutation({
    mutationFn: async ({ file, sessionId }: { file: File, sessionId?: string }) => {
      const res = await api.uploadDocument(file, sessionId);
      if (res.job_id) {
        const es = new EventSource(`http://localhost:8000/api/v1/documents/status/${res.job_id}`);
        es.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.status === "completed" || data.status === "failed") {
              es.close();
              queryClient.invalidateQueries({ queryKey: ['documents'] });
              if (sessionId) {
                queryClient.invalidateQueries({ queryKey: ['sessionDocuments', sessionId] });
              }
            }
          } catch (e) {
            console.error("SSE parse error", e);
          }
        };
        es.onerror = () => {
          es.close();
          queryClient.invalidateQueries({ queryKey: ['documents'] });
          if (sessionId) {
            queryClient.invalidateQueries({ queryKey: ['sessionDocuments', sessionId] });
          }
        };
      }
      return res;
    },
    onSuccess: (data, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      if (sessionId) {
        queryClient.invalidateQueries({ queryKey: ['sessionDocuments', sessionId] });
      } else if (data?.document_id) {
        setDraftLinkedDocIds(prev => Array.from(new Set([...prev, data.document_id])));
      }
    }
  });

  const handleAddDocument = async (file: File, sessionId?: string) => {
    uploadDocMutation.mutate({ file, sessionId });
  };

  const linkDocMutation = useMutation({
    mutationFn: async ({ sessionId, documentId }: { sessionId: string, documentId: string }) => {
      await api.linkDocumentToSession(sessionId, documentId);
    },
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ['sessionDocuments', sessionId] });
    }
  });

  const handleLinkDocument = async (sessionId: string, documentId: string) => {
    if (!sessionId) {
      setDraftLinkedDocIds(prev => prev.includes(documentId) ? prev : [...prev, documentId]);
      return;
    }
    const alreadyLinked = sessionDocuments.some((d: any) => d.id === documentId || d.public_id === documentId);
    if (alreadyLinked) return;
    linkDocMutation.mutate({ sessionId, documentId });
  };

  const deleteDocMutation = useMutation({
    mutationFn: async (id: string) => {
      await api.deleteDocument(id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    }
  });

  const handleDeleteDocument = (id: string) => {
    deleteDocMutation.mutate(id);
  };

  const deleteSessionMutation = useMutation({
    mutationFn: async (id: string) => {
      await api.deleteSession(id);
    }
  });

  const handleDeleteSession = (id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id));
    deleteSessionMutation.mutate(id);
  };

  const renameSessionMutation = useMutation({
    mutationFn: async ({ id, newName }: { id: string, newName: string }) => {
      await api.renameSession(id, newName);
    }
  });

  const handleRenameSession = (id: string, newName: string) => {
    setConversations(prev => prev.map(c => c.id === id ? { ...c, title: newName } : c));
    renameSessionMutation.mutate({ id, newName });
  };

  const handleSelectSession = (id: string) => {
    setActiveConvId(id);
    setCurrentTab("conversations");
  };

  const activeConversation = activeConvId === "" ? null : conversations.find((c) => c.id === activeConvId) || conversations[0];

  if (!isAuthenticated) {
    return <AuthView />;
  }

  return (
    <div className="bg-background text-foreground flex h-screen overflow-hidden font-sans selection:bg-primary/20 selection:text-primary">
      {/* Mobile Top Navigation rail bar */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-14 bg-background/80 backdrop-blur-md border-b border-border flex items-center justify-between px-6 z-40 select-none">
        <div className="flex items-center space-x-2">
          <Menu
            className="w-6 h-6 text-primary cursor-pointer active:scale-90 transition-transform"
            onClick={() => setIsMobileSidebarOpen(true)}
          />
          <h1 className="text-sm font-black text-primary tracking-tight">DocuMind</h1>
        </div>
        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-primary" />
          <span className="text-[10px] uppercase tracking-wider font-extrabold text-secondary">v4.2.0</span>
        </div>
      </div>

      {isMobileSidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-background/50 backdrop-blur-sm z-40"
          role="button"
          tabIndex={0}
          aria-label="Close sidebar"
          onClick={() => setIsMobileSidebarOpen(false)}
          onKeyDown={(e) => e.key === "Enter" || e.key === " " ? setIsMobileSidebarOpen(false) : undefined}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed md:relative top-0 bottom-0 left-0 z-50 md:z-auto transition-transform duration-300 md:transform-none ${isMobileSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}`}>
        <Sidebar
          currentTab={currentTab}
          setCurrentTab={(tab) => { setCurrentTab(tab); setIsMobileSidebarOpen(false); }}
          onNewResearch={handleNewResearch}
          conversationsCount={conversations.length}
        />
      </div>

      <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden pt-14 md:pt-0">
        {currentTab === "dashboard" && (
          <DashboardView
            documents={documents}
            conversations={conversations}
            setCurrentTab={setCurrentTab}
            onNewResearch={handleNewResearch}
            onSelectSession={handleSelectSession}
            onTriggerUploadModal={() => setCurrentTab("documents")}
          />
        )}
        {currentTab === "conversations" && (
          <div className="flex-1 flex overflow-hidden">
            <ChatWorkspace
              messages={activeMessages}
              isProcessing={isProcessing}
              onSendMessage={handleSendMessage}
              documents={activeConvId ? sessionDocuments : documents.filter(d => draftLinkedDocIds.includes(d.id))}
              allDocuments={documents}
              activeConvId={activeConvId}
              onAddDocument={(file) => handleAddDocument(file, activeConvId)}
              onLinkDocument={(docId) => handleLinkDocument(activeConvId, docId)}
              onUnlinkDocument={!activeConvId ? ((docId) => setDraftLinkedDocIds(prev => prev.filter(id => id !== docId))) : undefined}
              selectedModel={params.selectedModel}
              sessionTitle={activeConvId === "" ? "New Chat" : (activeConversation?.title || "New Chat")}
            />
          </div>
        )}
        {currentTab === "documents" && (
          <DocumentsView
            documents={documents}
            onToggleActive={toggleDocumentActive}
            onDeleteDocument={handleDeleteDocument}
            onAddDocument={handleAddDocument}
            onSelectDocument={(id) => {
              useStore.getState().setActiveDocumentId(id);
              setCurrentTab("document_details");
            }}
          />
        )}
        {currentTab === "document_details" && (
          <DocumentDetailsView
            document={documents.find((d) => d.id === useStore.getState().activeDocumentId)}
            onClose={() => setCurrentTab("documents")}
          />
        )}
        {currentTab === "chats" && (
          <SessionsView
            conversations={conversations}
            documents={documents}
            onSelectSession={handleSelectSession}
            onDeleteSession={handleDeleteSession}
            onRenameSession={handleRenameSession}
          />
        )}
        {currentTab === "settings" && (
          <SettingsView params={params} onParamChange={setParams} />
        )}
      </main>
    </div>
  );
}
