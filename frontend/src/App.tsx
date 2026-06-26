import React, { useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Sidebar from "./components/Sidebar";
import MetadataPanel from "./components/MetadataPanel";
import ChatWorkspace from "./components/ChatWorkspace";
import DashboardView from "./components/DashboardView";
import ResearchView from "./components/ResearchView";
import SessionsView from "./components/SessionsView";
import SettingsView from "./components/SettingsView";
import { LibraryView } from "./components/OtherPanels";
import { Message, Conversation, DocType, SourceDocument } from "./types";
import { Menu, Database, Sparkles } from "lucide-react";
import { useStore } from "./store/useStore";

export default function App() {
  const queryClient = useQueryClient();
  
  const currentTab = useStore(s => s.currentTab);
  const setCurrentTab = useStore(s => s.setCurrentTab);
  const isMobileSidebarOpen = useStore(s => s.isMobileSidebarOpen);
  const setIsMobileSidebarOpen = useStore(s => s.setIsMobileSidebarOpen);
  const theme = useStore(s => s.theme);

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

  // Queries to sync data from backend
  useQuery({
    queryKey: ['documents'],
    queryFn: async () => {
      const res = await fetch("/api/v1/documents");
      if (!res.ok) throw new Error("Failed to fetch documents");
      const data = await res.json();
      if (data && data.length > 0) setDocuments(data);
      return data;
    },
    retry: 1,
    refetchOnWindowFocus: false,
  });

  useQuery({
    queryKey: ['sessions'],
    queryFn: async () => {
      const sessRes = await fetch("/api/v1/sessions");
      if (!sessRes.ok) throw new Error("Failed to fetch sessions");
      const sessions = await sessRes.json();
      
      const detailedSessions = await Promise.all(
        sessions.map(async (s: any) => {
          const histRes = await fetch(`/api/v1/sessions/${s.id}/history`);
          const messages = histRes.ok ? await histRes.json() : [];
          return {
            ...s,
            messages,
            title: s.name,
            timestamp: s.created_at
          };
        })
      );
      if (detailedSessions && detailedSessions.length > 0) {
        setConversations(detailedSessions);
        if (!activeConvId) setActiveConvId(detailedSessions[0].id);
      }
      return detailedSessions;
    },
    retry: 1,
    refetchOnWindowFocus: false,
  });

  // Mutations
  const chatMutation = useMutation({
    mutationFn: async ({ text, convId }: { text: string, convId: string }) => {
      const activeCorpus = documents.filter((d) => d.active);
      const res = await fetch(`/api/v1/chat/${convId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          activeSources: activeCorpus,
          systemInstruction: params.systemInstruction || undefined,
          temperature: params.temperature,
          model: params.selectedModel,
        }),
      });
      if (!res.ok) throw new Error("Primary API failed");
      return res.json();
    },
    onError: async (error, { text, convId }) => {
      // Fallback
      try {
        const fallbackRes = await fetch("/api/rag/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: text,
            activeSources: documents.filter((d) => d.active),
            systemInstruction: params.systemInstruction || undefined,
            temperature: params.temperature,
            model: params.selectedModel,
          }),
        });
        const fallbackData = await fallbackRes.json();
        
        const botMsg: Message = {
          id: `msg-${Date.now() + 1}`,
          sender: "assistant",
          text: fallbackData.answer,
          timestamp: new Date().toISOString(),
          citations: fallbackData.citations,
        };

        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id === convId) {
              return {
                ...conv,
                messages: [...conv.messages, botMsg],
                timestamp: new Date().toISOString(),
              };
            }
            return conv;
          })
        );
      } catch (innerErr) {
         const errorMsg: Message = {
          id: `msg-${Date.now() + 1}`,
          sender: "assistant",
          text: `⚠️ **Workspace Query Refusal**: Failed to communicate with the Gemini vector server. Ensure key credentials are correct in google AI Studio secrets.`,
          timestamp: new Date().toISOString(),
        };
        setConversations((prev) =>
          prev.map((conv) => (conv.id === convId ? { ...conv, messages: [...conv.messages, errorMsg] } : conv))
        );
      }
    },
    onSettled: () => {
      setIsProcessing(false);
    }
  });

  const handleSendMessage = (text: string) => {
    if (!text.trim() || isProcessing || !activeConvId) return;

    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      sender: "user",
      text,
      timestamp: new Date().toISOString(),
    };

    setConversations((prev) => prev.map((conv) => {
      if (conv.id === activeConvId) {
        return {
          ...conv,
          messages: [...conv.messages, userMsg],
          timestamp: new Date().toISOString(),
        };
      }
      return conv;
    }));
    setIsProcessing(true);
    chatMutation.mutate({ text, convId: activeConvId });
  };

  const createSessionMutation = useMutation({
    mutationFn: async ({ title, docIds }: { title: string, docIds: string[] }) => {
      await fetch("/api/v1/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: title, attachedDocIds: docIds })
      });
    }
  });

  const handleNewResearch = () => {
    const newId = `conv-${Date.now()}`;
    const newTitle = "New Research Workspace";
    
    const newConv: Conversation = {
      id: newId,
      title: newTitle,
      timestamp: new Date().toISOString(),
      messages: [],
    };

    setConversations((prev) => [newConv, ...prev]);
    setActiveConvId(newId);
    setCurrentTab("conversations");
    setIsMobileSidebarOpen(false);

    createSessionMutation.mutate({ 
      title: newTitle, 
      docIds: documents.filter(d => d.active).map(d => d.id) 
    });
  };

  const handleTakeToChat = (topic: string, selectedDocIds: string[]) => {
    const newId = `conv-${Date.now()}`;
    
    const newConv: Conversation = {
      id: newId,
      title: topic.substring(0, 30) + "...",
      timestamp: new Date().toISOString(),
      messages: [],
    };

    setConversations((prev) => [newConv, ...prev]);
    setActiveConvId(newId);
    setCurrentTab("conversations");

    createSessionMutation.mutate({ 
      title: topic.substring(0, 30) + "...", 
      docIds: selectedDocIds 
    });

    setTimeout(() => {
      handleSendMessage(topic);
    }, 500);
  };

  const handleAddDocument = (name: string, content: string, type: DocType, size: string) => {
    const newDoc: SourceDocument = {
      id: `doc-${Date.now()}`,
      name,
      content,
      type,
      size,
      active: true,
      addedAt: "Added Just Now",
      summary: "1. Uploaded successfully.\n2. Context processed.",
      chunkCount: 12,
      embeddingModel: "text-embedding-3-large"
    };
    setDocuments((prev) => [newDoc, ...prev]);
  };

  const deleteDocMutation = useMutation({
    mutationFn: async (id: string) => {
      await fetch(`/api/v1/documents/${id}`, { method: "DELETE" });
    }
  });

  const handleDeleteDocument = (id: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== id));
    deleteDocMutation.mutate(id);
  };

  const deleteSessionMutation = useMutation({
    mutationFn: async (id: string) => {
      await fetch(`/api/v1/sessions/${id}`, { method: "DELETE" });
    }
  });

  const handleDeleteSession = (id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id));
    deleteSessionMutation.mutate(id);
  };

  const renameSessionMutation = useMutation({
    mutationFn: async ({ id, newName }: { id: string, newName: string }) => {
      await fetch(`/api/v1/sessions/${id}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName })
      });
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

  const activeConversation = conversations.find((c) => c.id === activeConvId) || conversations[0];

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
        <div className="md:hidden fixed inset-0 bg-background/50 backdrop-blur-sm z-40" onClick={() => setIsMobileSidebarOpen(false)} />
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

      <main className="flex-1 flex flex-col min-w-0 pt-14 md:pt-0">
        {currentTab === "dashboard" && (
          <DashboardView
            documents={documents}
            conversations={conversations}
            setCurrentTab={setCurrentTab}
            onNewResearch={handleNewResearch}
            onSelectSession={handleSelectSession}
            onTriggerUploadModal={() => setCurrentTab("library")}
          />
        )}
        {currentTab === "conversations" && (
          <div className="flex-1 flex overflow-hidden">
            <ChatWorkspace
              messages={activeConversation ? activeConversation.messages : []}
              isProcessing={isProcessing}
              onSendMessage={handleSendMessage}
              documents={documents}
              onAddDocument={handleAddDocument}
              selectedModel={params.selectedModel}
            />
            <MetadataPanel
              documents={documents}
              toggleDocumentActive={toggleDocumentActive}
              params={params}
              onParamChange={setParams}
            />
          </div>
        )}
        {currentTab === "library" && (
          <LibraryView
            documents={documents}
            onToggleActive={toggleDocumentActive}
            onDeleteDocument={handleDeleteDocument}
            onAddDocument={handleAddDocument}
          />
        )}
        {currentTab === "research" && (
          <ResearchView documents={documents} onTakeToChat={handleTakeToChat} />
        )}
        {currentTab === "sessions" && (
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
