import React, { useEffect, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Sidebar from "./components/Sidebar";
import ChatWorkspace from "./components/ChatWorkspace";
import DashboardView from "./components/DashboardView";
import SessionsView from "./components/SessionsView";
import SettingsView from "./components/SettingsView";
import AuthView from "./components/AuthView";
import VerifyEmailView from "./components/VerifyEmailView";
import ResetPasswordView from "./components/ResetPasswordView";
import { DocumentsView } from "./components/DocumentsView";
import DocumentDetailsView from "./components/DocumentDetailsView";
import { Message, Conversation, DocType, SourceDocument, Citation } from "./types";
import { Menu, Database, Sparkles } from "lucide-react";
import { useStore } from "./store/useStore";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { api } from "./lib/api";
import { Toaster, toast } from 'sonner';
import { motion, AnimatePresence } from "framer-motion";
import LandingPage from "./landing/page";
import ArchitecturePage from "./landing/ArchitecturePage";
import NoInternetBanner from "./components/NoInternetBanner";

const normalizeSourceScore = (score: any): number | null => {
  if (score == null) return null;
  const numeric = Number(score);
  if (!Number.isFinite(numeric)) return null;
  const normalized = numeric >= 0 && numeric <= 1
    ? numeric
    : 1 / (1 + Math.exp(-numeric));
  return Math.round(Math.max(0, Math.min(1, normalized)) * 100);
};

const formatFileSizeFromKb = (sizeKb: any): string => {
  const numericKb = Number(sizeKb);
  if (!Number.isFinite(numericKb) || numericKb <= 0) return "0 KB";

  if (numericKb < 1024) {
    return `${Math.round(numericKb)} KB`;
  }

  const sizeMb = numericKb / 1024;
  return `${sizeMb.toFixed(2).replace(/\.?0+$/, "")} MB`;
};

const mapSourcesToCitations = (sources: any): Citation[] => {
  const sourceList = Array.isArray(sources)
    ? sources
    : Array.isArray(sources?.sources)
      ? sources.sources
      : [];

  return sourceList.map((s: any) => {
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

    const rawSource: string = s.source || '';
    const fileName = rawSource.split('/').pop()?.split('_').slice(1).join('_') || rawSource.split('/').pop() || s.name || 'Source';
    return {
      type: 'document',
      name: fileName,
      snippet: s.excerpt || s.text || '',
      fitScore: null,
      page: s.page,
      chunk_id: s.chunk_id,
    };
  });
};

function getTokenFromSearch(search: string): string | null {
  const params = new URLSearchParams(search);
  return params.get('token');
}

export default function App() {
  const queryClient = useQueryClient();
  const [hash, setHash] = useState(window.location.hash);
  const [pathname, setPathname] = useState(window.location.pathname);
  const [search, setSearch] = useState(window.location.search);

  useEffect(() => {
    const handleHash = () => {
      setHash(window.location.hash);
      setPathname(window.location.pathname);
      setSearch(window.location.search);
    };
    window.addEventListener("hashchange", handleHash);
    window.addEventListener("popstate", handleHash);
    return () => {
      window.removeEventListener("hashchange", handleHash);
      window.removeEventListener("popstate", handleHash);
    };
  }, []);
  
  const currentTab = useStore(s => s.currentTab);
  const setCurrentTab = useStore(s => s.setCurrentTab);
  const isMobileSidebarOpen = useStore(s => s.isMobileSidebarOpen);
  const setIsMobileSidebarOpen = useStore(s => s.setIsMobileSidebarOpen);
  const theme = useStore(s => s.theme);
  const isAuthenticated = useStore(s => s.isAuthenticated);
  const user = useStore(s => s.user);
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Handle OAuth Callback Tokens & Errors
  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    let accessToken = searchParams.get('access_token');
    let refreshToken = searchParams.get('refresh_token');
    let oauthError = searchParams.get('oauth_error');
    let errorMessage = searchParams.get('message');

    if (!accessToken && !oauthError && window.location.hash.includes('?')) {
      const hashParts = window.location.hash.split('?');
      if (hashParts.length > 1) {
        const hashParams = new URLSearchParams(hashParts[1]);
        accessToken = accessToken || hashParams.get('access_token');
        refreshToken = refreshToken || hashParams.get('refresh_token');
        oauthError = oauthError || hashParams.get('oauth_error');
        errorMessage = errorMessage || hashParams.get('message');
      }
    }

    if (accessToken) {
      useStore.getState().setAccessToken(accessToken);
      toast.success("Successfully logged in via OAuth!");
      window.history.replaceState({}, document.title, window.location.pathname + "#/dashboard");
      useStore.getState().setCurrentTab("dashboard");
    } else if (oauthError) {
      const displayMsg = errorMessage || "OAuth authentication failed. Please try again.";
      toast.error(`Login Error: ${displayMsg}`);
      window.history.replaceState({}, document.title, window.location.pathname + "#/auth");
      useStore.getState().setCurrentTab("auth");
    }
  }, []);

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

  const updateProfileMutation = useMutation({
    mutationFn: async (body: { name?: string; email?: string }) => {
      const data = await api.updateMe(body);
      setUser(data);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['me'] });
    }
  });


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
        size: formatFileSizeFromKb(d.file_size_kb),
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
        size: formatFileSizeFromKb(d.file_size_kb),
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
  const [graphStatuses, setGraphStatuses] = useState<any[]>([]);

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
        citations: mapSourcesToCitations(m.citations),
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
        lastLoadedConvIdRef.current = targetConvId; // Prevent useEffect from wiping our optimistic message
        
        // Save the first query title immediately in the database
        await api.renameSession(targetConvId, title);
        
        const linkedDraftDocIds = [...draftLinkedDocIds];
        for (const docId of linkedDraftDocIds) {
          await api.linkDocumentToSession(targetConvId, docId);
        }
        if (linkedDraftDocIds.length > 0) {
          queryClient.setQueryData(
            ['sessionDocuments', targetConvId],
            documents.filter((doc) => linkedDraftDocIds.includes(doc.id))
          );
          queryClient.invalidateQueries({ queryKey: ['sessionDocuments', targetConvId] });
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
    setGraphStatuses([{ node: "main_router", label: "Deciding route", status: "running" }]);
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
            } else if (currentEvent === 'answer') {
              try {
                currentText = JSON.parse(dataStr);
              } catch {
                currentText = dataStr;
              }
              setActiveMessages((prev) => prev.map(m => m.id === botMsgId ? { ...m, text: currentText } : m));
            } else if (currentEvent === 'graph_status') {
              try {
                const status = JSON.parse(dataStr);
                setGraphStatuses((prev) => {
                  const existingIdx = prev.findIndex((item: any) => item.node === status.node);
                  if (existingIdx === -1) return [...prev, status];
                  return prev.map((item: any, idx) => idx === existingIdx ? { ...item, ...status } : item);
                });
              } catch (e) {}
            } else if (currentEvent === 'metadata') {
              try {
                const data = JSON.parse(dataStr);
                const mappedSources = mapSourcesToCitations(data.sources);
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
      queryClient.invalidateQueries({ queryKey: ['user-activity'] });
      setGraphStatuses([]);
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

  useKeyboardShortcuts({
    'u': () => setCurrentTab("documents"),
    'd': () => setCurrentTab("dashboard"),
    't': () => setCurrentTab("chats"),
    's': () => setCurrentTab("settings"),
    'cmd+k': () => handleNewResearch(),
  });

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
    },
    onError: (error: any) => {
      toast.error(`Upload failed: ${error.message || "An unknown error occurred"}`);
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

    const docToLink = documents.find((doc) => doc.id === documentId);
    if (docToLink) {
      queryClient.setQueryData(['sessionDocuments', sessionId], (old: any) => {
        const currentDocs = Array.isArray(old) ? old : [];
        if (currentDocs.some((doc: any) => doc.id === documentId || doc.public_id === documentId)) {
          return currentDocs;
        }
        return [...currentDocs, docToLink];
      });
    }
    linkDocMutation.mutate({ sessionId, documentId });
  };

  const unlinkDocMutation = useMutation({
    mutationFn: async ({ sessionId, documentId }: { sessionId: string, documentId: string }) => {
      await api.unlinkDocumentFromSession(sessionId, documentId);
    },
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ['sessionDocuments', sessionId] });
    }
  });

  const handleUnlinkDocument = async (sessionId: string, documentId: string) => {
    if (!sessionId) {
      setDraftLinkedDocIds(prev => prev.filter(id => String(id) !== String(documentId)));
      return;
    }
    unlinkDocMutation.mutate({ sessionId, documentId });
  };

  const handleRetryMessage = async (messageId: string) => {
    if (isProcessing) return;
    const msgIdx = activeMessages.findIndex(m => m.id === messageId);
    if (msgIdx === -1) return;

    let userMsgIdx = -1;
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (activeMessages[i].sender === "user") {
        userMsgIdx = i;
        break;
      }
    }
    if (userMsgIdx === -1) return;

    const userQuery = activeMessages[userMsgIdx].text;
    
    // Re-send the query directly so it appends to the bottom and preserves history
    handleSendMessage(userQuery);
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

  // --- Dedicated auth routes: check pathname first, then hash ---
  // Support both /verify-email?token=... and #/verify-email?token=... patterns
  const hashPath = hash.replace(/^#\/?/, '').split('?')[0];
  const hashSearch = hash.includes('?') ? '?' + hash.split('?')[1] : '';

  const resolvedPath = pathname !== '/' ? pathname : '/' + hashPath;
  const resolvedSearch = search || hashSearch;

  if (resolvedPath === '/verify-email') {
    const token = getTokenFromSearch(resolvedSearch);
    return (
      <VerifyEmailView
        token={token || ''}
        onNavigateToLogin={() => window.location.replace('/#/app')}
      />
    );
  }

  if (resolvedPath === '/reset-password') {
    const token = getTokenFromSearch(resolvedSearch);
    return (
      <ResetPasswordView
        token={token || ''}
        onNavigateToLogin={() => window.location.replace('/#/app')}
      />
    );
  }

  const landingHashes = ["", "#", "#/", "#hero", "#how-it-works", "#metrics", "#system-features", "#faq"];
  const isLanding = landingHashes.includes(hash);
  const isArchitecture = hash === "#/architecture";

  if (isArchitecture) {
    return <ArchitecturePage />;
  }

  if (isLanding) {
    return <LandingPage />;
  }

  if (!isAuthenticated) {
    return <AuthView />;
  }

  return (
    <div className="bg-background text-foreground flex h-screen overflow-hidden font-sans relative">
      <Toaster position="top-right" richColors />
      <NoInternetBanner />
      
      {/* Background Ambient Moving Orbs */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none hidden dark:block">
        <div className="ambient-orb orb-1 bg-primary/5 w-[500px] h-[500px] -top-[200px] -left-[200px]" />
        <div className="ambient-orb orb-2 bg-secondary/5 w-[600px] h-[600px] -bottom-[300px] -right-[100px]" />
        <div className="ambient-orb orb-3 bg-violet-600/5 w-[450px] h-[450px] top-[30%] left-[55%]" />
      </div>
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

      <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden pt-14 md:pt-0 relative z-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentTab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="flex-1 flex flex-col min-h-0 min-w-0 overflow-hidden"
          >
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
                  onUnlinkDocument={(docId) => handleUnlinkDocument(activeConvId, docId)}
                  onRetryMessage={handleRetryMessage}
                  selectedModel={params.selectedModel}
                  sessionTitle={activeConvId === "" ? "New Chat" : (activeConversation?.title || "New Chat")}
                  graphStatuses={graphStatuses}
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
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Onboarding Dialog Modal */}
      {showOnboarding && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-lg animate-in fade-in duration-300">
          <div className="w-full max-w-lg p-10 rounded-3xl bg-[#0d1527]/75 border border-white/10 shadow-[0_0_80px_-15px_rgba(59,130,246,0.25)] backdrop-blur-2xl animate-in zoom-in-95 duration-300 relative overflow-hidden">
            {/* Ambient decorative glow inside modal */}
            <div className="absolute -top-24 -left-24 w-48 h-48 rounded-full bg-blue-500/10 blur-[80px] pointer-events-none" />
            <div className="absolute -bottom-24 -right-24 w-48 h-48 rounded-full bg-indigo-500/10 blur-[80px] pointer-events-none" />

            <div className="flex flex-col items-center text-center space-y-6 relative z-10">
              {/* Premium Icon Container */}
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                <Sparkles className="w-8 h-8 text-white animate-pulse" />
              </div>
              
              <div className="space-y-2">
                <h2 className="text-3xl font-extrabold tracking-tight text-white">Welcome to DocuMind</h2>
                <p className="text-sm font-medium text-zinc-400 max-w-xs mx-auto leading-relaxed">
                  Before we begin your workspace experience, what would you like us to call you?
                </p>
              </div>

              <form 
                onSubmit={(e) => {
                  e.preventDefault();
                  const formData = new FormData(e.currentTarget);
                  const newName = formData.get("nickname") as string;
                  if (newName && newName.trim()) {
                    updateProfileMutation.mutate({ name: newName.trim() }, {
                      onSuccess: () => setShowOnboarding(false)
                    });
                  }
                }}
                className="w-full space-y-4 pt-3"
              >
                <div className="space-y-1.5 text-left">
                  <label className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase px-1">Display Name</label>
                  <input
                    type="text"
                    name="nickname"
                    required
                    autoFocus
                    placeholder="Enter your name or nickname"
                    className="w-full h-12 px-4 rounded-xl bg-black/40 border border-white/10 text-base text-white placeholder-zinc-500 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 text-center font-medium transition-all duration-250"
                  />
                </div>

                <button
                  type="submit"
                  disabled={updateProfileMutation.isPending}
                  className="w-full h-12 bg-white hover:bg-neutral-200 text-black font-bold text-base rounded-xl transition-all duration-300 shadow-[0_4px_20px_-5px_rgba(255,255,255,0.3)] hover:-translate-y-0.5 active:scale-[0.98] disabled:opacity-70 disabled:hover:translate-y-0 disabled:cursor-not-allowed"
                >
                  {updateProfileMutation.isPending ? "Saving Profile..." : "Let's get started"}
                </button>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
