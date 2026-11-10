import { useState, useRef, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { toast } from "sonner";
import { useStore } from "../store/useStore";
import { Message, Citation, SourceDocument, Conversation } from "../types";
import { mapSourcesToCitations } from "../lib/formatters";

interface UseChatStreamProps {
  activeConvId: string | null;
  setActiveConvId: (id: string | null) => void;
  setConversations: (fn: (prev: Conversation[]) => Conversation[]) => void;
  documents: SourceDocument[];
  draftLinkedDocIds: string[];
  setDraftLinkedDocIds: (ids: string[]) => void;
  setCurrentTab: (tab: string) => void;
  setIsMobileSidebarOpen: (open: boolean) => void;
}

export function useChatStream({
  activeConvId,
  setActiveConvId,
  setConversations,
  documents,
  draftLinkedDocIds,
  setDraftLinkedDocIds,
  setCurrentTab,
  setIsMobileSidebarOpen,
}: UseChatStreamProps) {
  const queryClient = useQueryClient();
  const isProcessing = useStore((s) => s.isProcessing);
  const setIsProcessing = useStore((s) => s.setIsProcessing);
  const isAuthenticated = useStore((s) => s.isAuthenticated);

  const [activeMessages, setActiveMessages] = useState<Message[]>([]);
  const [graphStatuses, setGraphStatuses] = useState<any[]>([]);
  const lastLoadedConvIdRef = useRef<string | null>(null);

  // Sync historical messages for the active conversation
  const { data: historyMessages } = useQuery({
    queryKey: ["sessionHistory", activeConvId],
    queryFn: async () => {
      if (!activeConvId) return [];
      const fetchHistory = api.getSessionHistory || (api as any).getSessionMessages;
      const data = await fetchHistory(activeConvId);
      const rawMessages = data.messages || [];

      return rawMessages.map((m: any, idx: number) => {
        const metadata = m.meta_data || {};
        const sources = m.citations || metadata.sources || [];
        const mappedSources = mapSourcesToCitations(sources);

        return {
          id: m.message_id ? String(m.message_id) : (m.id ? `db-${m.id}` : `msg-${m.created_at}-${idx}`),
          dbId: m.message_id || m.id || undefined,
          sender: m.role === "human" ? "user" : m.role || "assistant",
          text: m.content || "",
          timestamp: m.created_at || new Date().toISOString(),
          citations: mappedSources.length > 0 ? mappedSources : undefined,
          chunks: m.chunks || metadata.chunks || undefined,
        };
      });
    },
    enabled: !!activeConvId && isAuthenticated,
    retry: 1,
    refetchOnWindowFocus: false,
  });

  useEffect(() => {
    if (!activeConvId) {
      setActiveMessages([]);
      lastLoadedConvIdRef.current = null;
      return;
    }

    if (historyMessages && (lastLoadedConvIdRef.current !== activeConvId || activeMessages.length === 0)) {
      lastLoadedConvIdRef.current = activeConvId;
      setActiveMessages(historyMessages);
    }
  }, [activeConvId, historyMessages, activeMessages.length]);

  const handleSendMessage = async (text: string, mode: "auto" | "research" = "auto") => {
    if (!text.trim() || isProcessing) return;

    let targetConvId = activeConvId;

    if (!targetConvId) {
      setIsProcessing(true);
      try {
        const createRes = await api.createSession();
        targetConvId = createRes.session_id;

        const words = text.trim().split(/\s+/);
        const title = words.length <= 6 ? text.trim() : words.slice(0, 6).join(" ") + "...";

        const newConv = {
          id: targetConvId,
          title,
          timestamp: new Date().toISOString(),
          messages: [],
        };
        setConversations((prev) => [newConv, ...prev]);
        setActiveConvId(targetConvId);
        setActiveMessages([]);
        lastLoadedConvIdRef.current = targetConvId;

        await api.renameSession(targetConvId, title);

        const linkedDraftDocIds = [...draftLinkedDocIds];
        for (const docId of linkedDraftDocIds) {
          await api.linkDocumentToSession(targetConvId, docId);
        }
        if (linkedDraftDocIds.length > 0) {
          queryClient.setQueryData(
            ["sessionDocuments", targetConvId],
            documents.filter((doc) => linkedDraftDocIds.includes(doc.id))
          );
          queryClient.invalidateQueries({ queryKey: ["sessionDocuments", targetConvId] });
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

    setConversations((prev) =>
      prev.map((conv) => {
        if (conv.id === finalConvId) {
          return {
            ...conv,
            timestamp: new Date().toISOString(),
          };
        }
        return conv;
      })
    );
    setActiveMessages((prev) => [...prev, userMsg]);
    setGraphStatuses([{ node: "main_router", label: "Deciding route", status: "running" }]);
    setIsProcessing(true);

    const botMsgId = `msg-${Date.now() + 1}`;
    setActiveMessages((prev) => [
      ...prev,
      {
        id: botMsgId,
        sender: "assistant",
        text: "",
        timestamp: new Date().toISOString(),
      },
    ]);

    let currentText = "";
    let finalCitations: Citation[] = [];
    let finalChunks: any = undefined;
    let finalMessageId: number | undefined = undefined;

    try {
      const res = await api.streamChat(finalConvId, text, mode);

      if (!res.ok || !res.body) {
        if (res.status === 429) {
          const errorData = await res.json().catch(() => null);
          const isResearchQuota = errorData?.detail?.error === "RESEARCH_QUOTA_EXCEEDED" || errorData?.detail?.code === "RESEARCH_QUOTA_EXCEEDED" || errorData?.error === "RESEARCH_QUOTA_EXCEEDED";
          const isDailyQuota = errorData?.detail?.error === "QUOTA_EXCEEDED" || errorData?.error === "QUOTA_EXCEEDED";
          const rateLimitMsg = isResearchQuota
            ? `🛑 **Deep Research Quota Exceeded** — ${errorData?.detail?.message || "You have reached your daily research allowance."} Please upgrade to Pro for higher limits.`
            : isDailyQuota
            ? `🛑 **Daily Query Quota Exceeded** — ${errorData?.detail?.message || "You have reached your daily query allowance."} Please upgrade to Pro for 500 queries/day.`
            : "⏳ **Rate Limited** — The AI provider is busy. Please wait a moment and try again.";
          setActiveMessages((prev) => prev.map((m) => (m.id === botMsgId ? { ...m, text: rateLimitMsg } : m)));
          toast.warning(errorData?.detail?.message || "Quota limit reached.", { duration: 7000 });
          api.getQuota().then((q) => useStore.getState().setQuota(q)).catch(() => {});
          return;
        }
        if (res.status === 401) {
          useStore.getState().logout();
          toast.error("Session expired. Please log in again.");
          return;
        }
        throw new Error(`Chat failed with status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let currentEvent = "message";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (let line of lines) {
          if (line.endsWith("\r")) {
            line = line.slice(0, -1);
          }

          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7);
          } else if (line.startsWith("data: ") || line === "data:") {
            const dataStr = line.startsWith("data: ") ? line.slice(6) : "";
            if (dataStr === "[DONE]") continue;

            if (currentEvent === "token") {
              const tokenText = dataStr === "" ? "\n" : dataStr;
              currentText += tokenText;
              setActiveMessages((prev) => prev.map((m) => (m.id === botMsgId ? { ...m, text: currentText } : m)));
            } else if (currentEvent === "answer") {
              try {
                currentText = JSON.parse(dataStr);
              } catch {
                currentText = dataStr;
              }
              setActiveMessages((prev) => prev.map((m) => (m.id === botMsgId ? { ...m, text: currentText } : m)));
            } else if (currentEvent === "graph_status") {
              try {
                const status = JSON.parse(dataStr);
                setGraphStatuses((prev) => {
                  const existingIdx = prev.findIndex((item: any) => item.node === status.node);
                  if (existingIdx === -1) return [...prev, status];
                  return prev.map((item: any, idx) => (idx === existingIdx ? { ...item, ...status } : item));
                });
              } catch (e) {}
            } else if (currentEvent === "metadata") {
              try {
                const data = JSON.parse(dataStr);
                const mappedSources = mapSourcesToCitations(data.sources);
                finalCitations = mappedSources;
                finalChunks = data.chunks || undefined;
                finalMessageId = data.message_id || undefined;
                setActiveMessages((prev) =>
                  prev.map((m) =>
                    m.id === botMsgId
                      ? {
                          ...m,
                          dbId: data.message_id || undefined,
                          citations: mappedSources,
                          chunks: data.chunks || undefined,
                        }
                      : m
                  )
                );
              } catch (e) {}
            } else if (currentEvent === "rate_limit") {
              const rateLimitMsg = "⏳ **Rate Limited** — The AI provider is busy. Please wait a moment and try again.";
              currentText = rateLimitMsg;
              setActiveMessages((prev) => prev.map((m) => (m.id === botMsgId ? { ...m, text: rateLimitMsg } : m)));
              toast.warning("Rate limited by AI provider. Try again in a few seconds.", { duration: 6000 });
            } else if (currentEvent === "error") {
              const errMsg = dataStr ? `⚠️ **Error** — ${dataStr}` : "⚠️ **Error** — Something went wrong. Please try again.";
              currentText = errMsg;
              setActiveMessages((prev) => prev.map((m) => (m.id === botMsgId ? { ...m, text: errMsg } : m)));
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
      setActiveMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);

      const finalAssistantMsg: Message = {
        id: botMsgId,
        dbId: finalMessageId,
        sender: "assistant",
        text: currentText || "⚠️ **Workspace Query Refusal**: Connection to backend failed.",
        timestamp: new Date().toISOString(),
        citations: finalCitations,
        chunks: finalChunks,
      };

      queryClient.setQueryData(["sessionHistory", finalConvId], (old: any) => {
        const filtered = (old || []).filter(
          (m: any) => m.id !== botMsgId && m.id !== userMsg.id && !m.id.startsWith("msg-error-")
        );
        return [...filtered, userMsg, finalAssistantMsg];
      });

      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      queryClient.invalidateQueries({ queryKey: ["user-activity"] });
      api.getQuota().then((q) => useStore.getState().setQuota(q)).catch(() => {});
      setGraphStatuses([]);
    }
  };

  const createSessionMutation = useMutation({
    mutationFn: async ({ title, docIds }: { title: string; docIds: string[] }) => {
      const createRes = await api.createSession();
      const sessionId = createRes.session_id;

      for (const docId of docIds) {
        await api.linkDocumentToSession(sessionId, docId);
      }
      return sessionId;
    },
    onSuccess: (newSessionId) => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setActiveConvId(newSessionId);
    },
  });

  const handleTakeToChat = (topic: string, selectedDocIds: string[]) => {
    setCurrentTab("conversations");
    createSessionMutation.mutate(
      { title: topic, docIds: selectedDocIds },
      {
        onSuccess: (newSessionId) => {
          setTimeout(() => {
            useStore.getState().setActiveConvId(newSessionId);
            handleSendMessage(topic);
          }, 100);
        },
      }
    );
  };

  const handleRetryMessage = async (messageId: string) => {
    if (isProcessing) return;
    const msgIdx = activeMessages.findIndex((m) => m.id === messageId);
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
    handleSendMessage(userQuery);
  };

  const deleteSessionMutation = useMutation({
    mutationFn: async (id: string) => {
      await api.deleteSession(id);
    },
  });

  const handleDeleteSession = (id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    deleteSessionMutation.mutate(id);
  };

  const renameSessionMutation = useMutation({
    mutationFn: async ({ id, newName }: { id: string; newName: string }) => {
      await api.renameSession(id, newName);
    },
  });

  const handleRenameSession = (id: string, newName: string) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, title: newName } : c)));
    renameSessionMutation.mutate({ id, newName });
  };

  const handleNewResearch = () => {
    setCurrentTab("conversations");
    setIsMobileSidebarOpen(false);
    setActiveConvId("");
    setDraftLinkedDocIds([]);
  };

  return {
    activeMessages,
    setActiveMessages,
    graphStatuses,
    handleSendMessage,
    handleRetryMessage,
    handleTakeToChat,
    handleNewResearch,
    handleDeleteSession,
    handleRenameSession,
    createSessionMutation,
  };
}
