import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { API_BASE } from "../lib/api";
import { toast } from "sonner";
import { SourceDocument } from "../types";

interface UseDocumentUploadProps {
  documents: SourceDocument[];
  sessionDocuments: any[];
}

export function useDocumentUpload({ documents, sessionDocuments }: UseDocumentUploadProps) {
  const queryClient = useQueryClient();
  const [draftLinkedDocIds, setDraftLinkedDocIds] = useState<string[]>([]);

  const uploadDocMutation = useMutation({
    mutationFn: async ({ file, sessionId }: { file: File; sessionId?: string }) => {
      const res = await api.uploadDocument(file, sessionId);
      if (res.job_id) {
        const es = new EventSource(`${API_BASE}/documents/status/${res.job_id}`);
        es.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.status === "completed" || data.status === "failed") {
              es.close();
              queryClient.invalidateQueries({ queryKey: ["documents"] });
              if (sessionId) {
                queryClient.invalidateQueries({ queryKey: ["sessionDocuments", sessionId] });
              }
            }
          } catch (e) {
            console.error("SSE parse error", e);
          }
        };
        es.onerror = () => {
          es.close();
          queryClient.invalidateQueries({ queryKey: ["documents"] });
          if (sessionId) {
            queryClient.invalidateQueries({ queryKey: ["sessionDocuments", sessionId] });
          }
        };
      }
      return res;
    },
    onSuccess: (data, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      if (sessionId) {
        queryClient.invalidateQueries({ queryKey: ["sessionDocuments", sessionId] });
      } else if (data?.document_id) {
        setDraftLinkedDocIds((prev) => Array.from(new Set([...prev, data.document_id])));
      }
    },
    onError: (error: any) => {
      toast.error(`Upload failed: ${error.message || "An unknown error occurred"}`);
    },
  });

  const handleAddDocument = async (file: File, sessionId?: string) => {
    uploadDocMutation.mutate({ file, sessionId });
  };

  const linkDocMutation = useMutation({
    mutationFn: async ({ sessionId, documentId }: { sessionId: string; documentId: string }) => {
      await api.linkDocumentToSession(sessionId, documentId);
    },
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ["sessionDocuments", sessionId] });
    },
  });

  const handleLinkDocument = async (sessionId: string, documentId: string) => {
    if (!sessionId) {
      setDraftLinkedDocIds((prev) => (prev.includes(documentId) ? prev : [...prev, documentId]));
      return;
    }
    const alreadyLinked = sessionDocuments.some(
      (d: any) => d.id === documentId || d.public_id === documentId
    );
    if (alreadyLinked) return;

    const docToLink = documents.find((doc) => doc.id === documentId);
    if (docToLink) {
      queryClient.setQueryData(["sessionDocuments", sessionId], (old: any) => {
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
    mutationFn: async ({ sessionId, documentId }: { sessionId: string; documentId: string }) => {
      await api.unlinkDocumentFromSession(sessionId, documentId);
    },
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ["sessionDocuments", sessionId] });
    },
  });

  const handleUnlinkDocument = async (sessionId: string, documentId: string) => {
    if (!sessionId) {
      setDraftLinkedDocIds((prev) => prev.filter((id) => String(id) !== String(documentId)));
      return;
    }
    unlinkDocMutation.mutate({ sessionId, documentId });
  };

  const deleteDocMutation = useMutation({
    mutationFn: async (id: string) => {
      await api.deleteDocument(id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  const handleDeleteDocument = (id: string) => {
    deleteDocMutation.mutate(id);
  };

  return {
    draftLinkedDocIds,
    setDraftLinkedDocIds,
    uploadDocMutation,
    handleAddDocument,
    handleLinkDocument,
    handleUnlinkDocument,
    handleDeleteDocument,
  };
}
