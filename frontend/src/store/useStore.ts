import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { SourceDocument, Conversation, ModelParams, DocType } from '../types';
import { INITIAL_DOCUMENTS } from '../data';

interface AppState {
  theme: "light" | "dark";
  setTheme: (theme: "light" | "dark") => void;
  currentTab: string;
  isMobileSidebarOpen: boolean;
  documents: SourceDocument[];
  conversations: Conversation[];
  activeConvId: string | null;
  params: ModelParams;
  isProcessing: boolean;
  
  setCurrentTab: (tab: string) => void;
  setIsMobileSidebarOpen: (isOpen: boolean) => void;
  setDocuments: (docs: SourceDocument[] | ((prev: SourceDocument[]) => SourceDocument[])) => void;
  setConversations: (convs: Conversation[] | ((prev: Conversation[]) => Conversation[])) => void;
  setActiveConvId: (id: string | null) => void;
  setParams: (params: Partial<ModelParams>) => void;
  setIsProcessing: (isProcessing: boolean) => void;
  toggleDocumentActive: (id: string) => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      theme: "dark",
      setTheme: (theme) => set({ theme }),
      currentTab: 'dashboard',
      isMobileSidebarOpen: false,
      documents: INITIAL_DOCUMENTS,
      conversations: [
        {
          id: "conv-1",
          title: "Modular RAG Analysis",
          timestamp: new Date().toISOString(),
          messages: [
            {
              id: "msg-1",
              sender: "assistant",
              text: "Analysis of the provided document set suggests a 24% increase in operational efficiency when applying the **Modular RAG Framework**. The core findings point toward three primary vectors:\n\n1. **Latency Reduction**: By caching semantic embeddings, we reduce retrieval time by 120ms.\n2. **Contextual Accuracy**: The use of hierarchical re-ranking improves result relevance.",
              timestamp: new Date(Date.now() - 3600000).toISOString(),
              citations: [
                {
                  name: "whitepaper_2024_v2.pdf",
                  fitScore: 98,
                  snippet: "The application of multi-stage validation ensures that generated text adheres to the ground truth provided by source documents. Feeding precise, re-ranked snippets rather than raw paragraphs increases accuracy by 34%."
                },
                {
                  name: "internal_wiki_efficiency",
                  fitScore: 82,
                  snippet: "Deploying the Modular RAG Framework across the principal research directories has successfully triggered a 24% increase in research discovery rates and operational execution. Embedding generation latency: reduced to 18ms."
                }
              ]
            },
            {
              id: "msg-2",
              sender: "user",
              text: "Can you cross-reference the efficiency gains with the Q3 fiscal projections for the Monolith project?",
              timestamp: new Date(Date.now() - 1800000).toISOString()
            }
          ]
        }
      ],
      activeConvId: "conv-1",
      params: {
        temperature: 0.7,
        tokenEfficiency: 89,
        systemInstruction: "",
        selectedModel: "gemini-3.5-flash",
      },
      isProcessing: false,
      
      setCurrentTab: (tab) => set({ currentTab: tab }),
      setIsMobileSidebarOpen: (isOpen) => set({ isMobileSidebarOpen: isOpen }),
      setDocuments: (docsOrUpdater) => set((state) => ({
        documents: typeof docsOrUpdater === 'function' ? docsOrUpdater(state.documents) : docsOrUpdater
      })),
      setConversations: (convsOrUpdater) => set((state) => ({
        conversations: typeof convsOrUpdater === 'function' ? convsOrUpdater(state.conversations) : convsOrUpdater
      })),
      setActiveConvId: (id) => set({ activeConvId: id }),
      setParams: (newParams) => set((state) => ({ params: { ...state.params, ...newParams } })),
      setIsProcessing: (isProcessing) => set({ isProcessing }),
      toggleDocumentActive: (id) => set((state) => ({
        documents: state.documents.map(doc => doc.id === id ? { ...doc, active: !doc.active } : doc)
      })),
    }),
    {
      name: 'documind-storage',
      partialize: (state) => ({ 
        theme: state.theme,
        documents: state.documents, 
        conversations: state.conversations,
        params: state.params
      }), // only persist these fields
    }
  )
);
