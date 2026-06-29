import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { SourceDocument, Conversation, ModelParams, DocType } from '../types';
import { INITIAL_DOCUMENTS } from '../data';

interface UserData {
  name: string;
  email: string;
}

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
  activeDocumentId: string | null;

  setCurrentTab: (tab: string) => void;
  setIsMobileSidebarOpen: (isOpen: boolean) => void;
  setActiveDocumentId: (id: string | null) => void;
  setDocuments: (docs: SourceDocument[] | ((prev: SourceDocument[]) => SourceDocument[])) => void;
  setConversations: (convs: Conversation[] | ((prev: Conversation[]) => Conversation[])) => void;
  setActiveConvId: (id: string | null) => void;
  setParams: (params: Partial<ModelParams>) => void;
  setIsProcessing: (isProcessing: boolean) => void;
  toggleDocumentActive: (id: string) => void;

  isAuthenticated: boolean;
  setIsAuthenticated: (val: boolean) => void;
  authMode: 'login' | 'register';
  setAuthMode: (mode: 'login' | 'register') => void;
  accessToken: string | null;
  setAccessToken: (token: string | null) => void;
  user: UserData | null;
  setUser: (user: UserData | null) => void;
  logout: () => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      theme: "dark",
      setTheme: (theme) => set({ theme }),
      isAuthenticated: false,
      setIsAuthenticated: (val) => set({ isAuthenticated: val }),
      authMode: 'login',
      setAuthMode: (mode) => set({ authMode: mode }),
      accessToken: null,
      setAccessToken: (token) => set({ accessToken: token, isAuthenticated: !!token }),
      user: null,
      setUser: (user) => set({ user }),
      logout: () => set({ accessToken: null, isAuthenticated: false, activeConvId: null, user: null }),

      currentTab: 'dashboard',
      isMobileSidebarOpen: false,
      activeDocumentId: null,
      documents: INITIAL_DOCUMENTS,
      conversations: [],
      activeConvId: null,
      params: {
        temperature: 0.7,
        tokenEfficiency: 89,
        systemInstruction: "",
        selectedModel: "gemini-3.5-flash",
      },
      isProcessing: false,

      setCurrentTab: (tab) => set({ currentTab: tab }),
      setIsMobileSidebarOpen: (isOpen) => set({ isMobileSidebarOpen: isOpen }),
      setActiveDocumentId: (id) => set({ activeDocumentId: id }),
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
        accessToken: state.accessToken,
        params: state.params
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
           state.isAuthenticated = !!state.accessToken;
        }
      }
    }
  )
);
