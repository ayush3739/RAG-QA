import React, { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import AuthView from "./components/AuthView";
import VerifyEmailView from "./components/VerifyEmailView";
import ResetPasswordView from "./components/ResetPasswordView";
import NotFoundView from "./components/NotFoundView";
import RateLimitView from "./components/RateLimitView";
import ServerErrorView from "./components/ServerErrorView";
import ProWelcomeModal, { isProWelcomed, markProWelcomed } from "./components/ProWelcomeModal";
import { AppShell } from "./components/layout/AppShell";
import { AppWorkspace } from "./components/layout/AppWorkspace";
import { useDocumentUpload } from "./hooks/useDocumentUpload";
import { useChatStream } from "./hooks/useChatStream";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { formatFileSizeFromKb, getTokenFromSearch } from "./lib/formatters";
import { api } from "./lib/api";
import { toast } from "sonner";
import { Sparkles } from "lucide-react";
import { useStore } from "./store/useStore";

const LandingPage = React.lazy(() => import("./landing/page"));
const ArchitecturePage = React.lazy(() => import("./landing/ArchitecturePage"));

export default function App() {
  const queryClient = useQueryClient();
  const [hash, setHash] = useState(window.location.hash);
  const [pathname, setPathname] = useState(window.location.pathname);
  const [search, setSearch] = useState(window.location.search);

  useEffect(() => {
    const handleLocationChange = () => {
      let currentHash = window.location.hash;
      let currentPath = window.location.pathname;

      // Cleanly normalize any legacy '#/route' to pure path '/route'
      if (currentHash.startsWith('#/')) {
        const cleanPath = currentHash.replace(/^#/, '');
        window.history.replaceState(null, '', cleanPath);
        currentPath = cleanPath;
        currentHash = '';
      } else if (currentHash === '#' || currentHash === '#/') {
        window.history.replaceState(null, '', currentPath || '/');
        currentHash = '';
      }

      setHash(currentHash);
      setPathname(currentPath);
      setSearch(window.location.search);

      if (currentPath === '/signup' || currentPath === '/register') {
        useStore.getState().setAuthMode('register');
      } else if (currentPath === '/login' || currentPath === '/auth') {
        useStore.getState().setAuthMode('login');
      }
    };

    handleLocationChange();
    window.addEventListener("hashchange", handleLocationChange);
    window.addEventListener("popstate", handleLocationChange);
    return () => {
      window.removeEventListener("hashchange", handleLocationChange);
      window.removeEventListener("popstate", handleLocationChange);
    };
  }, []);

  const currentTab = useStore((s) => s.currentTab);
  const setCurrentTab = useStore((s) => s.setCurrentTab);
  const isMobileSidebarOpen = useStore((s) => s.isMobileSidebarOpen);
  const setIsMobileSidebarOpen = useStore((s) => s.setIsMobileSidebarOpen);
  const theme = useStore((s) => s.theme);
  const isAuthenticated = useStore((s) => s.isAuthenticated);
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
      window.history.replaceState({}, document.title, "/dashboard");
      useStore.getState().setCurrentTab("dashboard");
    } else if (oauthError) {
      const displayMsg = errorMessage || "OAuth authentication failed. Please try again.";
      toast.error(`Login Error: ${displayMsg}`);
      window.history.replaceState({}, document.title, "/login");
      useStore.getState().setCurrentTab("auth");
    }
  }, []);

  useEffect(() => {
    const effectiveTheme = theme || "dark";
    if (effectiveTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  const documents = useStore((s) => s.documents);
  const setDocuments = useStore((s) => s.setDocuments);
  const toggleDocumentActive = useStore((s) => s.toggleDocumentActive);

  const conversations = useStore((s) => s.conversations);
  const setConversations = useStore((s) => s.setConversations);

  const activeConvId = useStore((s) => s.activeConvId);
  const setActiveConvId = useStore((s) => s.setActiveConvId);

  const params = useStore((s) => s.params);
  const setParams = useStore((s) => s.setParams);

  const isProcessing = useStore((s) => s.isProcessing);
  const user = useStore((s) => s.user);
  const setUser = useStore((s) => s.setUser);
  const setIsProWelcomeOpen = useStore((s) => s.setIsProWelcomeOpen);

  // First-time Pro celebratory modal trigger
  useEffect(() => {
    if (isAuthenticated && user?.role?.toLowerCase() === "pro") {
      if (!isProWelcomed(user) && !showOnboarding) {
        setIsProWelcomeOpen(true);
        markProWelcomed(user);
      }
    }
  }, [isAuthenticated, user?.role, user?.id, user?.email, showOnboarding, setIsProWelcomeOpen]);

  const updateProfileMutation = useMutation({
    mutationFn: async (body: { name?: string; email?: string }) => {
      const data = await api.updateMe(body);
      setUser(data);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['me'] });
    },
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
        type: d.mime_type?.includes("pdf")
          ? "pdf"
          : d.mime_type?.includes("spreadsheet") || d.mime_type?.includes("csv")
          ? "spreadsheet"
          : "doc",
        size: formatFileSizeFromKb(d.file_size_kb),
        sizeKb: d.file_size_kb,
        addedAt: "Just now",
        summary: d.status === "indexed" ? "Indexed successfully" : d.status === "failed" ? "Indexing failed" : "Processing...",
        active: true,
        chunkCount: d.chunk_count || 0,
        status: d.status,
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
        type: d.mime_type?.includes("pdf")
          ? "pdf"
          : d.mime_type?.includes("spreadsheet") || d.mime_type?.includes("csv")
          ? "spreadsheet"
          : "doc",
        size: formatFileSizeFromKb(d.file_size_kb),
        sizeKb: d.file_size_kb,
        addedAt: "Just now",
        summary: d.status === "indexed" ? "Indexed successfully" : d.status === "failed" ? "Indexing failed" : "Processing...",
        active: true,
        chunkCount: d.chunk_count || 0,
        status: d.status,
        public_id: d.public_id,
      }));
    },
    enabled: isAuthenticated && !!activeConvId && currentTab === "conversations",
    retry: 1,
    refetchOnWindowFocus: false,
  });
  const sessionDocuments = sessionDocsData || [];

  // Modular Document Upload & Link hook
  const {
    draftLinkedDocIds,
    setDraftLinkedDocIds,
    handleAddDocument,
    handleLinkDocument,
    handleUnlinkDocument,
    handleDeleteDocument,
  } = useDocumentUpload({ documents, sessionDocuments });

  // Modular Chat SSE streaming & Session hook
  const {
    activeMessages,
    graphStatuses,
    handleSendMessage,
    handleRetryMessage,
    handleNewResearch,
    handleDeleteSession,
    handleRenameSession,
  } = useChatStream({
    activeConvId,
    setActiveConvId,
    setConversations,
    documents,
    draftLinkedDocIds,
    setDraftLinkedDocIds,
    setCurrentTab,
    setIsMobileSidebarOpen,
  });

  // Sync pure pathname to application state
  useEffect(() => {
    const handleRouteSync = () => {
      const path = window.location.pathname;

      if (path === "/rate-limit") {
        setCurrentTab("rate_limit");
        return;
      }
      if (path === "/500") {
        setCurrentTab("server_error");
        return;
      }
      if (path === "/404") {
        setCurrentTab("not_found");
        return;
      }

      if (!isAuthenticated) return;

      const parts = path.replace(/^\//, "").split("/");
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
      } else if (route === "admin") {
        const currentUser = useStore.getState().user;
        if (currentUser?.role?.toLowerCase() === "admin") {
          setCurrentTab("admin");
        } else {
          setCurrentTab("dashboard");
        }
      } else if (route === "chat") {
        setCurrentTab("conversations");
        if (param === "new") {
          setActiveConvId("");
        } else if (param) {
          setActiveConvId(param);
        } else {
          setActiveConvId("");
        }
      } else if (route === "" || route === "architecture" || route === "login" || route === "signup") {
        // Root / public routes
      } else {
        setCurrentTab("not_found");
      }
    };

    window.addEventListener("popstate", handleRouteSync);
    window.addEventListener("hashchange", handleRouteSync);
    handleRouteSync();
    return () => {
      window.removeEventListener("popstate", handleRouteSync);
      window.removeEventListener("hashchange", handleRouteSync);
    };
  }, [setCurrentTab, setActiveConvId, isAuthenticated]);

  // Sync state to pure URL pathname
  useEffect(() => {
    if (!isAuthenticated) return;

    let targetPath = "";
    if (currentTab === "dashboard") {
      targetPath = "/dashboard";
    } else if (currentTab === "documents") {
      targetPath = "/documents";
    } else if (currentTab === "document_details") {
      targetPath = "/documents/details";
    } else if (currentTab === "chats") {
      targetPath = "/chats";
    } else if (currentTab === "settings") {
      targetPath = "/settings";
    } else if (currentTab === "admin") {
      targetPath = "/admin";
    } else if (currentTab === "conversations") {
      if (activeConvId === "") {
        targetPath = "/chat/new";
      } else if (activeConvId) {
        targetPath = `/chat/${activeConvId}`;
      } else {
        targetPath = "/chat/new";
      }
    }

    if (targetPath && window.location.pathname !== targetPath) {
      window.history.pushState(null, "", targetPath);
      setPathname(targetPath);
    }
  }, [currentTab, activeConvId, isAuthenticated]);

  useQuery({
    queryKey: ['sessions'],
    queryFn: async () => {
      const sessData = await api.getSessions();
      const sessions = sessData.sessions || [];

      const mappedSessions = sessions.map((s: any) => ({
        id: s.session_id,
        title: s.title,
        timestamp: s.updated_at || new Date().toISOString(),
        messages: [],
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

  useKeyboardShortcuts({
    'u': () => setCurrentTab("documents"),
    'd': () => setCurrentTab("dashboard"),
    't': () => setCurrentTab("chats"),
    's': () => setCurrentTab("settings"),
    'cmd+k': () => handleNewResearch(),
  });

  const handleSelectSession = (id: string) => {
    setActiveConvId(id);
    setCurrentTab("conversations");
  };

  const activeConversation =
    activeConvId === ""
      ? undefined
      : conversations.find((c) => c.id === activeConvId) || conversations[0];

  // Dedicated auth routes: check pathname first, then hash
  const hashPath = hash.replace(/^#\/?/, '').split('?')[0];
  const hashSearch = hash.includes('?') ? '?' + hash.split('?')[1] : '';

  const resolvedPath = pathname !== '/' ? pathname : '/' + hashPath;
  const resolvedSearch = search || hashSearch;

  if (resolvedPath === '/verify-email') {
    const token = getTokenFromSearch(resolvedSearch);
    return (
      <VerifyEmailView
        token={token || ''}
        onNavigateToLogin={() => window.location.replace('/login')}
      />
    );
  }

  if (resolvedPath === '/reset-password') {
    const token = getTokenFromSearch(resolvedSearch);
    return (
      <ResetPasswordView
        token={token || ''}
        onNavigateToLogin={() => window.location.replace('/login')}
      />
    );
  }

  const isArchitecture = pathname === "/architecture" || hash === "#/architecture";
  const isRateLimit = pathname === "/rate-limit" || hash === "#/rate-limit" || currentTab === "rate_limit";
  const isServerError = pathname === "/500" || hash === "#/500" || currentTab === "server_error";
  const isNotFound = pathname === "/404" || hash === "#/404" || currentTab === "not_found";
  const isLanding = pathname === "/" && !isArchitecture && !isRateLimit && !isServerError && !isNotFound;

  if (isRateLimit) {
    return <RateLimitView />;
  }

  if (isServerError) {
    return <ServerErrorView />;
  }

  if (isNotFound) {
    return <NotFoundView onNavigate={setCurrentTab} />;
  }

  if (isArchitecture) {
    return (
      <React.Suspense fallback={<div className="min-h-screen bg-[#020617] flex items-center justify-center" />}>
        <ArchitecturePage />
      </React.Suspense>
    );
  }

  if (isLanding) {
    return (
      <React.Suspense fallback={<div className="min-h-screen bg-[#020617] flex items-center justify-center" />}>
        <LandingPage />
      </React.Suspense>
    );
  }

  if (!isAuthenticated) {
    return <AuthView />;
  }

  return (
    <AppShell
      currentTab={currentTab}
      setCurrentTab={setCurrentTab}
      isMobileSidebarOpen={isMobileSidebarOpen}
      setIsMobileSidebarOpen={setIsMobileSidebarOpen}
      onNewResearch={handleNewResearch}
      conversationsCount={conversations.length}
    >
      <AppWorkspace
        currentTab={currentTab}
        documents={documents}
        conversations={conversations}
        activeConvId={activeConvId}
        sessionDocuments={sessionDocuments}
        draftLinkedDocIds={draftLinkedDocIds}
        activeConversation={activeConversation}
        activeMessages={activeMessages}
        isProcessing={isProcessing}
        graphStatuses={graphStatuses}
        params={params}
        setParams={setParams}
        setCurrentTab={setCurrentTab}
        handleNewResearch={handleNewResearch}
        handleSelectSession={handleSelectSession}
        handleSendMessage={handleSendMessage}
        handleAddDocument={handleAddDocument}
        handleLinkDocument={handleLinkDocument}
        handleUnlinkDocument={handleUnlinkDocument}
        handleRetryMessage={handleRetryMessage}
        toggleDocumentActive={toggleDocumentActive}
        handleDeleteDocument={handleDeleteDocument}
        handleDeleteSession={handleDeleteSession}
        handleRenameSession={handleRenameSession}
      />

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
                <h2 className="text-3xl font-extrabold tracking-tight text-white">Welcome to KnowVex</h2>
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
                    updateProfileMutation.mutate(
                      { name: newName.trim() },
                      {
                        onSuccess: () => setShowOnboarding(false),
                      }
                    );
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

      {/* Celebratory Pro Welcome Modal */}
      <ProWelcomeModal />
    </AppShell>
  );
}
