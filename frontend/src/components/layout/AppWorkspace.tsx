import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import DashboardView from "../DashboardView";
import ChatWorkspace from "../ChatWorkspace";
import { DocumentsView } from "../DocumentsView";
import DocumentDetailsView from "../DocumentDetailsView";
import SessionsView from "../SessionsView";
import SettingsView from "../SettingsView";
import AdminView from "../AdminView";
import { useStore } from "../../store/useStore";
import { SourceDocument, Conversation, Message, ModelParams } from "../../types";

interface AppWorkspaceProps {
  currentTab: string;
  documents: SourceDocument[];
  conversations: Conversation[];
  activeConvId: string | null;
  sessionDocuments: any[];
  draftLinkedDocIds: string[];
  activeConversation?: Conversation;
  activeMessages: Message[];
  isProcessing: boolean;
  graphStatuses: any[];
  params: ModelParams;
  setParams: (p: Partial<ModelParams>) => void;
  setCurrentTab: (tab: string) => void;
  handleNewResearch: () => void;
  handleSelectSession: (id: string) => void;
  handleSendMessage: (text: string, mode?: "auto" | "research") => void;
  handleAddDocument: (file: File, sessionId?: string) => void;
  handleLinkDocument: (sessionId: string, documentId: string) => void;
  handleUnlinkDocument: (sessionId: string, documentId: string) => void;
  handleRetryMessage: (messageId: string, mode?: "auto" | "research") => void;
  toggleDocumentActive: (id: string) => void;
  handleDeleteDocument: (id: string) => void;
  handleDeleteSession: (id: string) => void;
  handleRenameSession: (id: string, newName: string) => void;
}

export function AppWorkspace({
  currentTab,
  documents,
  conversations,
  activeConvId,
  sessionDocuments,
  draftLinkedDocIds,
  activeConversation,
  activeMessages,
  isProcessing,
  graphStatuses,
  params,
  setParams,
  setCurrentTab,
  handleNewResearch,
  handleSelectSession,
  handleSendMessage,
  handleAddDocument,
  handleLinkDocument,
  handleUnlinkDocument,
  handleRetryMessage,
  toggleDocumentActive,
  handleDeleteDocument,
  handleDeleteSession,
  handleRenameSession,
}: AppWorkspaceProps) {
  return (
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
              documents={activeConvId ? sessionDocuments : documents.filter((d) => draftLinkedDocIds.includes(d.id))}
              allDocuments={documents}
              activeConvId={activeConvId || undefined}
              onAddDocument={(file) => handleAddDocument(file, activeConvId || undefined)}
              onLinkDocument={(docId) => handleLinkDocument(activeConvId || "", docId)}
              onUnlinkDocument={(docId) => handleUnlinkDocument(activeConvId || "", docId)}
              onRetryMessage={handleRetryMessage}
              selectedModel={params.selectedModel}
              sessionTitle={activeConvId === "" ? "New Chat" : activeConversation?.title || "New Chat"}
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
        {currentTab === "settings" && <SettingsView params={params} onParamChange={setParams} />}
        {currentTab === "admin" && <AdminView />}
      </motion.div>
    </AnimatePresence>
  );
}
