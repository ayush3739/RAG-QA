export type DocType = "pdf" | "spreadsheet" | "doc" | "link";

export interface SourceDocument {
  id: string;
  name: string;
  type: DocType;
  content?: string;
  addedAt: string;
  size: string;
  active: boolean;
  status?: "queued" | "indexing" | "indexed" | "failed";
  summary?: string;
  chunkCount?: number;
  embeddingModel?: string;
}

export interface Citation {
  name: string;
  fitScore: number;
  snippet: string;
}

export interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  timestamp: string;
  citations?: Citation[];
  isProcessing?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  timestamp: string;
  documentCount?: number;
  attachedDocIds?: string[];
}

export interface ModelParams {
  temperature: number;
  tokenEfficiency: number;
  systemInstruction: string;
  selectedModel: string;
}
