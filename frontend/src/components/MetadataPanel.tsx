import React from "react";
import { SourceDocument, ModelParams } from "../types";
import { FileText, FileSpreadsheet, Globe, Sliders, Sparkles, BookOpen } from "lucide-react";
import { cn } from "../lib/utils";

interface MetadataPanelProps {
  documents: SourceDocument[];
  toggleDocumentActive: (id: string) => void;
  params: ModelParams;
  onParamChange: (newParams: Partial<ModelParams>) => void;
}

export default function MetadataPanel({
  documents,
  toggleDocumentActive,
  params,
  onParamChange,
}: MetadataPanelProps) {
  
  const getTempLabel = (temp: number) => {
    if (temp <= 0.2) return "Deterministic";
    if (temp <= 0.5) return "Analytical";
    if (temp <= 0.8) return "Balanced";
    if (temp <= 1.0) return "Conversational";
    return "Imaginative";
  };

  return (
    <aside
      id="metadata-panel"
      className="hidden xl:flex w-80 flex-col bg-surface-container-lowest border-l border-border p-6 overflow-y-auto select-none space-y-8"
    >
      <div className="flex items-center space-x-2 text-foreground mb-2">
        <Sparkles className="w-4 h-4" />
        <h3 className="text-sm font-semibold tracking-tight">
          Metadata
        </h3>
      </div>

      <div className="space-y-8">
        {/* Active Sources Section */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center">
              Active Sources
            </h4>
            <span className="text-[10px] bg-background border border-border text-foreground font-semibold px-2 py-0.5 rounded-md">
              {documents.filter((d) => d.active).length} ON
            </span>
          </div>

          <div className="space-y-2 max-h-[30vh] overflow-y-auto pr-1">
            {documents.map((doc) => {
              const pseudoFit = Math.min(100, Math.max(65, 75 + (doc.name.charCodeAt(doc.name.length - 1) % 25)));
              
              return (
                <div
                  key={doc.id}
                  className={cn(
                    "p-3 rounded-lg border transition-colors",
                    doc.active ? "bg-background border-border shadow-sm" : "bg-surface border-transparent opacity-60 hover:opacity-100"
                  )}
                >
                  <div className="flex items-start justify-between space-x-2">
                    <div className="flex items-center space-x-2 min-w-0">
                      <div className={cn("p-1.5 rounded-md flex-shrink-0", doc.active ? "bg-surface border border-border text-foreground" : "text-muted-foreground")}>
                        {doc.type === "spreadsheet" ? <FileSpreadsheet className="w-3 h-3" /> : doc.type === "link" ? <Globe className="w-3 h-3" /> : <FileText className="w-3 h-3" />}
                      </div>
                      <p className="text-xs font-medium text-foreground truncate cursor-default">
                        {doc.name}
                      </p>
                    </div>

                    <input
                      type="checkbox"
                      checked={doc.active}
                      onChange={() => toggleDocumentActive(doc.id)}
                      className="w-3.5 h-3.5 rounded text-foreground focus:ring-foreground border-border cursor-pointer mt-1 bg-surface accent-foreground"
                    />
                  </div>

                  <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/50">
                    <span className="text-[10px] text-muted-foreground">
                      {doc.addedAt}
                    </span>
                    {doc.active && (
                      <span className="text-[10px] font-medium text-foreground">
                        {pseudoFit}% Match
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Model parameters Section */}
        <section className="space-y-5">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center">
            Parameters
          </h4>

          <div className="space-y-2">
            <label className="text-[11px] text-muted-foreground font-medium block">
              Model
            </label>
            <select
              value={params.selectedModel}
              onChange={(e) => onParamChange({ selectedModel: e.target.value })}
              className="w-full bg-background border border-border rounded-lg px-2.5 py-1.5 text-xs font-medium text-foreground focus:outline-none focus:ring-1 focus:ring-foreground/20 cursor-pointer shadow-sm"
            >
              <option value="gemini-3.5-flash">Gemini 3.5 Flash</option>
              <option value="gemini-3.1-pro-preview">Gemini 3.1 Pro</option>
              <option value="gemini-3.1-flash-lite">Gemini 3.1 Flash Lite</option>
            </select>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-muted-foreground font-medium">Temperature</span>
              <span className="text-foreground font-mono bg-background border border-border px-1.5 py-0.5 rounded-md">
                {params.temperature.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.2"
              step="0.1"
              value={params.temperature}
              onChange={(e) => onParamChange({ temperature: parseFloat(e.target.value) })}
              className="w-full accent-foreground bg-surface border border-border h-1.5 rounded-full appearance-none cursor-pointer outline-none"
            />
            <p className="text-[10px] text-muted-foreground text-right">{getTempLabel(params.temperature)}</p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-muted-foreground font-medium">Token Efficiency</span>
              <span className="text-foreground font-mono bg-background border border-border px-1.5 py-0.5 rounded-md">
                {params.tokenEfficiency}%
              </span>
            </div>
            <input
              type="range"
              min="20"
              max="100"
              step="5"
              value={params.tokenEfficiency}
              onChange={(e) => onParamChange({ tokenEfficiency: parseInt(e.target.value) })}
              className="w-full accent-foreground bg-surface border border-border h-1.5 rounded-full appearance-none cursor-pointer outline-none"
            />
          </div>
        </section>
      </div>
    </aside>
  );
}
