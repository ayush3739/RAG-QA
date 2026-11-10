import { Citation } from "../types";

export const normalizeSourceScore = (score: any): number | null => {
  if (score == null) return null;
  const numeric = Number(score);
  if (!Number.isFinite(numeric)) return null;
  const normalized = numeric >= 0 && numeric <= 1
    ? numeric
    : 1 / (1 + Math.exp(-numeric));
  return Math.round(Math.max(0, Math.min(1, normalized)) * 100);
};

export const formatFileSizeFromKb = (sizeKb: any): string => {
  const numericKb = Number(sizeKb);
  if (!Number.isFinite(numericKb) || numericKb <= 0) return "0 KB";

  if (numericKb < 1024) {
    return `${Math.round(numericKb)} KB`;
  }

  const sizeMb = numericKb / 1024;
  return `${sizeMb.toFixed(2).replace(/\.?0+$/, "")} MB`;
};

export const mapSourcesToCitations = (sources: any): Citation[] => {
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
    const baseName = rawSource.split('/').pop()?.split('_').slice(1).join('_') || rawSource.split('/').pop() || s.name || 'Source';
    const fileName = s.document_name || baseName;
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

export function getTokenFromSearch(search: string): string | null {
  const params = new URLSearchParams(search);
  return params.get('token');
}

export function formatResetTime(resetAt?: string): { timeRemaining: string; localTime: string; summary: string } {
  let targetDate: Date;

  if (resetAt) {
    targetDate = new Date(resetAt);
  } else {
    // Pure frontend calculation: next midnight UTC (00:00:00 UTC)
    const now = new Date();
    targetDate = new Date(Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate() + 1,
      0, 0, 0, 0
    ));
  }

  const now = new Date();
  const diffMs = targetDate.getTime() - now.getTime();

  const localTime = targetDate.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

  if (diffMs <= 0) {
    return { timeRemaining: "shortly", localTime, summary: `Resets at ${localTime}` };
  }

  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

  const timeRemaining = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  const summary = `Resets at ${localTime} (in ${timeRemaining})`;

  return { timeRemaining, localTime, summary };
}
