# DocuMind — Frontend Specification v2.0

> **React AI Workspace**
> This document is the single source of truth for the DocuMind React frontend. Every page includes purpose, layout, component list, all states, and backend integration. The backend is fully built — build from this, not from assumptions.

---

## Table of Contents

1. [Product Vision](#1-product-vision)
2. [Design Language](#2-design-language)
3. [Application Layout](#3-application-layout)
4. [Pages](#4-pages)
   - 4.1 [Dashboard](#41-dashboard----dashboard)
   - 4.2 [Documents](#42-documents----documents)
   - 4.3 [Chat](#43-chat----chatsessionid)
   - 4.4 [Research Workspace](#44-research-workspace----research)
   - 4.5 [Sessions](#45-sessions----sessions)
   - 4.6 [Settings](#46-settings----settings)
5. [Component Library](#5-component-library)
6. [User Flows](#6-user-flows)
7. [Component Hierarchy](#7-component-hierarchy)
8. [Folder Structure](#8-folder-structure)
9. [State Management](#9-state-management)
10. [Backend Integration by Page](#10-backend-integration-by-page)
11. [Mobile Behavior](#11-mobile-behavior)
12. [Future Features](#12-future-features)

---

## 1. Product Vision

### 1.1 What DocuMind Is

DocuMind is an **AI Workspace for knowledge work** — not a PDF chatbot. Users upload documents, and the system reasons over them using a tool-routing agent that decides — per query — whether to retrieve from documents, search the web, or answer directly. Every response comes with a confidence score, source citations, and a tool trace showing exactly how the answer was produced.

The frontend must make this reasoning **transparent and interactive**. The AI's decision-making process is a first-class citizen of the UI.

### 1.2 Primary Workflows

| Workflow | User Goal | Key Pages |
|---|---|---|
| Upload → Index | Bring a document into the workspace | Documents |
| Chat | Ask questions, get cited + confidence-scored answers | Chat + Inspector |
| Research | Run a structured one-shot research query | Research Workspace |
| Manage Knowledge | Organize documents and sessions | Documents, Sessions |
| Inspect & Verify | Understand *how* an answer was produced | Inspector Panel |

### 1.3 Design Philosophy

- **Transparency over magic** — always show confidence, sources, and tool trace
- **Workspace, not chat** — persistent panels, inspector, multi-document context
- **Progressive disclosure** — simple by default, deep detail on demand
- **Every answer is a structured artifact** — not just text in a bubble
- **Speed through clarity** — skeleton loaders, optimistic updates, no blank screens

### 1.4 Core User Flows at a Glance

| Flow | Steps |
|---|---|
| Upload Flow | Drop file → Queued → Indexing (progress %) → Auto-summary → Ready |
| Chat Flow | Send message → SSE tokens stream → Metadata arrives → Inspector populates → Follow-ups appear |
| Research Flow | Enter topic → Select docs → `POST /research` → Structured JSON → Render report |
| Session Flow | Create session → Attach documents → Chat → Rename → Export |

---

## 2. Design Language

### 2.1 Visual Identity

- **Premium SaaS aesthetic** — sleek, highly professional, minimal "AI slop"
- Zinc-based foundation for a sophisticated, high-contrast monochrome look
- Surface hierarchy: `background` → `surface-container-lowest` → `surface`
- Glass-style panels: `backdrop-filter: blur`, `1px border-border`, subtle premium shadows
- Zero loud gradients in UI chrome — gradients reserved only for the subtlest brand accents
- Minimalist, icon-forward navigation (inspired by Linear and Cursor)
- Bento-box grid layout for dashboards (clean, structured, flat)

### 2.2 Theme Tokens (Tailwind v4)

```css
  /* Primary mapping */
  --color-primary: #18181B; /* Zinc-900 for stark high-contrast primary actions */
  --color-primary-container: #27272A; /* Zinc-800 */
  --color-on-primary-container: #FAFAFA;
  
  /* Surfaces (Linear-style extremely subtle grays) */
  --color-background: #FAFAFA;
  --color-foreground: #09090B;
  
  --color-surface: #F4F4F5;           /* zinc-100 */
  --color-surface-container-lowest: #FFFFFF; /* white */
  --color-surface-container-low: #FAFAFA;    /* zinc-50 */
  --color-surface-container: #F4F4F5;
  --color-surface-container-high: #E4E4E7;   /* zinc-200 */
  
  /* Text */
  --color-on-surface: #09090B;        /* zinc-950 */
  --color-secondary: #3F3F46;         /* zinc-700 */
  --color-on-secondary-container: #52525B; /* zinc-600 */
  --color-muted: #71717A;             /* zinc-500 */
  --color-muted-foreground: #A1A1AA;  /* zinc-400 */
  
  /* Borders & Shadows */
  --color-border: #E4E4E7;            /* zinc-200 */
  --color-ring: #18181B;              /* zinc-900 */
  --shadow-premium: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
```

### 2.3 Typography

| Role | Font | Size | Weight | Color |
|---|---|---|---|---|
| Page Title | Inter | 24px | 700, 800 | `--foreground` |
| Section Heading | Inter | 18px | 600, 700 | `--foreground` |
| Card Title | Inter | 14px | 600 | `--foreground` |
| Body Text | Inter | 14px | 400, 500 | `--muted-foreground` |
| Caption / Meta | Inter | 11px-12px | 500, 600 | `--muted-foreground` |
| Code / Mono | JetBrains Mono | 13px | 400 | `--foreground` |
| Badge / Chip | Inter | 10px-11px | 600, 700 | varies |

### 2.4 Spacing Scale

Use an **8px base grid** exclusively: `4, 8, 12, 16, 24, 32, 48, 64`. No arbitrary spacing values.

### 2.5 Motion & Animation

| Animation | Element | Spec |
|---|---|---|
| Fade in | Page transitions, modals | `opacity 0→1, 150ms ease-out` |
| Slide in right | Inspector panel, drawers | `translateX(100%)→0, 200ms ease-out` |
| Slide in left | Sidebar on mobile | `translateX(-100%)→0, 200ms ease-out` |
| Scale up | Dropdowns, tooltips | `scale(0.95)→1, 120ms ease-out` |
| Skeleton pulse | All loading states | `opacity 0.4→1, 1.2s ease-in-out infinite` |
| Streaming cursor | Chat typing indicator | `opacity 0→1, 0.6s step-start infinite` |
| Progress fill | Upload / indexing bars | `width 0→100%, linear, driven by poll data` |
| Chip stagger | Follow-up question chips | `translateY(8px)→0 + opacity, 100ms delay each` |
| Confidence fill | ConfidenceMeter bar | `width 0→value%, 600ms ease-out on mount` |

### 2.6 Iconography

- Library: **`lucide-react` exclusively** — no mixing of icon sets
- Sizes: `16px` inline, `20px` standalone, `24px` in empty states
- Color: inherits from parent. Never hardcode icon colors except for status icons
- Status icons are **always paired with a text label** — never color-only signaling

### 2.7 Dark Mode

- Full dark mode support from day one using Tailwind's `dark:` prefix
- Dark surface hierarchy: `zinc-950 → zinc-900 → zinc-800 → zinc-700`
- Brand color stays `#1A56DB`; text shifts to `zinc-100/200/400`
- Store preference in `uiStore`, persist to `localStorage`
- Toggle in the top bar as a sun/moon icon button

---

## 3. Application Layout

### 3.1 Shell Structure

All authenticated pages share a persistent three-zone shell. The shell **never re-mounts** on route changes.

```
┌───────────────┐  ┌──────────────────────────┐  ┌────────────────────┐
│  SIDEBAR      │  │  MAIN WORKSPACE           │  │  INSPECTOR PANEL   │
│  240px fixed  │  │  flex-fill                │  │  320px (Chat only) │
│               │  │                           │  │                    │
│  Logo         │  │  Top Bar (48px)           │  │  Confidence        │
│  Nav Links    │  │  ─────────────────────    │  │  Sources           │
│  ─────────    │  │  Page Content             │  │  Tool Trace        │
│  User Info    │  │  (scrollable)             │  │  Follow-ups        │
└───────────────┘  └──────────────────────────┘  └────────────────────┘
```

### 3.2 Sidebar

- Fixed left, **240px** wide. Collapses to **64px** icon-only mode.
- Top: DocuMind logo + wordmark. Collapsed: icon only.
- Nav links with icon + label. Active: filled pill in brand color.
- Bottom: user avatar, display name, dark mode toggle, logout.
- Collapse state persists in `uiStore` (localStorage).
- On collapse: labels fade out (150ms). Icons remain with tooltip on hover.

### 3.3 Top Bar

- Height: **48px**. Sits above page content in the main workspace zone.
- Left: dynamic page title driven by current route.
- Right: LLM mode badge (Local / Cloud), dark mode toggle, user avatar menu.
- The **LLM mode badge is always visible** — it is a global trust signal, not buried in settings.

### 3.4 Inspector Panel — Core UI Element

> ⚠️ **Critical Design Decision:** The Inspector Panel is not an optional sidebar. It is a first-class UI zone, always present in the Chat page and visible by default after the first response. It makes DocuMind feel like a professional AI workspace — comparable to how Cursor shows context or Perplexity shows sources.

- Width: **320px**, fixed right, within the main workspace.
- **Visible on Chat page only.** Hidden on all other pages.
- **Auto-opens** when the first `metadata` SSE event arrives.
- Can be pinned open or collapsed. Collapse state persists in `uiStore` per session.
- Content updates live as each new assistant message is selected or received.
- Sections (in order): Confidence Score → Sources → Tool Trace → Retrieved Chunks (debug, hidden by default) → Follow-up Questions.

### 3.5 Navigation Map

| Route | Page | Auth Required | Inspector Visible |
|---|---|---|---|
| `/login` | Login | No | No |
| `/register` | Register | No | No |
| `/dashboard` | Dashboard | Yes | No |
| `/documents` | Documents | Yes | No |
| `/chat/:sessionId` | Chat | Yes | Yes (after first response) |
| `/research` | Research Workspace | Yes | No |
| `/sessions` | Sessions | Yes | No |
| `/settings` | Settings | Yes | No |

---

## 4. Pages

> Every page includes: **Purpose, Layout, Components, Actions, Backend APIs, and all three states (Loading / Empty / Error).** Build all states — they are not optional.

---

### 4.1 Dashboard — `/dashboard`

#### Purpose
The entry point after login. Provides orientation: what documents exist, what sessions are active, and quick access to the three main workflows. Not a metrics dashboard — a jumping-off point.

#### Layout
- Full-width main workspace. No Inspector. No sub-sidebar.
- **Row 1:** Three Quick Action Cards — Upload, New Chat, Run Research.
- **Row 2:** Two-column — Recent Sessions (left 60%) + Recent Documents (right 40%).
- **Bottom:** Health strip (backend online/offline indicator).

#### Components
- `QuickActionCard` — icon, title, description, CTA button. Three variants: Upload, Chat, Research.
- `SessionCard` (compact) — session name, document count badge, last message preview, timestamp, Open + Delete actions.
- `DocumentCard` (compact) — file icon, name, status badge, page count, Chat action.
- `HealthStrip` — calls `GET /api/v1/health` every 30s. Green dot = online, amber = unreachable.

#### Actions
- **Upload Document** → opens `DocumentUploadModal`
- **New Chat** → `POST /api/v1/sessions` → navigate to `/chat/:id`
- **Run Research** → navigate to `/research`
- **Open Session** → navigate to `/chat/:sessionId`
- **Delete Session** → `DELETE /api/v1/sessions/{id}` with confirmation

#### Backend APIs

| Method | Endpoint | Trigger | Response |
|---|---|---|---|
| `GET` | `/api/v1/sessions` | Page mount | `Session[]` (use first 5) |
| `GET` | `/api/v1/documents` | Page mount | `Document[]` (use first 5) |
| `POST` | `/api/v1/sessions` | New Chat button | `{ id, name, created_at }` |
| `DELETE` | `/api/v1/sessions/{id}` | Delete icon | `204` |
| `GET` | `/api/v1/health` | Every 30s | `{ status, components }` |

#### States

| State | UI Behavior |
|---|---|
| **Loading** | Skeleton cards for session and document rows. Quick Action Cards render immediately (no data needed). |
| **Empty (new user)** | No sessions, no documents. Quick Action Cards shown prominently. Below: empty-state illustration with "Upload your first document to get started" CTA. |
| **Error** | Toast: "Could not load workspace data." Retry button. Quick Action Cards still visible and functional. |
| **Health offline** | Amber strip at bottom: "Backend unreachable — some features unavailable." Auto-retries every 10s. |

---

### 4.2 Documents — `/documents`

#### Purpose
The knowledge management hub. Users upload documents, monitor indexing progress, review auto-summaries, and manage their document library.

#### Layout
- **Left panel (360px):** Document list with search, sort, and filter.
- **Right panel (flex fill):** Document Detail when a doc is selected; Upload Zone when nothing is selected.
- The Upload Zone is also a drag target across the **entire page**.

#### Document Card (in list)

| Element | Content |
|---|---|
| Icon | PDF / DOCX / TXT / Web based on file type |
| Name | Filename, truncated with tooltip on overflow |
| Status Badge | Queued / Indexing N% / Indexed / Failed |
| Metadata | Page count, upload date |
| Actions | Chat, Research, Delete |

#### Document Detail Panel (right)
- Header: document name (editable inline), file type, page count, upload date.
- Action row: **Start Chat**, **Run Research**, **Delete**.
- **Auto-Summary Card:** 5 bullet key topics + estimated reading time. Shows skeleton while generating.
- **Indexing Metadata:** chunk count, embedding model, BM25 status (Hybrid / Vector-only).
- **Linked Sessions:** list of sessions this document is attached to, each clickable.

#### Upload Flow

1. User drops file or pastes URL into Upload Zone.
2. Client-side validates: accepted types (`.pdf` `.docx` `.txt` `.md`), max 50MB.
3. `POST /api/v1/documents/upload` → response: `{ job_id, document_id, status: 'queued' }`.
4. Document appears at top of list with **Queued** badge immediately.
5. Client polls `GET /api/v1/documents/status/{job_id}` every **2 seconds**.
6. Status transitions: `queued → indexing (N%) → indexed`.
7. On `indexed`: success toast, green badge, detail panel loads auto-summary.
8. On `failed`: red badge, inline error, **Retry** button.

#### Backend APIs

| Method | Endpoint | Trigger | Response |
|---|---|---|---|
| `GET` | `/api/v1/documents` | Page mount | `Document[]` |
| `POST` | `/api/v1/documents/upload` | File drop / URL | `{ job_id, document_id, status }` |
| `GET` | `/api/v1/documents/status/{job_id}` | Polled every 2s | `{ status, progress: 0–100 }` |
| `DELETE` | `/api/v1/documents/{id}` | Delete button | `204` |

#### States

| State | UI Behavior |
|---|---|
| **Loading** | 4 skeleton document cards in the left panel. |
| **Empty** | Upload Zone fills the right panel. Left shows "No documents yet" with icon. |
| **Uploading** | Progress bar appears in the document card row for each file uploading. |
| **Indexing** | Spinner in status badge. Progress % shown inline. |
| **Failed** | Red badge. Inline error text. Retry button. |
| **Detail loading** | Skeleton for auto-summary card. |

---

### 4.3 Chat — `/chat/:sessionId`

#### Purpose
The primary interaction surface. Conversational interface powered by SSE streaming, with the Inspector Panel always present to surface the AI's reasoning for every answer.

#### Layout

```
┌─────────────┐  ┌──────────────────────────┐  ┌──────────────────┐
│ SESSION     │  │ CHAT THREAD              │  │ INSPECTOR        │
│ SIDEBAR     │  │ (scrollable)             │  │ 320px            │
│ 260px       │  │                          │  │                  │
│             │  │ Messages                 │  │ Confidence       │
│ + New Chat  │  │ ─────────────────────    │  │ Sources          │
│ ─────────── │  │ Input Bar (fixed bottom) │  │ Tool Trace       │
│ Session List│  │                          │  │ Follow-ups       │
└─────────────┘  └──────────────────────────┘  └──────────────────┘
```

#### Session Sidebar (260px)

- **"+ New Chat"** button → `POST /api/v1/sessions` → navigate to new session.
- Scrollable list of all sessions, most recent first.
- Each `SessionCard`: name, document badge count, last activity timestamp.
- Active session: brand-light background + brand-colored left border.
- `⋯` menu on hover: **Rename** (inline edit), **Export**, **Delete**.
- Below session list: "Documents in this session" accordion with **+ Attach** button.

#### Chat Thread

- Auto-scrolls to bottom on new message. **Stops** auto-scroll if user has manually scrolled up.
- **User messages:** right-aligned, `zinc-800` bg, white text.
- **Assistant messages:** left-aligned, white card, `1px border`, `shadow-sm`.
- Each assistant card contains: rendered markdown answer, confidence badge, source chips, feedback bar (👍 / 👎).
- Clicking any message in the thread updates the Inspector with that message's metadata.

#### Streaming Behavior (SSE)

> ⚠️ **Use `@microsoft/fetch-event-source`** — the endpoint is `POST`, not `GET`. Native `EventSource` cannot be used.

- On send: user message appears **immediately** (optimistic). Input disabled. "Thinking…" state begins.
- `event: token` → append text to assistant bubble. Show blinking cursor.
- `event: metadata` → parse JSON, populate Inspector, render confidence badge + source chips.
- `event: done` → remove cursor. Re-enable input. Animate in follow-up chips.
- **Stop button:** `AbortController` aborts the fetch. Partial answer is preserved with `⚠` warning icon.

#### Input Bar

- `<textarea>` that auto-grows to **5 lines** then scrolls. Never use `<input>`.
- `Enter` = send. `Shift+Enter` = newline.
- Send button disabled when: empty, currently streaming, or SSE is connecting.
- **Attach Documents** icon button — opens inline document picker.
- When no documents are attached: subtle chip — "No documents attached — agent will answer from general knowledge or web."

#### Inspector Panel Content

| Section | Content | Data Source |
|---|---|---|
| Confidence Score | Animated meter (0–1), badge (High / Medium / Low), numeric value | `metadata.confidence` |
| Sources | Card per source: doc page or web URL, excerpt, type icon | `metadata.sources[]` |
| Tool Trace | Timeline pills in order. `"none"` = direct answer. | `metadata.tool_trace[]` |
| Retrieved Chunks | Collapsible. Raw chunks with bm25 / vector / reranker scores. Debug mode only. | `metadata.chunks[]` |
| Follow-up Questions | 3 clickable chips. Click populates input and sends immediately. | `metadata.follow_up_questions[]` |

#### Backend APIs

| Method | Endpoint | Trigger | Response |
|---|---|---|---|
| `GET` | `/api/v1/sessions` | Sidebar mount | `Session[]` |
| `POST` | `/api/v1/sessions` | New Chat button | `{ id, name, created_at }` |
| `GET` | `/api/v1/sessions/{id}/history` | Session select | `Message[]` |
| `DELETE` | `/api/v1/sessions/{id}` | Delete in menu | `204` |
| `POST` | `/api/v1/chat/{session_id}` | Send message | SSE stream (`token` / `metadata` / `done`) |
| `POST` | `/api/v1/feedback` | Thumbs up / down | `201 Created` |

#### SSE Metadata Shape (type this in `src/types/index.ts`)

```ts
interface SSEMetadata {
  confidence: number; // 0.0–1.0, sigmoid-normalized reranker score
  sources: {
    type: "document" | "web";
    document_name?: string;
    page_label?: number;
    source?: string;
    text?: string;       // chunk excerpt
    url?: string;        // web sources only
    title?: string;
  }[];
  chunks: {             // debug only — show when settingsStore.showChunks = true
    chunk_id: string;
    page_label: number;
    source: string;
    text: string;
    bm25_score: number;
    reranker_score: number;
    vector_score: number;
  }[];
  tool_trace: string[]; // e.g. ["retrieve_from_document"] | ["none"] | ["retrieve_from_document", "web_search"]
}
```

#### States

| State | UI Behavior |
|---|---|
| **Idle (no session)** | Welcome card with sample questions. Prompt to attach a document. |
| **Idle (session loaded)** | Message history rendered. Input bar active. |
| **Connecting** | Spinner below last message. |
| **Streaming** | Assistant bubble growing. Blinking cursor. Input disabled. Stop button visible. |
| **Metadata received** | Inspector populates with animation. Confidence badge + source chips appear. |
| **Complete** | Input re-enabled. Follow-up chips stagger in. |
| **SSE Error** | Partial answer preserved with `⚠` icon. Toast: "Response interrupted." One auto-retry after 1500ms. |
| **No docs attached** | Warning chip in input bar. Agent still works for web / direct answers. |

---

### 4.4 Research Workspace — `/research`

#### Purpose
A structured **one-shot research interface**. Unlike chat, research is not conversational — it produces a single structured report from `POST /api/v1/research`. The UI renders the JSON response as a professional, sectioned report.

#### Layout
- Single column, max-width `860px`, centered.
- **Top:** Research Input Form.
- **Bottom:** Research Report (appears after submit, replaces a skeleton loader).

#### Research Input Form

| Field | Type | Details |
|---|---|---|
| Topic | `textarea` | Min 10 chars, max 500. Live character counter. |
| Documents | Multi-select combobox | Populated from `GET /api/v1/documents`. Searchable. Selected shown as removable chips. |
| Include Web Search | Toggle | Default **on**. Maps to `include_web` in request body. |
| Output Format | Segmented control | **Structured Report** (default) \| **Bullet Points**. Maps to `output_format`. |
| Submit | Button | "Run Research". Disabled during loading. Shows spinner + "Researching… up to 8s". |

#### Research Report Sections

1. **Summary** — rendered markdown prose or bullet list (based on `output_format`).
2. **Key Findings** — numbered list. Each finding as a card with bold statement.
3. **Sources** — two-column grid: Document Source Cards (left) + Web Source Cards (right).
4. **Confidence + Tool Trace** — badge row: confidence score + tool trace pills.
5. **Follow-up Questions** — 3 clickable cards. Click re-populates the form topic field.
6. **Export row** — "Copy JSON", "Export Markdown", "Take to Chat" buttons.

#### Take to Chat Button
Creates a new session via `POST /api/v1/sessions`, attaches the selected documents, navigates to `/chat/:id`, and pre-populates the input bar with the research topic.

#### Request Body

```json
{
  "topic": "string",
  "collection": "document_public_id",
  "include_web": true,
  "output_format": "structured"
}
```

#### Response Shape

```json
{
  "summary": "...",
  "key_findings": ["...", "..."],
  "sources": [
    { "type": "document", "page": 4, "source": "doc.pdf", "text": "..." },
    { "type": "web", "url": "https://...", "title": "..." }
  ],
  "confidence": 0.87,
  "tool_trace": ["retrieve_from_document", "web_search"],
  "follow_up_questions": ["...", "...", "..."]
}
```

#### Backend APIs

| Method | Endpoint | Trigger | Response |
|---|---|---|---|
| `GET` | `/api/v1/documents` | Form mount | `Document[]` for doc selector |
| `POST` | `/api/v1/research` | Submit button | Full research response JSON |
| `POST` | `/api/v1/sessions` | "Take to Chat" | `{ id }` then navigate |

#### States

| State | UI Behavior |
|---|---|
| **Idle** | Form visible, full width. No report below. |
| **Loading** | Form collapses to summary row. Skeleton report: gray blocks for each section. |
| **Results** | Report sections fade in with stagger. Form minimizes to a bar with "Edit" button. |
| **Error** | Toast: "Research failed. Please try again." Form re-expands. |
| **No docs selected** | Warning chip: "No documents selected — researching from web only." Still allows submit. |

---

### 4.5 Sessions — `/sessions`

#### Purpose
Full management view for all chat sessions. Provides bulk operations, filtering, and a detail drawer that goes beyond the sidebar on the Chat page.

#### Layout
- Full-width **table view** with a right detail drawer.
- Toolbar: search input, filter by document dropdown, date range picker.
- Table below toolbar. Right drawer slides in on row click.

#### Sessions Table

| Column | Content | Sortable |
|---|---|---|
| Name | Session name. Editable inline on double-click. | Yes |
| Documents | Chips for each attached document. | No |
| Messages | Integer count. | Yes |
| Last Active | Relative time ("2h ago"). | Yes |
| Actions | Open Chat, Export (Markdown), Delete. | No |

#### Session Detail Drawer
- Slides in from the right (320px) when a row is clicked.
- Contents: session name, created date, message count, attached documents.
- First 3 messages shown as compact Q&A preview.
- Buttons: **Open Full Chat**, **Export Conversation**, **Delete Session**.

#### Export Conversation
- `GET /api/v1/sessions/{id}/history` → client-side formats → browser download.
- Format options: **Markdown** (default), **Plain Text**.
- Markdown export includes: document name, date, all Q&A pairs, confidence scores, source citations.

#### Backend APIs

| Method | Endpoint | Trigger | Response |
|---|---|---|---|
| `GET` | `/api/v1/sessions` | Page mount | `Session[]` |
| `GET` | `/api/v1/sessions/{id}/history` | Drawer open / Export | `Message[]` |
| `DELETE` | `/api/v1/sessions/{id}` | Delete action | `204` |

#### States

| State | UI Behavior |
|---|---|
| **Loading** | Table skeleton: 5 rows of gray blocks. |
| **Empty** | "No sessions yet. Start a chat to create one." + New Chat button. |
| **Drawer loading** | Skeleton in the drawer while history loads. |
| **Bulk delete confirm** | Modal: "Delete N sessions? This cannot be undone." |

---

### 4.6 Settings — `/settings`

#### Purpose
User preferences and system configuration. **Scoped to what the current backend actually supports.** No placeholder calls to unimplemented endpoints.

> ⚠️ **Scope:** LLM provider and web search toggles are **frontend-persisted preferences** stored in `settingsStore` and applied as request params. There are no dedicated settings `PATCH` endpoints in the current backend.

#### Sections

**LLM Preference**
- Toggle: **Local (Ollama)** vs **Cloud (GPT-4o-mini)**. Stored in `settingsStore` (localStorage).
- Warning banner when Cloud is active: "Queries will be sent to an external API."

**Web Search**
- Toggle: Enable / Disable. Stored in `settingsStore`.
- When disabled: `include_web: false` is sent with all research and chat requests.

**Developer Options**
- **Show Retrieved Chunks** — toggle. Enables the debug Chunks section in the Inspector.
- **Inspector Default State** — toggle: open by default vs collapsed on chat page load.

**System Status**
- "Check Backend Health" button → `GET /api/v1/health` → displays raw JSON response.

**Account**
- Display name (editable). Email (read-only from auth state). Change password form.
- Danger zone: Delete Account with confirmation modal.

#### Backend APIs

| Method | Endpoint | Trigger | Response |
|---|---|---|---|
| `GET` | `/api/v1/health` | Check Status button | `{ status, components }` |

---

## 5. Component Library

> Build each component once. Never inline one-off implementations of shared UI patterns.

### 5.1 `StatusBadge`

| Variant | Color | Icon | Label |
|---|---|---|---|
| `queued` | Amber | `Clock` | Queued |
| `indexing` | Blue | `Loader` (spin) | Indexing N% |
| `indexed` | Green | `CheckCircle` | Indexed |
| `failed` | Red | `XCircle` | Failed |
| `online` | Green | `Wifi` | Online |
| `offline` | Red | `WifiOff` | Offline |

Props: `status: string`, `progress?: number`, `size?: 'sm' | 'md'`

---

### 5.2 `ConfidenceBadge` + `ConfidenceMeter`

| Range | Badge Color | Label | Inline Note |
|---|---|---|---|
| `> 0.7` | Green | High Confidence | — |
| `0.4 – 0.7` | Amber | Medium Confidence | — |
| `< 0.4` | Red | Low Confidence | "This answer may be incomplete." |
| `null / 0` | Zinc | Direct Answer | "Answered from general knowledge." |

- **`ConfidenceMeter`:** horizontal bar 0–1, animated fill (600ms ease-out on mount), color matches badge.
- Both components accept `confidence: number` prop. Always rendered together in the Inspector.

---

### 5.3 `SourceCard`

Two variants: `document` and `web`.

- **Document:** file icon, document name, page number, section excerpt (collapsible), "Open in Chat" action.
- **Web:** globe icon, URL (external link), title, snippet excerpt.
- Clicking in the Inspector highlights the corresponding citation chip in the message bubble.

---

### 5.4 `ToolTracePill`

| Tool Value | Icon | Color | Label |
|---|---|---|---|
| `retrieve_from_document` | `FileSearch` | Blue | Retrieved from document |
| `web_search` | `Globe` | Green | Web search |
| both (escalated) | Both | Blue + Green | Doc + web |
| `none` | `Sparkles` | Zinc | Answered directly |

- When multiple tools were used, pills are shown **in order** with an arrow between them.
- Used in both the Inspector Panel and the Research Report.

---

### 5.5 `DocumentCard`

Props: `name`, `fileType`, `status`, `pages`, `chunkCount`, `uploadDate`, `onChat`, `onDelete`, `onResearch`

- `StatusBadge` always visible.
- Progress bar visible **only** when `status === 'indexing'`, driven by poll data.
- Actions: Chat (icon + label), Research (icon), Delete (icon, danger style).

---

### 5.6 `SessionCard`

Props: `name`, `documentCount`, `lastMessagePreview`, `lastActiveAt`, `onOpen`, `onDelete`

- Document count shown as a badge.
- Last message preview truncated to 60 characters, muted text.

---

### 5.7 `MarkdownRenderer`

Wraps `react-markdown` with custom renderers:

- Code blocks: syntax highlighted via `react-syntax-highlighter` (`github` / `github-dark` themes).
- Links: always open in new tab with `rel="noopener noreferrer"`.
- Tables: styled with Tailwind, not raw HTML defaults.
- Headings: capped at `h3` inside answers to avoid hierarchy conflicts with page headings.

---

### 5.8 `FollowUpChip`

- On click: populates the input bar (chat) or topic textarea (research) and **immediately submits**.
- Animation: staggered fade-up, 100ms delay per chip, after `metadata` event arrives.

---

### 5.9 `UploadZone`

- `react-dropzone` powered. Accepts: `.pdf`, `.docx`, `.txt`, `.md`.
- Drag-over state: dashed border turns solid, background tints to `brand-light`.
- URL input field below the drop zone for web URL ingestion.
- File size validated client-side before any network call.

---

### 5.10 `DocumentUploadModal`

- Modal wrapper around `UploadZone`.
- Opened from Dashboard quick actions, Documents page, and Chat sidebar.
- On success: adds document to `documentStore`, closes modal, shows progress toast.
- **Reusable from any page** — does not require being on the Documents page.

---

### 5.11 `ResearchReportCard`

Props: `summary`, `keyFindings[]`, `sources[]`, `confidence`, `toolTrace[]`, `followUpQuestions[]`

- Renders each section as a distinct card with heading and content.
- Export and Take to Chat actions are buttons inside this component.

---

## 6. User Flows

### 6.1 Upload Flow

| Step | UI State | Backend Call |
|---|---|---|
| Drop file | File preview appears in UploadZone | — |
| Submit | Upload begins | `POST /api/v1/documents/upload` |
| Queued | Document card appears with "Queued" badge | Response: `{ job_id }` |
| Polling | Status updates every 2s | `GET /status/{job_id}` |
| Indexing | Progress bar animates, badge shows "Indexing N%" | Repeated polls |
| Indexed | Badge turns green, auto-summary loads | Final poll: `status=indexed` |
| Failed | Badge turns red, Retry button shown | Final poll: `status=failed` |

### 6.2 Chat Flow

| Step | UI State | SSE Event / Call |
|---|---|---|
| User types | Input textarea active | — |
| User sends | Message appends optimistically. Input disabled. | `POST /api/v1/chat/{session_id}` |
| Connecting | Spinner below last message | Connection opening |
| Streaming | Bubble grows with each token. Cursor blinks. | `event: token` (repeated) |
| Metadata | Inspector populates. Confidence + source chips render. | `event: metadata` |
| Complete | Input re-enabled. Follow-up chips animate in. | `event: done` |
| Feedback | Thumbs icon highlighted. | `POST /api/v1/feedback` |

### 6.3 Research Flow

| Step | UI State | Backend Call |
|---|---|---|
| Fill form | Topic entered, docs selected, options set | — |
| Submit | Form collapses. Skeleton report appears. | `POST /api/v1/research` |
| Loading | Skeleton for up to 8s | Waiting |
| Report renders | Sections fade in with stagger | Response arrives |
| Click follow-up | Topic field refills. Form expands. | — |
| Take to Chat | New session created. Navigate to `/chat`. | `POST /api/v1/sessions` then navigate |

---

## 7. Component Hierarchy

```
App
├── AuthLayout
│   ├── LoginPage
│   └── RegisterPage
└── AppLayout  (authenticated shell)
    ├── Sidebar
    │   ├── NavLink (×6)
    │   └── UserMenu
    ├── TopBar
    │   ├── PageTitle
    │   ├── LLMModeBadge
    │   └── DarkModeToggle
    └── PageOutlet
        ├── DashboardPage
        │   ├── QuickActionCard (×3)
        │   ├── SessionCard (×5, recent)
        │   ├── DocumentCard (×5, compact)
        │   └── HealthStrip
        ├── DocumentsPage
        │   ├── UploadZone
        │   ├── DocumentList
        │   │   └── DocumentCard (×n)
        │   └── DocumentDetailPanel
        │       ├── StatusBadge
        │       └── AutoSummaryCard
        ├── ChatPage
        │   ├── SessionSidebar
        │   │   ├── SessionCard (×n)
        │   │   └── DocumentAttachPanel
        │   ├── ChatWindow
        │   │   ├── MessageList
        │   │   │   ├── UserMessage
        │   │   │   └── AssistantMessage
        │   │   │       ├── MarkdownRenderer
        │   │   │       ├── ConfidenceBadge
        │   │   │       ├── SourceCitationChip (×n)
        │   │   │       ├── FollowUpChip (×3)
        │   │   │       └── FeedbackBar
        │   │   └── ChatInputBar
        │   └── InspectorPanel
        │       ├── ConfidenceMeter
        │       ├── SourceCard (×n)
        │       ├── ToolTracePill
        │       ├── ChunkDebugView  (conditional on settingsStore)
        │       └── FollowUpChip (×3)
        ├── ResearchPage
        │   ├── ResearchForm
        │   │   ├── DocumentMultiSelect
        │   │   └── OptionToggles
        │   └── ResearchReportCard
        │       ├── ConfidenceBadge
        │       ├── ToolTracePill
        │       ├── SourceCard (×n)
        │       └── FollowUpChip (×3)
        ├── SessionsPage
        │   ├── SessionsTable
        │   └── SessionDetailDrawer
        └── SettingsPage
            ├── LLMToggle
            ├── WebSearchToggle
            ├── DeveloperOptions
            └── AccountSection
```

---

## 8. Folder Structure

```
src/
├── api/                    # One file per backend domain. No raw fetch() outside here.
│   ├── auth.ts             # login, register
│   ├── documents.ts        # upload, list, status, delete
│   ├── sessions.ts         # create, list, history, delete
│   ├── chat.ts             # SSE stream function (fetchEventSource wrapper)
│   ├── research.ts         # POST /research
│   └── health.ts           # GET /health
│
├── components/
│   ├── layout/
│   │   ├── AppLayout.tsx       # Shell: Sidebar + TopBar + Outlet
│   │   ├── Sidebar.tsx
│   │   ├── TopBar.tsx
│   │   └── InspectorPanel.tsx  # Core UI. Lives in layout, driven by chatStore.
│   │
│   ├── shared/             # Reusable components from Section 5
│   │   ├── StatusBadge.tsx
│   │   ├── ConfidenceBadge.tsx
│   │   ├── ConfidenceMeter.tsx
│   │   ├── SourceCard.tsx
│   │   ├── ToolTracePill.tsx
│   │   ├── DocumentCard.tsx
│   │   ├── SessionCard.tsx
│   │   ├── MarkdownRenderer.tsx
│   │   ├── FollowUpChip.tsx
│   │   ├── UploadZone.tsx
│   │   ├── DocumentUploadModal.tsx
│   │   └── ResearchReportCard.tsx
│   │
│   └── ui/                 # shadcn/ui output. Do not edit manually.
│
├── hooks/
│   ├── useSSEChat.ts           # Most critical hook. Full SSE lifecycle.
│   ├── useDocumentPolling.ts   # Polls /status/{job_id} while docs are indexing.
│   └── useHealthCheck.ts       # 30s poll of /health.
│
├── pages/
│   ├── auth/
│   │   ├── LoginPage.tsx
│   │   └── RegisterPage.tsx
│   ├── DashboardPage.tsx
│   ├── DocumentsPage.tsx
│   ├── ChatPage.tsx
│   ├── ResearchPage.tsx
│   ├── SessionsPage.tsx
│   └── SettingsPage.tsx
│
├── stores/                 # Zustand. One file per domain.
│   ├── authStore.ts        # persisted (localStorage)
│   ├── documentStore.ts
│   ├── sessionStore.ts
│   ├── chatStore.ts        # messages, streaming state, active metadata
│   ├── researchStore.ts    # current research result + loading state
│   └── uiStore.ts          # persisted: sidebar, inspector, dark mode
│
├── types/
│   └── index.ts            # All interfaces. Single source of truth.
│
├── lib/
│   ├── axios.ts            # Single Axios instance + auth + 401 interceptors
│   └── utils.ts            # formatDate, truncate, cn(), etc.
│
└── App.tsx                 # Route definitions only. No business logic.
```

---

## 9. State Management

### 9.1 Store Overview

| Store | Owns | Persisted | Key Actions |
|---|---|---|---|
| `authStore` | JWT token, user profile | localStorage | `login()`, `logout()`, `setUser()` |
| `documentStore` | Document list, job status map | No | `addDocument()`, `updateStatus()`, `removeDocument()` |
| `sessionStore` | All sessions, active session ID | No | `setSessions()`, `setActiveSession()`, `deleteSession()` |
| `chatStore` | Messages, streaming state, active metadata | No | `appendToken()`, `setMetadata()`, `setStreaming()` |
| `researchStore` | Current research result, loading | No | `setResult()`, `setLoading()`, `clear()` |
| `uiStore` | Sidebar collapsed, inspector open, dark mode | localStorage | `toggleSidebar()`, `toggleInspector()`, `setDarkMode()` |

### 9.2 `chatStore` Shape

```ts
interface ChatStore {
  messages: Message[];
  streamingState: 'idle' | 'connecting' | 'streaming' | 'done' | 'error';
  activeMessageId: string | null;
  activeMetadata: SSEMetadata | null;
  partialText: string;
  abortFn: (() => void) | null;
}
```

### 9.3 Polling Strategy

- **Document polling:** `useDocumentPolling` runs every **2000ms** while `documentStore` has any job with `status === 'queued' | 'indexing'`. Stops when all reach a terminal state.
- Store the interval ID in a `useRef`. Clear it in the `useEffect` cleanup. Never create a new interval on re-render.
- **Health check:** `useHealthCheck` runs every **30000ms**. Pauses when `document.visibilityState === 'hidden'`. Resumes on `visibilitychange`.

### 9.4 SSE State Machine

| State | Transitions From | Transitions To | UI Effect |
|---|---|---|---|
| `idle` | `done` or initial | `connecting` (on send) | Input enabled |
| `connecting` | `idle` | `streaming` or `error` | Spinner shows |
| `streaming` | `connecting` | `metadata` (on metadata event) | Tokens appending, cursor blinks |
| `metadata` | `streaming` | `done` (on done event) | Inspector populates |
| `done` | `metadata` | `idle` | Input re-enabled, follow-ups appear |
| `error` | `connecting` or `streaming` | `idle` (after toast) | Error toast, retry available |

---

## 10. Backend Integration by Page

### 10.1 Auth Interceptor (Set Up First)

Configure in `src/lib/axios.ts`:

```ts
// Request interceptor — inject JWT
axiosInstance.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Response interceptor — handle 401
axiosInstance.interceptors.response.use(
  (res) => res,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

No individual component should ever manually add auth headers.

### 10.2 HTTP Error Codes

| Code | Scenario | Frontend Action |
|---|---|---|
| `400` | Bad request body | Inline form error for the relevant field |
| `401` | Expired / invalid token | Axios interceptor: clear auth, redirect `/login` |
| `403` | Accessing another user's resource | Toast: "Access denied" |
| `404` | Document or session not found | Toast error + remove from store |
| `413` | File too large | Inline error in UploadZone: "File exceeds size limit" |
| `422` | Validation error | Toast with `error.message` from backend |
| `500` | Backend error | Toast: "Something went wrong." + Retry button |
| `503` | Backend down | Full-page `OfflineBanner`, poll `/health` every 10s |

### 10.3 SSE Error Handling

- On `fetchEventSource` error: wait **1500ms**, attempt one reconnect.
- If reconnect fails: set `chatStore.streamingState = 'error'`. Toast: "Response interrupted — please retry."
- **Preserve `partialText`** in the message bubble. Mark with `⚠` icon and label "Response may be incomplete."
- Never discard partial answers. Users should be able to read what arrived and decide to retry.

---

## 11. Mobile Behavior

### 11.1 Breakpoints

| Breakpoint | Width | Tailwind Prefix |
|---|---|---|
| Mobile | < 768px | (base) |
| Tablet | 768px – 1023px | `md:` |
| Desktop | 1024px – 1439px | `lg:` |
| Wide | 1440px+ | `xl:` |

### 11.2 Per-Page Mobile Layout

| Page | Mobile Changes |
|---|---|
| Dashboard | Single column. Quick Actions stack vertically. Stats: 2×2 grid. |
| Documents | List goes full width. Detail panel becomes a **bottom sheet** that slides up on card tap. |
| Chat | Session sidebar hidden behind hamburger. Inspector becomes a **bottom drawer** opened via "View Details" tap. |
| Research | Single column. Form fields stack. Report sections collapse to accordions. |
| Sessions | Table becomes a card list. Detail drawer goes full-screen. |
| Settings | Sections stack vertically. No structural change needed. |

### 11.3 Touch Interaction Notes

- All tap targets minimum **44×44px** (WCAG 2.5.5).
- Swipe right on a session card in the Sessions list to reveal Delete (mobile only).
- Swipe down on the Inspector bottom drawer to dismiss.
- Long-press on a message bubble: Copy Text, Regenerate, Report Issue.

---

## 12. Future Features (v2 Scope)

> Do not stub these out or add placeholder UI. Structure stores and API layers so adding them requires minimal refactoring.

| Feature | Why It Fits DocuMind | Prep Needed |
|---|---|---|
| Research Agent (Planner + Critic) | Multi-step autonomous research with self-correction | `researchStore` needs `planState` and `iterationHistory` |
| Document Compare | Side-by-side Q&A across two documents | `ChatPage` needs a compare mode layout variant |
| Collections / Workspaces | Group documents by project | `documentStore` needs collection grouping; sessions link to `collection_id` |
| Notebook / Highlights | Save answer excerpts to a persistent notebook | New `notebookStore`; `AssistantMessage` needs "Save Highlight" action |
| Voice Input | Speak questions instead of typing | `ChatInputBar` needs mic icon + Web Speech API |
| Team Collaboration | Shared sessions and document libraries | `authStore` needs `team_id`; session ownership becomes shared |
| Export to Notion / Obsidian | Push reports to PKM tools | New `api/integrations.ts`; Settings gains Integrations section |

## 13. Updated Tech Stack & UI Refinements

This section documents the specific technologies adopted and the UI/UX enhancements implemented in the most recent updates to fulfill the premium SaaS aesthetic goals.

### 13.1 Frontend Technology Stack
- **Language**: TypeScript
- **UI Framework**: React (Vite)
- **Styling**: Tailwind CSS v4 (using native CSS variables for dynamic theming)
- **Component Library**: shadcn/ui
- **Icons**: Lucide React
- **Animations**: Framer Motion (used for subtle layout transitions and interactive hover effects)
- **Forms & Validation**: React Hook Form + Zod
- **Data Fetching & Caching**: TanStack Query (React Query)
- **State Management**: Zustand (UI state, leveraging `persist` middleware for theme and preferences)
- **Markdown Rendering**: react-markdown + remark-gfm
- **Data Visualization**: Recharts

### 13.2 Key UI Enhancements Implemented
- **Full Dark Mode Integration**: Engineered a robust dark mode utilizing `.dark` mapping in Tailwind v4 and dynamic CSS variables in `index.css`. State is persisted via Zustand and can be toggled directly from the sidebar or settings.
- **Semantic Color Coding**: Replaced uniform icons with semantic coloring to improve scannability across the app (Emerald for Spreadsheets, Sky Blue for Web Links, Rose for PDFs, Amber for Text/Markdown).
- **Premium Data Visualizations**: Upgraded the Recharts area chart in the Dashboard to use a multi-stop Indigo-to-Cyan linear gradient.
- **Glowing Interactive Flourishes**: Integrated subtle, iridescent gradients on hover for Quick Action Bento cards, gave the AI Assistant avatar a distinctive gradient glow, and added pulsing shadow effects to the Node Health indicator.
- **Polished Layout Constraints**: Refined layout spacing, standardized widget borders, and leveraged layered surface containers to elevate distinct UI zones.

---

*Build from this document. The backend is ready.*