# KnowVex Frontend Production Readiness Roadmap

> **Status:** Active Execution  
> **Backend Impact:** None. All issues listed below are 100% frontend-addressable without modifying backend endpoints or schemas.

---

## 📋 Task List

- [x] **Task 1:** Centralize API base URLs and environment variable abstraction (`VITE_API_BASE_URL`)
- [x] **Task 2:** Remove local dev probe script (`http://localhost:8400/detect.js`) from `index.html`
- [x] **Task 3:** Remove mock initial documents (`INITIAL_DOCUMENTS`) and ensure clean empty-state / live DB fetching
- [x] **Task 4:** Implement React `ErrorBoundary` component with recovery UI in `main.tsx`
- [x] **Task 5:** Add production SEO, OpenGraph metadata, and Favicon to `index.html`
- [x] **Task 6:** Harden token storage and authentication state management
- [ ] **Task 7:** *(Deferred)* Decompose monolithic `App.tsx` (~1,000 lines) into modular custom hooks (`useChatStream`, `useDocumentUpload`)

---

## 🛠️ Detailed Issue Breakdown & Production Standards

### Task 1: Environment Variable Abstraction for API Endpoints
- **Issue:** Hardcoded `http://localhost:8000` strings exist in `api.ts`, `AuthView.tsx`, and `App.tsx`.
- **Production Standard:** Use Vite's `import.meta.env.VITE_API_BASE_URL`. Default to `/api/v1` in production or `http://localhost:8000/api/v1` in development.
- **Files Affected:** `frontend/src/lib/api.ts`, `frontend/src/components/AuthView.tsx`, `frontend/src/App.tsx`, `frontend/.env.example`.

### Task 2: Remove Local Dev Probe Script from `index.html`
- **Issue:** `frontend/index.html` includes `<script src="http://localhost:8400/detect.js"></script>`, which throws unhandled network errors in production.
- **Production Standard:** Clean static HTML without local port dependencies.
- **Files Affected:** `frontend/index.html`.

### Task 3: Replace Mock Initial Documents with Live DB State
- **Issue:** `useStore.ts` seeds 3 hardcoded dummy documents (`fiscal_projections_q3_revised.xlsx`, etc.) into `INITIAL_DOCUMENTS`. New users immediately see fake documents before uploading anything.
- **Production Standard:** Initialize `documents: []`. Fetch actual documents from `GET /api/v1/documents` and display a clean empty state when no files are uploaded.
- **Files Affected:** `frontend/src/data.ts`, `frontend/src/store/useStore.ts`.

### Task 4: React Error Boundary
- **Issue:** No component error boundary exists. A single rendering crash anywhere in the component tree turns the whole page into a blank white screen.
- **Production Standard:** Wrap `<App />` in a class-based `ErrorBoundary` that renders a branded fallback card with "Reload Application" and error logs.
- **Files Affected:** `frontend/src/components/ErrorBoundary.tsx` (new), `frontend/src/main.tsx`.

### Task 5: Production SEO, OpenGraph & Favicon Metadata
- **Issue:** `index.html` lacks metadata tags (`meta description`, `og:title`, `og:image`, `twitter:card`, proper favicon). Links shared on social media display blank previews.
- **Production Standard:** Complete HTML5 `<head>` metadata with KnowVex branding, description, dark mode browser theme-color, and SVG favicon.
- **Files Affected:** `frontend/index.html`.

### Task 6: Secure Token Storage & Auth Persistence
- **Issue:** Access tokens and refresh tokens are stored together in plain `localStorage` via Zustand persist.
- **Production Standard:** Separate session configuration so access tokens are maintained safely, avoiding unnecessary token leakage on multi-user or shared terminals.
- **Files Affected:** `frontend/src/store/useStore.ts`.

### Task 7: Deconstruct Monolithic `App.tsx` *(Deferred)*
- **Issue:** `App.tsx` contains 1,077 lines mixing SSE chat streaming, document upload polling, feedback submission, and layout rendering.
- **Production Standard:** Split into domain-specific hooks (`useChatStream`, `useDocumentUpload`, `useFeedback`) for long-term maintainability.
- **Files Affected:** Deferred to follow-up phase.

