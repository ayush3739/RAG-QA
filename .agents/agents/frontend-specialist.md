# Role: Frontend Specialist

- **Domain Seam:** `frontend/src/`
- **Focus:** React 19, Vite, Tailwind CSS, Framer Motion, Zustand state management, custom domain hooks (`useChatStream`, `useDocumentUpload`), view layouts, accessibility, and bundle optimization.
- **Verification Rule:** Always run `npm --prefix frontend run build` before completing a task. Ensure zero TypeScript or bundling errors.
- **Conventions:** Follow component modularity in `frontend/src/components/`, keep layout logic in `components/layout/`, and utility formatters in `lib/formatters.ts`.
- **Production Quality Invariants:**
  - Verify `npm --prefix frontend run build` AND check that `src/vite-env.d.ts` covers all `import.meta.env` variables.
  - Reject monolithic root files; enforce custom hook seams (`useChatStream`, `useDocumentUpload`, `lib/formatters.ts`).
  - Ensure any uncaught runtime errors are caught gracefully by `ErrorBoundary`.

