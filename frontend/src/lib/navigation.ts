/**
 * Modern HTML5 History navigation helper for Single-Page Application (SPA) routing.
 * Provides pure URL routing (e.g. /dashboard, /chat/new, /architecture) without '#' hash fragments.
 */
export function navigate(path: string, options?: { replace?: boolean }) {
  if (options?.replace) {
    window.history.replaceState(null, "", path);
  } else {
    window.history.pushState(null, "", path);
  }
  window.dispatchEvent(new PopStateEvent("popstate"));
}

