import { useStore } from '../store/useStore';

const API_BASE = 'http://localhost:8000/api/v1';

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(status: number, message: string, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export class NetworkError extends Error {
  constructor() {
    super('No internet connection. Please check your network and try again.');
    this.name = 'NetworkError';
  }
}

interface CustomRequestInit extends RequestInit {
  _isRetry?: boolean;
  retries?: number;
}

const AUTH_EXCLUDED_REFRESH_ENDPOINTS = [
  '/auth/login',
  '/auth/register',
  '/auth/refresh-token',
  '/auth/forgot-password',
  '/auth/reset-password',
  '/auth/verify-email',
  '/system/status',
  '/health',
  '/health/db',
];

let refreshPromise: Promise<string | null> | null = null;

export async function authenticatedFetch(endpointOrUrl: string, options: CustomRequestInit = {}): Promise<Response> {
  const token = useStore.getState().accessToken;
  const headers = new Headers(options.headers || {});
  
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  
  // Only set Content-Type to JSON if it's not a FormData or URLSearchParams request
  if (!headers.has('Content-Type') && !(options.body instanceof FormData) && !(options.body instanceof URLSearchParams)) {
    headers.set('Content-Type', 'application/json');
  }
  if (options.body instanceof URLSearchParams && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/x-www-form-urlencoded');
  }

  const fullUrl = endpointOrUrl.startsWith('http://') || endpointOrUrl.startsWith('https://')
    ? endpointOrUrl
    : `${API_BASE}${endpointOrUrl.startsWith('/') ? '' : '/'}${endpointOrUrl}`;

  // Safe transient retry for network hiccups (default 1 retry for GET, 0 for mutations unless specified)
  const maxNetworkRetries = options.retries ?? (options.method && options.method !== 'GET' ? 0 : 1);
  let networkAttempt = 0;
  let response: Response | undefined;

  while (networkAttempt <= maxNetworkRetries) {
    try {
      response = await fetch(fullUrl, {
        ...options,
        credentials: 'include',
        headers,
      });
      break;
    } catch (err: any) {
      networkAttempt++;
      if (networkAttempt > maxNetworkRetries) {
        throw new NetworkError();
      }
      await new Promise(res => setTimeout(res, 350 * networkAttempt));
    }
  }

  if (!response) {
    throw new NetworkError();
  }

  const isAuthExcluded = AUTH_EXCLUDED_REFRESH_ENDPOINTS.some(path => fullUrl.includes(path));

  // Handle 401 Unauthorized with automatic refresh using HttpOnly cookie
  if (response.status === 401 && !isAuthExcluded && !options._isRetry) {
    const hasAuth = !!useStore.getState().accessToken || useStore.getState().isAuthenticated;
    if (hasAuth) {
      if (!refreshPromise) {
        refreshPromise = fetch(`${API_BASE}/auth/refresh-token`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
        }).then(async res => {
          if (res.ok) {
            const data = await res.json();
            useStore.getState().setAccessToken(data.access_token);
            return data.access_token;
          }
          throw new Error('Refresh failed');
        }).catch(() => null).finally(() => {
          refreshPromise = null;
        });
      }

      const newToken = await refreshPromise;
      if (newToken) {
        // Retry original request once with new token
        const retryHeaders = new Headers(options.headers || {});
        retryHeaders.set('Authorization', `Bearer ${newToken}`);
        return authenticatedFetch(endpointOrUrl, {
          ...options,
          headers: retryHeaders,
          _isRetry: true,
        });
      }
    }

    if (!isAuthExcluded) {
      useStore.getState().logout();
    }
  }

  return response;
}

async function apiFetch(endpoint: string, options: CustomRequestInit = {}) {
  const response = await authenticatedFetch(endpoint, options);

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    const msg = errorData?.detail?.[0]?.msg || errorData?.detail || (response.status === 401 ? 'Unauthorized' : 'API Error');
    throw new ApiError(response.status, msg, errorData);
  }

  // Handle empty responses (like 204 or empty 200s)
  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

export const api = {
  // Auth
  login: async (email: string, password: string) => {
    // OAuth2PasswordBearer expects form data
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    
    return apiFetch('/auth/login', {
      method: 'POST',
      body: formData,
    });
  },
  
  register: async (name: string, email: string, password: string) => {
    return apiFetch('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ name, email, password }),
    });
  },

  verifyEmail: async (token: string) => {
    return apiFetch('/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ token }),
    });
  },

  forgotPassword: async (email: string) => {
    return apiFetch('/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  resetPassword: async (token: string, newPassword: string) => {
    return apiFetch('/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ token, new_password: newPassword }),
    });
  },

  refreshToken: async (refreshToken: string) => {
    return apiFetch('/auth/refresh-token', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  },

  logout: async () => {
    try {
      await apiFetch('/auth/logout', { method: 'POST' });
    } catch (e) {
      // Ignore logout network errors
    }
    useStore.getState().logout();
  },

  logoutAll: async () => {
    try {
      await apiFetch('/auth/logout-all', { method: 'POST' });
    } catch (e) {
      // Ignore logout network errors
    }
    useStore.getState().logout();
  },

  getMe: () => apiFetch('/auth/me'),

  updateMe: (body: { name?: string; email?: string }) => apiFetch('/user/me', {
    method: 'PATCH',
    body: JSON.stringify(body),
  }),

  deleteMe: () => apiFetch('/user/me', { method: 'DELETE' }),

  // Documents
  getDocuments: () => apiFetch('/documents/all'),
  
  uploadDocument: (file: File, sessionId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    return apiFetch('/documents/upload', {
      method: 'POST',
      body: formData
    });
  },
  
  deleteDocument: (publicId: string) => apiFetch(`/document/${publicId}`, { method: 'DELETE' }),

  getDocumentStatus: (jobId: string) => apiFetch(`/documents/status/${jobId}`),

  // Sessions
  getSessions: () => apiFetch('/sessions/'),
  
  createSession: () => apiFetch('/sessions/', { method: 'POST' }),

  deleteSession: (sessionId: string) => apiFetch(`/sessions/${sessionId}`, { method: 'DELETE' }),
  
  getSessionHistory: (sessionId: string) => apiFetch(`/sessions/${sessionId}/history`),

  getSessionDocuments: (sessionId: string) => apiFetch(`/sessions/${sessionId}/documents`),

  linkDocumentToSession: (sessionId: string, documentId: string) => apiFetch(`/sessions/${sessionId}/documents/${documentId}`, { method: 'POST' }),
  
  unlinkDocumentFromSession: (sessionId: string, documentId: string) => apiFetch(`/sessions/${sessionId}/documents/${documentId}`, { method: 'DELETE' }),

  renameSession: (sessionId: string, newName: string) => apiFetch(`/sessions/${sessionId}/rename`, {
    method: 'POST',
    body: JSON.stringify({ name: newName })
  }),

  submitFeedback: (payload: {
    message_id: number;
    query: string;
    answer: string;
    rating: number;
    comment: string;
    confidence: number;
    tool_used?: string;
  }) => apiFetch('/feedback', {
    method: 'POST',
    body: JSON.stringify(payload)
  }),

  // Chat/Research Stream with unified 401 refresh & retry
  streamChat: (sessionId: string, question: string) =>
    authenticatedFetch(`/chat/${sessionId}`, {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),

  getChatEndpoint: (sessionId: string) => `${API_BASE}/chat/${sessionId}`,

  getHealth: async () => {
    return apiFetch('/health/db');
  },

  // Analytics
  getActivity: async (days: number = 7) => {
    return apiFetch(`/analytics/activity?days=${days}`);
  },
};

