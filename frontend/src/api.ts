/**
 * Visual RAG Studio - Centralized Frontend API Client
 *
 * Resolves the backend API base URL from the frontend environment variable
 * `VITE_API_BASE_URL` (configured via frontend/.env).
 *
 * Configuration Flow:
 * Frontend .env -> Frontend API Client -> Backend API -> Python RAG System
 */

/**
 * Returns the configured backend API base URL.
 * Reads `VITE_API_BASE_URL` from the Vite environment.
 * Defaults to 'http://127.0.0.1:8001' for local development.
 */
export function getApiBaseUrl(): string {
  const envUrl = import.meta.env?.VITE_API_BASE_URL
  if (typeof envUrl === 'string' && envUrl.trim().length > 0) {
    return envUrl.trim().replace(/\/+$/, '')
  }
  return 'http://127.0.0.1:8001'
}

/**
 * Resolves a given path into a fully qualified API URL.
 * If the path is already an absolute URL (starts with http:// or https://), it is returned as is.
 * Otherwise, prepends getApiBaseUrl().
 */
export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path
  }
  const baseUrl = getApiBaseUrl()
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return baseUrl ? `${baseUrl}${normalizedPath}` : normalizedPath
}

/**
 * Centralized fetch wrapper that routes all frontend API requests through the
 * environment-configured backend API base URL.
 */
export async function apiFetch(
  input: string | URL | Request,
  init?: RequestInit,
): Promise<Response> {
  const targetUrl =
    typeof input === 'string'
      ? apiUrl(input)
      : input instanceof URL
        ? input.toString()
        : apiUrl(input.url)

  if (input instanceof Request) {
    return fetch(new Request(targetUrl, input))
  }

  return fetch(targetUrl, init)
}

/**
 * Centralized API client helper object with convenience methods.
 */
export const apiClient = {
  getBaseUrl: getApiBaseUrl,
  url: apiUrl,
  fetch: apiFetch,
  get: (path: string, init?: RequestInit) => apiFetch(path, { ...init, method: 'GET' }),
  post: (path: string, body?: any, init?: RequestInit) =>
    apiFetch(path, {
      ...init,
      method: 'POST',
      headers:
        body instanceof FormData
          ? init?.headers
          : { 'Content-Type': 'application/json', ...init?.headers },
      body:
        body instanceof FormData
          ? body
          : body !== undefined
            ? JSON.stringify(body)
            : undefined,
    }),
  delete: (path: string, init?: RequestInit) => apiFetch(path, { ...init, method: 'DELETE' }),
}

export default apiClient

