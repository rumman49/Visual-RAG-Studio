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
 * API key for hosted instances (AD-12). Held in module memory only: never written to
 * localStorage/sessionStorage and never read from a VITE_ variable.
 */
let apiKey: string | null = null
export type AuthFailure = 'api_key_required' | 'api_key_invalid'
type AuthListener = (failure: AuthFailure) => void
const authListeners = new Set<AuthListener>()

export function setApiKey(key: string | null): void {
  apiKey = key && key.trim() ? key.trim() : null
}

export function hasApiKey(): boolean {
  return apiKey !== null
}

/** Subscribe to 401 responses caused by a missing or invalid API key. Returns an unsubscribe. */
export function onAuthFailure(listener: AuthListener): () => void {
  authListeners.add(listener)
  return () => authListeners.delete(listener)
}

async function notifyIfAuthFailure(response: Response): Promise<Response> {
  if (response.status !== 401) return response
  try {
    const body = await response.clone().json()
    const code = body?.detail?.code
    if (code === 'api_key_required' || code === 'api_key_invalid') {
      authListeners.forEach((listener) => listener(code))
    }
  } catch {
    // Non-JSON 401s are not ours.
  }
  return response
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

  const attachKey = targetUrl.match(/\/api\/v1(\/|$|\?)/) !== null && apiKey !== null
  const withKey = (headers?: HeadersInit): Headers => {
    const merged = new Headers(headers)
    merged.set('X-API-Key', apiKey as string)
    return merged
  }

  if (input instanceof Request) {
    const request = new Request(targetUrl, input)
    const outgoing = attachKey ? new Request(request, { headers: withKey(request.headers) }) : request
    return notifyIfAuthFailure(await fetch(outgoing))
  }

  const outgoingInit = attachKey ? { ...init, headers: withKey(init?.headers) } : init
  return notifyIfAuthFailure(await fetch(targetUrl, outgoingInit))
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

