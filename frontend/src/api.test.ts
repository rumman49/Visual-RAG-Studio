import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { apiClient, apiFetch, apiUrl, getApiBaseUrl } from './api'

describe('Centralized Frontend API Client', () => {
  const originalEnv = import.meta.env.VITE_API_BASE_URL

  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })

  afterEach(() => {
    ;(import.meta.env as Record<string, string | undefined>).VITE_API_BASE_URL = originalEnv
  })

  it('reads backend API URL from VITE_API_BASE_URL environment variable', () => {
    ;(import.meta.env as Record<string, string | undefined>).VITE_API_BASE_URL = 'http://127.0.0.1:8001'
    expect(getApiBaseUrl()).toBe('http://127.0.0.1:8001')
  })

  it('strips trailing slashes from VITE_API_BASE_URL', () => {
    ;(import.meta.env as Record<string, string | undefined>).VITE_API_BASE_URL = 'http://127.0.0.1:8001///'
    expect(getApiBaseUrl()).toBe('http://127.0.0.1:8001')
  })

  it('falls back to default local backend URL if VITE_API_BASE_URL is not set', () => {
    ;(import.meta.env as Record<string, string | undefined>).VITE_API_BASE_URL = ''
    expect(getApiBaseUrl()).toBe('http://127.0.0.1:8001')
  })

  it('resolves relative paths to fully-qualified backend URLs', () => {
    ;(import.meta.env as Record<string, string | undefined>).VITE_API_BASE_URL = 'http://127.0.0.1:8001'
    expect(apiUrl('/api/v1/query')).toBe('http://127.0.0.1:8001/api/v1/query')
    expect(apiUrl('api/v1/documents')).toBe('http://127.0.0.1:8001/api/v1/documents')
    expect(apiUrl('/api/health/live')).toBe('http://127.0.0.1:8001/api/health/live')
  })

  it('leaves absolute URLs unchanged', () => {
    expect(apiUrl('http://example.com/api')).toBe('http://example.com/api')
    expect(apiUrl('https://example.com/api')).toBe('https://example.com/api')
  })

  it('routes apiFetch calls through the configured backend base URL', async () => {
    ;(import.meta.env as Record<string, string | undefined>).VITE_API_BASE_URL = 'http://127.0.0.1:8001'
    const mockFetch = vi.fn().mockResolvedValue(new Response('{"status":"ok"}', { status: 200 }))
    vi.stubGlobal('fetch', mockFetch)

    const res = await apiFetch('/api/v1/registry')
    expect(mockFetch).toHaveBeenCalledWith('http://127.0.0.1:8001/api/v1/registry', undefined)
    expect(res.status).toBe(200)
  })

  it('supports apiClient convenience methods', async () => {
    ;(import.meta.env as Record<string, string | undefined>).VITE_API_BASE_URL = 'http://127.0.0.1:8001'
    const mockFetch = vi.fn().mockResolvedValue(new Response('{"status":"ok"}', { status: 200 }))
    vi.stubGlobal('fetch', mockFetch)

    await apiClient.get('/api/health/ready')
    expect(mockFetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8001/api/health/ready',
      expect.objectContaining({ method: 'GET' }),
    )

    await apiClient.post('/api/v1/workflows/validate', { test: true })
    expect(mockFetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8001/api/v1/workflows/validate',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ test: true }),
      }),
    )

    await apiClient.delete('/api/v1/documents/123')
    expect(mockFetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8001/api/v1/documents/123',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })
})
