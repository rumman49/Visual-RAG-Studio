import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { setApiKey } from './api'
import { AuthBoundary } from './AuthBoundary'

vi.mock('./VisualBuilder', async () => {
  const { apiFetch } = await import('./api')
  const { useEffect } = await import('react')
  return { VisualBuilder: () => { useEffect(() => { void apiFetch('/api/v1/registry') }, []); return <div>builder</div> } }
})

const unauthorized = (code: string) =>
  new Response(JSON.stringify({ detail: { code, message: 'x' } }), { status: 401 })

describe('API key flow', () => {
  afterEach(() => { cleanup(); setApiKey(null); vi.unstubAllGlobals(); window.localStorage.clear(); window.sessionStorage.clear() })

  it('prompts on 401, sends X-API-Key on retry, and keeps the key out of storage', async () => {
    const seen: (string | null)[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/api/health/ready')) return new Response(JSON.stringify({ auth: 'required' }), { status: 200 })
      const key = new Headers(init?.headers).get('X-API-Key')
      seen.push(key)
      return key === 'good' ? new Response('{}', { status: 200 }) : unauthorized(key ? 'api_key_invalid' : 'api_key_required')
    }))
    render(<AuthBoundary />)
    expect(await screen.findByText('API key required.')).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('API key'), 'bad')
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('API key invalid.')).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('API key'), 'good')
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
    expect(seen).toEqual([null, 'bad', 'good'])
    expect(JSON.stringify({ ...window.localStorage })).not.toContain('good')
    expect(JSON.stringify({ ...window.sessionStorage })).not.toContain('good')
  })

  it('shows the no-auth notice when the backend reports auth is off', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) =>
      new Response(String(input).endsWith('/api/health/ready') ? JSON.stringify({ auth: 'none' }) : '{}', { status: 200 })))
    render(<AuthBoundary />)
    expect(await screen.findByRole('status')).toHaveTextContent('No API key configured')
  })

  it('does not attach the key to non-API-v1 requests', async () => {
    const { apiFetch } = await import('./api')
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    setApiKey('k')
    await apiFetch('/api/health/live')
    await apiFetch('/api/v1/registry')
    const keys = fetchMock.mock.calls.map((call) => new Headers((call as unknown as [string, RequestInit | undefined])[1]?.headers).get('X-API-Key'))
    expect(keys).toEqual([null, 'k'])
  })
})
