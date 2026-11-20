import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { FoundationStatus } from './FoundationStatus'

describe('foundation screen', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('shows truthful RAG status and API liveness', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'alive' }) }))
    render(<FoundationStatus />)
    expect(screen.getByText('RAG EXECUTION NOT CONFIGURED')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('API connected')).toBeInTheDocument())
  })

  it('shows an unavailable API state without concealing the failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    render(<FoundationStatus />)
    await waitFor(() => expect(screen.getByText('API unavailable')).toBeInTheDocument())
  })
})
