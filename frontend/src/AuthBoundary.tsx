import { useEffect, useState, type FormEvent } from 'react'
import { apiFetch, onAuthFailure, setApiKey, type AuthFailure } from './api'
import { VisualBuilder } from './VisualBuilder'

const bar = {
  position: 'fixed', left: 12, bottom: 12, zIndex: 50, display: 'flex', gap: 8, alignItems: 'center',
  padding: '8px 11px', borderRadius: 7, fontSize: 11, maxWidth: 560,
} as const

/** Asks for the API key when the backend requires one; the key lives in memory only. */
export function AuthBoundary() {
  const [failure, setFailure] = useState<AuthFailure | null>(null)
  const [draft, setDraft] = useState('')
  const [epoch, setEpoch] = useState(0)
  const [authMode, setAuthMode] = useState<'required' | 'none' | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => onAuthFailure(setFailure), [])
  useEffect(() => {
    let cancelled = false
    void apiFetch('/api/health/ready')
      .then(async (res) => (res.ok ? res.json() : null))
      .then((body) => { if (!cancelled && (body?.auth === 'required' || body?.auth === 'none')) setAuthMode(body.auth) })
      .catch(() => undefined)
    return () => { cancelled = true }
  }, [])

  function retry(event: FormEvent) {
    event.preventDefault()
    setApiKey(draft)
    setDraft('')
    setFailure(null)
    setEpoch((value) => value + 1) // remount so every mount-time request is retried with the key
  }

  return <>
    <VisualBuilder key={epoch} />
    {failure && <form role="alert" onSubmit={retry} style={{ ...bar, background: '#302020', border: '1px solid #b05d55', color: '#f0b0a8' }}>
      <strong>{failure === 'api_key_invalid' ? 'API key invalid.' : 'API key required.'}</strong>
      <input aria-label="API key" type="password" autoComplete="off" spellCheck={false} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Enter API key" />
      <button type="submit" disabled={!draft.trim()}>Retry</button>
    </form>}
    {!failure && authMode === 'none' && !dismissed && <div role="status" style={{ ...bar, background: '#29251d', border: '1px solid #8f7840', color: '#d0bd91' }}>
      <span>No API key configured: this backend is open and must stay on loopback.</span>
      <button type="button" aria-label="Dismiss notice" onClick={() => setDismissed(true)}>×</button>
    </div>}
  </>
}
