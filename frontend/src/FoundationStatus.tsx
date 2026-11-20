import { useEffect, useState } from 'react'
import { apiFetch } from './api'

export function FoundationStatus() {
  const [api, setApi] = useState('checking')

  useEffect(() => {
    apiFetch('/api/health/live')
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then(() => setApi('connected'))
      .catch(() => setApi('unavailable'))
  }, [])

  return (
    <main className="foundation">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">V</span><span>VISUAL RAG STUDIO</span></div>
        <div className="project"><strong>Untitled project</strong><span>Foundation setup</span></div>
        <div className="status"><span className={`dot ${api}`} /> API {api}</div>
      </header>
      <section className="welcome">
        <div className="eyebrow">PROJECT FOUNDATION · STORY 1</div>
        <h1>Your RAG workspace<br /><span>starts here.</span></h1>
        <p>The backend and frontend foundation is in place. The visual builder and real RAG execution arrive in the next verified stories.</p>
        <div className="cards">
          <article><span className="number">01</span><h2>Build visually</h2><p>Compose a workflow from compatible, schema-defined components.</p><span className="badge planned">NEXT STORY</span></article>
          <article><span className="number">02</span><h2>Run with evidence</h2><p>Execute configured providers and inspect retrieved source provenance.</p><span className="badge planned">PLANNED</span></article>
          <article><span className="number">03</span><h2>Stay in control</h2><p>Local-first by default, with explicit providers and visible errors.</p><span className="badge ready">FOUNDATION</span></article>
        </div>
        <footer><span>LOCAL DEVELOPMENT MODE</span><span>RAG EXECUTION NOT CONFIGURED</span></footer>
      </section>
    </main>
  )
}
