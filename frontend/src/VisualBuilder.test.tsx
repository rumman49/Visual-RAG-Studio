import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Edge } from '@xyflow/react'
import { apiErrorDetail, arePortTypesCompatible, serializeWorkflow, VisualBuilder, type FlowNode } from './VisualBuilder'

const registry = {
  catalog: [
    { id: 'traditional-rag', name: 'Traditional RAG', category: 'Foundational', kind: 'architecture', description: 'Baseline retrieval workflow', implementation_status: 'planned', template_id: 'traditional-rag', known_limitations: [] },
    { id: 'agentic-rag', name: 'Agentic RAG', category: 'Agentic and corrective', kind: 'architecture', description: 'Bounded agent graph', implementation_status: 'planned', template_id: 'agentic-rag', known_limitations: [] },
    { id: 'start-from-blank-canvas', name: 'Start from Blank Canvas', category: 'Foundational', kind: 'blank_canvas', description: 'Empty graph', implementation_status: 'planned', known_limitations: [] },
  ],
  templates: ['traditional-rag', 'agentic-rag'].map((id) => ({ id, architecture_id: id, name: id === 'agentic-rag' ? 'Agentic RAG' : 'Traditional RAG', implementation_status: 'planned', graph: null, known_limitations: ['No runnable graph'] })),
  node_types: [
    { node_type_id: 'document.source', display_name: 'Document Source', category: 'Ingestion', description: 'Select source documents.', implementation_status: 'planned', inputs: [], outputs: [{ id: 'documents', display_name: 'Documents', data_type: 'document_set' }], config_schema: { properties: {}, allow_additional_properties: false }, config_defaults: {}, dependencies: [], provider_requirements: [], known_limitations: ['Schema only'] },
    { node_type_id: 'chunk.split', display_name: 'Chunk Documents', category: 'Ingestion', description: 'Split parsed content.', implementation_status: 'planned', inputs: [{ id: 'parsed_documents', display_name: 'Parsed Documents', data_type: 'parsed_documents', required: true }], outputs: [{ id: 'chunks', display_name: 'Chunks', data_type: 'chunks' }], config_schema: { properties: { strategy: { value_type: 'string', description: 'Chunking strategy.', required: true, enum: ['recursive', 'fixed'] }, chunk_size: { value_type: 'integer', description: 'Target size.', default: 128, minimum: 1, maximum: 1000 } }, allow_additional_properties: false }, config_defaults: {}, dependencies: [], provider_requirements: [], known_limitations: ['No runtime'] },
    { node_type_id: 'query.input', display_name: 'Query Input', category: 'Query', description: 'Question input.', implementation_status: 'tested', inputs: [], outputs: [{ id: 'query', display_name: 'Query', data_type: 'query' }], config_schema: { properties: {}, allow_additional_properties: false }, config_defaults: {}, dependencies: [], provider_requirements: [], known_limitations: [] },
    { node_type_id: 'embedding.create', display_name: 'Create Embeddings', category: 'Indexing', description: 'Create embeddings.', implementation_status: 'implemented', inputs: [{ id: 'chunks', display_name: 'Chunks', data_type: 'chunks' }], outputs: [{ id: 'embeddings', display_name: 'Embeddings', data_type: 'embeddings' }], config_schema: { properties: { model: { value_type: 'string', description: 'Model.' }, dimension: { value_type: 'integer', description: 'Dimension.', default: 384 } }, allow_additional_properties: false }, config_defaults: {}, dependencies: [], provider_requirements: [], known_limitations: [] },
    { node_type_id: 'generation.answer', display_name: 'Generate Answer', category: 'Generation', description: 'Generate a grounded answer.', implementation_status: 'implemented', inputs: [{ id: 'query', display_name: 'Query', data_type: 'query' }, { id: 'context', display_name: 'Context', data_type: 'context' }], outputs: [{ id: 'answer', display_name: 'Answer', data_type: 'answer' }], config_schema: { properties: { provider: { value_type: 'string', description: 'LLM provider.' }, model: { value_type: 'string', description: 'Model ID.' }, temperature: { value_type: 'number', description: 'Sampling temperature.' } }, allow_additional_properties: false }, config_defaults: {}, dependencies: [], provider_requirements: [], known_limitations: [] },
  ],
  chunking_strategies: [{ id: 'recursive', display_name: 'Recursive', implementation_status: 'planned', known_limitations: ['No runtime'] }],
}
const validReport: { valid: boolean; executable: boolean; issues: { code: string; severity: 'error' | 'warning' | 'info'; message: string; node_id?: string | null; connection_index?: number | null }[]; topological_order: string[] } = { valid: true, executable: false, issues: [], topological_order: [] }

function mockBackend(validation = validReport) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    void init
    const url = String(input)
    if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
    if (url.endsWith('/api/v1/documents')) return new Response('[]', { status: 200 })
    if (url.endsWith('/api/v1/storage/summary')) return new Response(JSON.stringify({ metadata_backend: 'SQLite + SQLAlchemy', metadata_location: 'Configured local database', metadata_document_count: 0, source_files_present: 0, vector_backend: 'Chroma PersistentClient', vector_persistence: 'Configured local directory', vector_store_healthy: true, vector_count: 0, collections: [], checked_at: new Date().toISOString() }), { status: 200 })
    if (url.includes('/api/v1/chunks')) return new Response(JSON.stringify({ total: 0, limit: 20, offset: 0, chunks: [] }), { status: 200 })
    if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'groq', model: 'openai/gpt-oss-20b', configured: false, credential_present: false, groq_api_key_present: false, huggingface_token_present: false, readiness: 'not_tested' }), { status: 200 })
    return new Response(JSON.stringify(validation), { status: 200 })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('visual workflow builder', () => {
  afterEach(() => { cleanup(); window.localStorage.clear(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('accepts compatible port types and rejects incompatible types', () => {
    expect(arePortTypesCompatible('chunks', 'chunks')).toBe(true)
    expect(arePortTypesCompatible('text', 'chunks')).toBe(false)
    expect(arePortTypesCompatible('text', 'chunks', ['any'])).toBe(true)
  })

  it('formats structured API errors as readable text instead of object strings', () => {
    expect(apiErrorDetail({ code: 'workflow_not_executable', message: 'Graph validation failed.', issues: [{ message: 'Generation provider is unavailable.' }] }))
      .toBe('Graph validation failed. Generation provider is unavailable.')
    expect(apiErrorDetail({ code: 'provider_error' })).toBe('{"code":"provider_error"}')
  })

  it('serializes nodes and connections using the backend graph contract', () => {
    const definition = registry.node_types[0]
    const nodes = [{ id: 'source-a', type: 'capability', position: { x: 10, y: 20 }, data: { definition, config: { enabled: true } } }] as unknown as FlowNode[]
    const edges = [{ id: 'edge-a', source: 'source-a', sourceHandle: 'documents', target: 'parser-b', targetHandle: 'documents' }] as Edge[]
    const graph = serializeWorkflow('Draft', null, null, nodes, edges)
    expect(graph.nodes[0]).toEqual({ id: 'source-a', node_type_id: 'document.source', config: { enabled: true }, provider_instance_id: null })
    expect(graph.connections[0]).toEqual({ source_node_id: 'source-a', source_port_id: 'documents', target_node_id: 'parser-b', target_port_id: 'documents', kind: 'data', label: null })
    expect(graph.execution.mode).toBe('deterministic_dag')
  })

  it('preserves template provider bindings in the executable graph payload', () => {
    const definition = registry.node_types[0]
    const nodes = [{ id: 'embedder', type: 'capability', position: { x: 0, y: 0 }, data: { definition, config: {}, provider_instance_id: 'ollama-embedding' } }] as unknown as FlowNode[]
    expect(serializeWorkflow('Traditional RAG', 'traditional-rag', 'traditional-rag', nodes, []).nodes[0].provider_instance_id).toBe('ollama-embedding')
  })

  it('loads nodes from the registry and supports search and category filtering', async () => {
    mockBackend()
    render(<VisualBuilder />)
    expect(await screen.findByRole('button', { name: /Document Source/ })).toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: 'Search nodes' }), { target: { value: 'query.input' } })
    expect(screen.getByRole('button', { name: /Query Input/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Document Source/ })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: 'Search nodes' }), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'Ingestion' }))
    expect(screen.getByRole('button', { name: /Chunk Documents/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Query Input/ })).not.toBeInTheDocument()
  })

  it('opens with the registered Traditional RAG graph on the canvas', async () => {
    const traditional = registry.templates[0] as { graph: any }
    const originalGraph = traditional.graph
    traditional.graph = {
      schema_version: 1, id: 'traditional-rag-draft', name: 'Traditional RAG', architecture_id: 'traditional-rag', template_id: 'traditional-rag',
      nodes: [{ id: 'source', node_type_id: 'document.source', config: {}, provider_instance_id: null }],
      connections: [], execution: { mode: 'deterministic_dag' },
    }
    try {
      mockBackend()
      const { container } = render(<VisualBuilder />)
      await waitFor(() => expect(screen.getByRole('textbox', { name: 'Workflow title' })).toHaveValue('Traditional RAG'))
      expect(container.querySelector('.canvas-count')).toHaveTextContent('1 nodes')
      expect(screen.getByRole('button', { name: /WORKFLOW TEMPLATE/i })).toHaveTextContent('Traditional RAG')
    } finally { traditional.graph = originalGraph }
  })

  it('defaults to Groq and ignores the previous browser-saved provider/model', async () => {
    window.localStorage.setItem('vrs-generation-config', JSON.stringify({ provider: 'ollama', model: 'qwen3-embedding' }))
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
      if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'groq', model: 'openai/gpt-oss-20b', configured: false, credential_present: false, groq_api_key_present: false, huggingface_token_present: false, readiness: 'not_tested' }), { status: 200 })
      if (url.endsWith('/api/v1/documents')) return new Response('[]', { status: 200 })
      if (url.endsWith('/api/v1/providers/health')) return new Response(JSON.stringify({ provider: 'sentence_transformers', online: true, embedding_model_available: true }), { status: 200 })
      return new Response(JSON.stringify(validReport), { status: 200 })
    }))
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Configure Generate Answer' }))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Generation provider' })).toHaveValue('groq'))
    expect(screen.getByRole('textbox', { name: 'Generation model' })).toHaveValue('openai/gpt-oss-20b')
    expect(window.localStorage.getItem('vrs-generation-config')).toBeNull()
  })

  it('shows storage, document, and chunk explorer views from backend endpoints', async () => {
    mockBackend()
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Storage' }))
    expect(await screen.findByText('SQLite + SQLAlchemy')).toBeInTheDocument()
    expect(screen.getByText('Chroma PersistentClient')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Documents' }))
    expect(await screen.findByRole('heading', { name: 'Documents' })).toBeInTheDocument()
    expect(screen.getByText('No persisted documents found.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Chunks' }))
    expect(await screen.findByRole('heading', { name: 'Chunk explorer' })).toBeInTheDocument()
    expect(screen.getByText(/0 stored chunks/)).toBeInTheDocument()
  })

  it('explains that the playground needs an indexed document instead of hiding the question form', async () => {
    mockBackend()
    const { container } = render(<VisualBuilder />)
    expect(await screen.findByText(/There are no indexed sources available on this backend right now/)).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Ask a question' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
    expect(container.querySelector('.playground input[type="file"]')).toBeNull()
    expect(within(container.querySelector('.playground') as HTMLElement).queryByText(/embeddings/)).not.toBeInTheDocument()
    const resizeHandle = screen.getByRole('separator', { name: /Resize playground/ })
    const initialHeight = Number(resizeHandle.getAttribute('aria-valuenow'))
    fireEvent.keyDown(resizeHandle, { key: 'ArrowUp' })
    expect(resizeHandle).toHaveAttribute('aria-valuenow', String(initialHeight + 24))
    fireEvent.click(screen.getByRole('button', { name: 'Open document source' }))
    expect(await screen.findByLabelText('Upload document')).toBeInTheDocument()
    expect(screen.getByText('DOCUMENT SOURCE')).toBeInTheDocument()
  })

  it('sends a Groq key to the backend once without saving it in browser storage', async () => {
    const calls: { url: string; init?: RequestInit }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); calls.push({ url, init })
      if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
      if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'groq', model: 'openai/gpt-oss-20b', configured: false, credential_present: false, groq_api_key_present: false, huggingface_token_present: false, readiness: 'not_tested' }), { status: 200 })
      if (url.endsWith('/api/v1/documents')) return new Response('[]', { status: 200 })
      if (url.endsWith('/api/v1/providers/health')) return new Response(JSON.stringify({ provider: 'sentence_transformers', online: true, embedding_model_available: true }), { status: 200 })
      if (url.endsWith('/api/v1/providers/credentials/groq')) return new Response(JSON.stringify({ configured: true, source: 'session' }), { status: 200 })
      return new Response(JSON.stringify(validReport), { status: 200 })
    }))
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Configure Generate Answer' }))
    const keyInput = await screen.findByLabelText('Groq API key')
    fireEvent.change(keyInput, { target: { value: 'gsk-test-secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save key' }))
    await waitFor(() => expect(screen.getByLabelText('Groq API key')).toHaveValue(''))
    const call = calls.find((entry) => entry.url.endsWith('/api/v1/providers/credentials/groq'))
    expect(JSON.parse(String(call?.init?.body))).toEqual({ api_key: 'gsk-test-secret' })
    expect(window.localStorage.getItem('vrs-generation-config')).toBeNull()
  })

  it('clears a key after an outdated backend rejects the credential route', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
      if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'groq', model: 'openai/gpt-oss-20b', configured: false, credential_present: false, groq_api_key_present: false, huggingface_token_present: false, readiness: 'not_tested' }), { status: 200 })
      if (url.endsWith('/api/v1/documents')) return new Response('[]', { status: 200 })
      if (url.endsWith('/api/v1/providers/health')) return new Response('{}', { status: 200 })
      if (url.endsWith('/api/v1/providers/credentials/groq')) return new Response('Not Found', { status: 404 })
      return new Response(JSON.stringify(validReport), { status: 200 })
    }))
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Configure Generate Answer' }))
    const keyInput = await screen.findByLabelText('Groq API key')
    fireEvent.change(keyInput, { target: { value: 'gsk-temporary' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save key' }))
    expect(await screen.findByText(/Restart it from this workspace on port 8001/)).toBeInTheDocument()
    expect(screen.getByLabelText('Groq API key')).toHaveValue('')
  })

  it('explains the workspace API port when the registry returns 404', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => String(input).endsWith('/api/v1/registry')
      ? new Response('not found', { status: 404 })
      : new Response('[]', { status: 200 })))
    render(<VisualBuilder />)
    expect(await screen.findByText(/Start this workspace's backend on port 8001/)).toBeInTheDocument()
    expect(screen.getByText("Can't load node definitions.")).toBeInTheDocument()
  })

  it('collapses and restores the node palette and inspector', async () => {
    mockBackend()
    const { container } = render(<VisualBuilder />)
    await screen.findByRole('button', { name: /Document Source/ })
    fireEvent.click(screen.getByRole('button', { name: 'Hide node palette' }))
    fireEvent.click(screen.getByRole('button', { name: 'Hide inspector' }))
    expect(container.querySelector('.builder-layout')).toHaveClass('palette-collapsed', 'inspector-collapsed')
    expect(screen.queryByRole('heading', { name: 'Node palette' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Inspector' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Show node palette' }))
    fireEvent.click(screen.getByRole('button', { name: 'Show inspector' }))
    expect(screen.getByRole('heading', { name: 'Node palette' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Inspector' })).toBeInTheDocument()
  })

  it('tests Hugging Face using only the backend token and selected model', async () => {
    const calls: { url: string; init?: RequestInit }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); calls.push({ url, init })
      if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
      if (url.endsWith('/api/v1/documents')) return new Response('[]', { status: 200 })
      if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'ollama', model: 'local-chat', configured: true, credential_present: true, huggingface_token_present: true, readiness: 'not_tested' }), { status: 200 })
      if (url.endsWith('/api/v1/providers/health')) return new Response(JSON.stringify({ provider: 'ollama', configured: true, online: true, embedding_model: 'nomic', generation_model: 'local-chat', embedding_model_available: true, generation_model_available: true }), { status: 200 })
      if (url.endsWith('/api/v1/providers/test')) return new Response(JSON.stringify({ provider: 'huggingface', model: 'org/model', ready: true, message: 'A real provider completion succeeded.' }), { status: 200 })
      return new Response(JSON.stringify(validReport), { status: 200 })
    }))
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Configure Generate Answer' }))
    fireEvent.change(await screen.findByRole('combobox', { name: 'Generation provider' }), { target: { value: 'huggingface' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Generation model' }), { target: { value: 'org/model' } })
    fireEvent.click(screen.getByRole('button', { name: 'Test provider' }))
    expect(await screen.findByText(/A real provider completion succeeded/)).toBeInTheDocument()
    const call = calls.find((entry) => entry.url.endsWith('/api/v1/providers/test'))
    expect(JSON.parse(String(call?.init?.body))).toEqual({ provider: 'huggingface', model: 'org/model' })
    expect(JSON.stringify(call?.init?.headers)).not.toMatch(/token|authorization/i)
  })

  it('supports switching to Ollama without requiring an API key and loads Ollama models', async () => {
    const calls: { url: string; init?: RequestInit }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); calls.push({ url, init })
      if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
      if (url.endsWith('/api/v1/documents')) return new Response('[]', { status: 200 })
      if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'groq', model: 'openai/gpt-oss-20b', configured: false, credential_present: false, groq_api_key_present: false, huggingface_token_present: false, readiness: 'not_tested' }), { status: 200 })
      if (url.endsWith('/api/v1/providers/health')) return new Response(JSON.stringify({ provider: 'sentence_transformers', online: true, embedding_model_available: true }), { status: 200 })
      if (url.endsWith('/api/v1/providers/ollama/models')) return new Response(JSON.stringify(['qwen3:1.7b', 'llama3.2:latest']), { status: 200 })
      if (url.endsWith('/api/v1/providers/test')) return new Response(JSON.stringify({ provider: 'ollama', model: 'qwen3:1.7b', ready: true, message: 'Ollama local generation succeeded.' }), { status: 200 })
      return new Response(JSON.stringify(validReport), { status: 200 })
    }))
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Configure Generate Answer' }))
    fireEvent.change(await screen.findByRole('combobox', { name: 'Generation provider' }), { target: { value: 'ollama' } })
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Generation model' })).toHaveValue('qwen3:1.7b'))
    expect(screen.queryByLabelText(/API key|API token/i)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Test provider' })).not.toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Test provider' }))
    expect(await screen.findByText(/Ollama local generation succeeded/)).toBeInTheDocument()
    const testCall = calls.find((entry) => entry.url.endsWith('/api/v1/providers/test'))
    expect(JSON.parse(String(testCall?.init?.body))).toEqual({ provider: 'ollama', model: 'qwen3:1.7b' })
  })

  it('supports configuring and saving an OpenAI API key in backend memory', async () => {
    const calls: { url: string; init?: RequestInit }[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); calls.push({ url, init })
      if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
      if (url.endsWith('/api/v1/documents')) return new Response('[]', { status: 200 })
      if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'groq', model: 'openai/gpt-oss-20b', configured: false, credential_present: false, groq_api_key_present: false, huggingface_token_present: false, readiness: 'not_tested' }), { status: 200 })
      if (url.endsWith('/api/v1/providers/health')) return new Response(JSON.stringify({ provider: 'sentence_transformers', online: true, embedding_model_available: true }), { status: 200 })
      if (url.endsWith('/api/v1/providers/credentials/openai')) {
        if (init?.method === 'POST') return new Response(JSON.stringify({ configured: true, source: 'session' }), { status: 200 })
        return new Response(JSON.stringify({ configured: false, source: null }), { status: 200 })
      }
      return new Response(JSON.stringify(validReport), { status: 200 })
    }))
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Configure Generate Answer' }))
    fireEvent.change(await screen.findByRole('combobox', { name: 'Generation provider' }), { target: { value: 'openai' } })
    const keyInput = await screen.findByLabelText('OpenAI API key')
    expect(keyInput).toBeInTheDocument()
    fireEvent.change(keyInput, { target: { value: 'sk-test-openai-secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save key' }))
    await waitFor(() => expect(screen.getByLabelText('OpenAI API key')).toHaveValue(''))
    const postCall = calls.find((c) => c.url.endsWith('/api/v1/providers/credentials/openai') && c.init?.method === 'POST')
    expect(JSON.parse(String(postCall?.init?.body))).toEqual({ api_key: 'sk-test-openai-secret' })
  })

  it('supports local extractive generation without external consent or credentials', async () => {
    mockBackend()
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Configure Generate Answer' }))
    fireEvent.change(await screen.findByRole('combobox', { name: 'Generation provider' }), { target: { value: 'extractive' } })
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Generation model' })).toHaveValue('extractive'))
    expect(screen.queryByLabelText(/API key|API token/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/Allow .* to answer/i)).not.toBeInTheDocument()
  })

  it('adds a registry node and serializes its configuration to backend validation', async () => {
    const fetchMock = mockBackend()
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: /Chunk Documents/ }))
    expect(await screen.findByText('Inspector')).toBeInTheDocument()
    await waitFor(() => {
      const request = fetchMock.mock.calls.find((call) => String(call[0]).endsWith('/api/v1/workflows/validate'))
      expect(request).toBeTruthy()
      const body = JSON.parse(String(request?.[1]?.body))
      expect(body.graph.nodes).toHaveLength(1)
      expect(body.graph.nodes[0]).toMatchObject({ node_type_id: 'chunk.split', config: { chunk_size: 128 } })
      expect(body.require_executable).toBe(false)
    }, { timeout: 2500 })
  })

  it('confirms clearing a populated draft back to the blank canvas', async () => {
    mockBackend()
    vi.stubGlobal('confirm', vi.fn(() => true))
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: /Document Source/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear canvas' }))
    expect(window.confirm).toHaveBeenCalledOnce()
    expect(await screen.findByText('Start building your workflow')).toBeInTheDocument()
  })

  it('edits schema fields and preserves unrelated node settings', async () => {
    const fetchMock = mockBackend()
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: /Chunk Documents/ }))
    const inspector = screen.getAllByText('Inspector')[0].closest('.inspector')
    expect(inspector).not.toBeNull()
    const strategy = within(inspector as HTMLElement).getByLabelText(/strategy/i)
    fireEvent.change(strategy, { target: { value: 'fixed' } })
    const size = within(inspector as HTMLElement).getByLabelText(/chunk size/i)
    fireEvent.change(size, { target: { value: '256' } })
    await waitFor(() => {
      const requests = fetchMock.mock.calls.filter((call) => String(call[0]).endsWith('/api/v1/workflows/validate'))
      const latest = JSON.parse(String(requests.at(-1)?.[1]?.body))
      expect(latest.graph.nodes[0].config).toEqual({ strategy: 'fixed', chunk_size: 256 })
    }, { timeout: 2500 })
  })

  it('shows planned template status and does not enable execution', async () => {
    mockBackend()
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: /WORKFLOW TEMPLATE/i }))
    fireEvent.click(screen.getByRole('button', { name: /Agentic RAG/ }))
    expect(await screen.findByText('Agentic RAG is planned')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Run workflow/ })).toBeDisabled()
    expect(screen.getByText('Traditional RAG Playground')).toBeInTheDocument()
  })

  it('shows chunking strategy metadata without fabricating a recommendation', async () => {
    mockBackend()
    render(<VisualBuilder />)
    const chunking = await screen.findByText('Document-aware chunking')
    fireEvent.click(chunking)
    expect(await screen.findByText('Recursive')).toBeInTheDocument()
    expect(screen.getByText(/Recommendations are based on extracted document structure/)).toBeInTheDocument()
  })

  it('renders backend issues and distinguishes validation API failures', async () => {
    const issue = { code: 'node_not_executable', severity: 'warning' as const, message: 'Planned node cannot execute.', node_id: 'node-a' }
    mockBackend({ valid: true, executable: false, issues: [issue], topological_order: ['node-a'] })
    render(<VisualBuilder />)
    expect(await screen.findByText('Planned node cannot execute.')).toBeInTheDocument()

    cleanup()
    const failedFetch = vi.fn(async (input: RequestInfo | URL) => String(input).endsWith('/api/v1/registry')
      ? new Response(JSON.stringify(registry), { status: 200 }) : new Response('failure', { status: 503 }))
    vi.stubGlobal('fetch', failedFetch)
    render(<VisualBuilder />)
    expect(await screen.findByText('Validation unavailable')).toBeInTheDocument()
    expect(screen.getAllByText(/Validation API returned 503/).length).toBeGreaterThan(0)
  })

  it('supports conversational playground chat and workflow saving', async () => {
    mockBackend({ valid: true, executable: true, issues: [], topological_order: [] })
    render(<VisualBuilder />)

    expect(await screen.findByRole('button', { name: /Save/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Workflows/ })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Ask a question' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Ask' })).toBeInTheDocument()
  })

  it('supports multi-file upload and filters duplicate provider fields from inspector configuration', async () => {
    mockBackend()
    render(<VisualBuilder />)

    // Document source supports multiple files
    fireEvent.click(await screen.findByRole('button', { name: /Document Source/ }))
    const uploadInput = await screen.findByLabelText('Upload document')
    expect(uploadInput).toHaveAttribute('multiple')

    // Generate answer filters provider & model from CONFIGURATION section
    fireEvent.click(screen.getByRole('button', { name: /Generate Answer/ }))
    expect(await screen.findByRole('combobox', { name: 'Generation provider' })).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Generation model' })).toBeInTheDocument()

    const inspector = screen.getAllByText('Inspector')[0].closest('.inspector')
    expect(inspector).not.toBeNull()
    const configSection = within(inspector as HTMLElement).getByText(/CONFIGURATION/).closest('section')
    expect(configSection).not.toBeNull()
    // temperature is in CONFIGURATION, but duplicate provider and model are excluded
    expect(within(configSection as HTMLElement).getByLabelText(/temperature/i)).toBeInTheDocument()
    expect(within(configSection as HTMLElement).queryByLabelText(/^provider$/i)).toBeNull()
    expect(within(configSection as HTMLElement).queryByLabelText(/^model$/i)).toBeNull()
  })

  it('supports deleting documents and clearing chunks from explorer views', async () => {
    const mockDoc = {
      document_id: 'doc-delete-test',
      filename: 'sample_guide.md',
      file_type: 'md',
      status: 'ready',
      chunk_count: 5,
      recommendation: null,
      selection: null,
      profile: null,
    }
    vi.stubGlobal('confirm', () => true)
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = init?.method || 'GET'
      if (url.endsWith('/api/v1/registry')) return new Response(JSON.stringify(registry), { status: 200 })
      if (url.endsWith('/api/v1/providers/config')) return new Response(JSON.stringify({ provider: 'groq', model: 'llama-3.1-8b-instant', configured: true, credential_present: true, groq_api_key_present: true, huggingface_token_present: false, readiness: 'not_tested' }), { status: 200 })
      if (url.endsWith('/api/v1/documents') && method === 'GET') return new Response(JSON.stringify([mockDoc]), { status: 200 })
      if (url.endsWith('/api/v1/documents/doc-delete-test') && method === 'DELETE') {
        return new Response(JSON.stringify({ deleted: true, document_id: 'doc-delete-test', chunks_removed: 5 }), { status: 200 })
      }
      if (url.endsWith('/api/v1/providers/health')) return new Response(JSON.stringify({ provider: 'sentence_transformers', online: true, embedding_model_available: true }), { status: 200 })
      return new Response(JSON.stringify(validReport), { status: 200 })
    }))

    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: 'Documents' }))
    expect(await screen.findByText('sample_guide.md')).toBeInTheDocument()

    fireEvent.click(screen.getByText('sample_guide.md'))
    const deleteBtn = await screen.findByRole('button', { name: /Delete document/i })
    expect(deleteBtn).toBeInTheDocument()

    fireEvent.click(deleteBtn)
    await waitFor(() => {
      expect(screen.getByText(/Deleted document "sample_guide.md"/i)).toBeInTheDocument()
    })
  })

  it('renders curated embedding models with multilingual options and auto-calibrates dimensions', async () => {
    mockBackend()
    render(<VisualBuilder />)
    fireEvent.click(await screen.findByRole('button', { name: /Create Embeddings/ }))
    const inspector = screen.getAllByText('Inspector')[0].closest('.inspector')
    expect(inspector).not.toBeNull()
    const modelSelect = within(inspector as HTMLElement).getByLabelText(/Model \(Curated Free Models\)/i) as HTMLSelectElement
    expect(modelSelect).toBeInTheDocument()
    expect(modelSelect.querySelector('optgroup[label*="Multilingual Free Models"]')).not.toBeNull()
    expect(modelSelect.querySelector('optgroup[label*="English / High-Performance"]')).not.toBeNull()
    expect(modelSelect.querySelector('option[value="BAAI/bge-m3"]')).not.toBeNull()

    // Select BAAI/bge-m3 and check that dimension auto-calibrates to 1024
    fireEvent.change(modelSelect, { target: { value: 'BAAI/bge-m3' } })
    const dimensionInput = within(inspector as HTMLElement).getByLabelText(/dimension/i)
    expect(dimensionInput).toHaveValue(1024)
  })
})


