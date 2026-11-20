import { createContext, useContext, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { marked } from 'marked'
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  PanOnScrollMode,
  Position,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  reconnectEdge,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import './builder.css'
import { apiFetch } from './api'
import { Activity, AlertCircle, Check, ChevronDown, CircleHelp, Clock, Code2, Database, FileText, FolderOpen, GitBranch, Layers3, ListFilter, Maximize2, MessageSquare, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, Play, Plus, RotateCcw, Save, Search, Settings2, ShieldCheck, Trash2, Workflow, X } from 'lucide-react'

export interface EmbeddingModelOption {
  id: string
  label: string
  category: 'Multilingual' | 'English' | 'Cloud / Ollama'
  dimension: number
  description: string
  tag: string
  recommended?: boolean
}

export const EMBEDDING_MODELS: EmbeddingModelOption[] = [
  // Multilingual Models
  {
    id: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    label: 'paraphrase-multilingual-MiniLM-L12-v2',
    category: 'Multilingual',
    dimension: 384,
    tag: '50+ Languages · Default',
    description: 'Fast, lightweight multilingual embedding (118M params, 384 dims)',
    recommended: true,
  },
  {
    id: 'BAAI/bge-m3',
    label: 'bge-m3',
    category: 'Multilingual',
    dimension: 1024,
    tag: '100+ Languages · SOTA',
    description: 'State-of-the-art multilingual embedding with dense & multi-vector retrieval (1024 dims)',
  },
  {
    id: 'intfloat/multilingual-e5-small',
    label: 'multilingual-e5-small',
    category: 'Multilingual',
    dimension: 384,
    tag: '100 Languages · Fast',
    description: 'Fast multilingual embedding for low-latency retrieval (384 dims)',
  },
  {
    id: 'intfloat/multilingual-e5-base',
    label: 'multilingual-e5-base',
    category: 'Multilingual',
    dimension: 768,
    tag: '100 Languages · Balanced',
    description: 'Balanced multilingual embedding quality and speed (768 dims)',
  },
  {
    id: 'intfloat/multilingual-e5-large',
    label: 'multilingual-e5-large',
    category: 'Multilingual',
    dimension: 1024,
    tag: '100 Languages · High Quality',
    description: 'High-accuracy multilingual representation (1024 dims)',
  },

  // English / High-Performance Open Weights
  {
    id: 'sentence-transformers/all-MiniLM-L6-v2',
    label: 'all-MiniLM-L6-v2',
    category: 'English',
    dimension: 384,
    tag: 'Ultra-fast · CPU Friendly',
    description: 'Very fast English sentence embedding, tiny footprint (61M params, 384 dims)',
  },
  {
    id: 'sentence-transformers/all-mpnet-base-v2',
    label: 'all-mpnet-base-v2',
    category: 'English',
    dimension: 768,
    tag: 'English Standard',
    description: 'Highest overall semantic quality for English dense retrieval (768 dims)',
  },
  {
    id: 'BAAI/bge-small-en-v1.5',
    label: 'bge-small-en-v1.5',
    category: 'English',
    dimension: 384,
    tag: 'MTEB Leader · Fast',
    description: 'Top-ranked compact English embedding on MTEB benchmarks (384 dims)',
  },
  {
    id: 'BAAI/bge-base-en-v1.5',
    label: 'bge-base-en-v1.5',
    category: 'English',
    dimension: 768,
    tag: 'MTEB Leader · High Accuracy',
    description: 'High performance open-source English embedding (768 dims)',
  },
  {
    id: 'nomic-ai/nomic-embed-text-v1.5',
    label: 'nomic-embed-text-v1.5',
    category: 'English',
    dimension: 768,
    tag: '8K Context Window',
    description: 'Long-context embedding model with dynamic Matryoshka dimensions (768 dims)',
  },
  {
    id: 'mixedbread-ai/mxbai-embed-large',
    label: 'mxbai-embed-large',
    category: 'English',
    dimension: 1024,
    tag: 'Large Open Weights',
    description: 'Large state-of-the-art open weights embedding model (1024 dims)',
  },

  // Cloud Free-tier & Ollama
  {
    id: 'models/text-embedding-004',
    label: 'models/text-embedding-004',
    category: 'Cloud / Ollama',
    dimension: 768,
    tag: 'Google Gemini · Free Tier',
    description: 'Google Cloud Gemini text embedding with free quota tier (768 dims)',
  },
  {
    id: 'nomic-embed-text',
    label: 'nomic-embed-text (Ollama)',
    category: 'Cloud / Ollama',
    dimension: 768,
    tag: 'Ollama Local',
    description: 'Popular Ollama local embedding runner (768 dims)',
  },
  {
    id: 'all-minilm',
    label: 'all-minilm (Ollama)',
    category: 'Cloud / Ollama',
    dimension: 384,
    tag: 'Ollama Local Fast',
    description: 'Fast local embedding runner in Ollama (384 dims)',
  },
  {
    id: 'bge-m3',
    label: 'bge-m3 (Ollama)',
    category: 'Cloud / Ollama',
    dimension: 1024,
    tag: 'Ollama Multilingual',
    description: 'Multilingual SOTA embedding runner in Ollama (1024 dims)',
  },
  {
    id: 'text-embedding-3-small',
    label: 'text-embedding-3-small (OpenAI)',
    category: 'Cloud / Ollama',
    dimension: 1536,
    tag: 'OpenAI Small',
    description: 'OpenAI 3rd generation small embedding API (1536 dims)',
  },
  {
    id: 'text-embedding-3-large',
    label: 'text-embedding-3-large (OpenAI)',
    category: 'Cloud / Ollama',
    dimension: 3072,
    tag: 'OpenAI High-Dim',
    description: 'OpenAI 3rd generation high dimension embedding API (3072 dims)',
  },
]

export const EMBEDDING_MODEL_DIMENSIONS: Record<string, number> = Object.fromEntries(
  EMBEDDING_MODELS.map((m) => [m.id, m.dimension])
)

const VECTOR_BACKEND_DEFAULTS: Record<string, { url: string; api_key?: string }> = {
  chroma: { url: 'http://localhost:8000', api_key: '' },
  memory: { url: 'local://memory', api_key: '' },
  qdrant: { url: 'http://localhost:6333', api_key: '' },
  pinecone: { url: 'https://api.pinecone.io', api_key: '' },
  mongodb: { url: 'mongodb://localhost:27017', api_key: '' },
}

type Status = 'planned' | 'experimental' | 'implemented' | 'tested'
type Port = { id: string; display_name: string; data_type: string; required?: boolean; accepts?: string[]; description?: string }
type ConfigField = { value_type: string; description: string; required?: boolean; default?: unknown; enum?: unknown[] | null; minimum?: number | null; maximum?: number | null; min_length?: number | null; max_length?: number | null; pattern?: string | null }
type NodeDefinition = {
  node_type_id: string; display_name: string; category: string; description: string; implementation_status: Status
  inputs: Port[]; outputs: Port[]; config_schema: { properties: Record<string, ConfigField>; allow_additional_properties: boolean }
  config_defaults: Record<string, unknown>; dependencies: string[]; provider_requirements: string[]; known_limitations: string[]
}
type CatalogEntry = { id: string; name: string; category: string; kind: 'architecture' | 'blank_canvas'; description: string; implementation_status: Status; template_id?: string | null; known_limitations: string[] }
type Template = { id: string; architecture_id: string; name: string; implementation_status: Status; graph: WorkflowGraph | null; known_limitations: string[] }
type Registry = { catalog: CatalogEntry[]; templates: Template[]; node_types: NodeDefinition[]; chunking_strategies: { id: string; display_name: string; implementation_status: Status; known_limitations: string[] }[] }
type WorkflowGraph = { schema_version: number; id: string; name: string; architecture_id: string | null; template_id: string | null; nodes: { id: string; node_type_id: string; config: Record<string, unknown>; provider_instance_id: string | null }[]; connections: { source_node_id: string; source_port_id: string; target_node_id: string; target_port_id: string; kind: string; label: string | null }[]; execution: { mode: string } }
type ValidationIssue = { code: string; severity: 'error' | 'warning' | 'info'; message: string; node_id?: string | null; connection_index?: number | null }
type ValidationReport = { valid: boolean; executable: boolean; workflow_state?: string; issues: ValidationIssue[]; topological_order: string[] }
type UploadedDocument = { document_id: string; filename: string; file_type: string; status: string; created_at?: string; profile: any; recommendation: any; plan?: any; selection: any; chunk_count: number; error?: string | null }
type ProviderStatus = { provider: string; configured: boolean; online: boolean; embedding_model?: string | null; generation_model?: string | null; embedding_model_available?: boolean | null; generation_model_available?: boolean | null; error?: string | null; weights_load_on_first_use?: boolean }
type GenerationProvider = 'groq' | 'openai' | 'gemini' | 'anthropic' | 'ollama' | 'huggingface' | 'extractive'
type GenerationProviderConfig = { provider: GenerationProvider; model: string | null; configured: boolean; credential_present: boolean; huggingface_token_present: boolean; groq_api_key_present: boolean; openai_api_key_present?: boolean; gemini_api_key_present?: boolean; anthropic_api_key_present?: boolean; execution_progress_supported?: boolean; readiness: 'not_tested'; provider_policy?: string | null }

const PROVIDER_INFO: Record<GenerationProvider, {
  name: string
  requiresKey: boolean
  keyLabel: string
  placeholder: string
  defaultModel: string
  models: string[]
}> = {
  groq: {
    name: 'Groq',
    requiresKey: true,
    keyLabel: 'Groq API key',
    placeholder: 'gsk_…',
    defaultModel: 'openai/gpt-oss-20b',
    models: [
      'openai/gpt-oss-20b',
      'openai/gpt-oss-120b',
      'qwen/qwen3.8-27b',
      'llama-3.3-70b-versatile',
      'llama-3.1-8b-instant',
      'mixtral-8x7b-32768',
      'gemma2-9b-it',
    ],
  },
  openai: {
    name: 'OpenAI',
    requiresKey: true,
    keyLabel: 'OpenAI API key',
    placeholder: 'sk-…',
    defaultModel: 'gpt-4o-mini',
    models: [
      'gpt-4o-mini',
      'gpt-4o',
      'gpt-3.5-turbo',
      'o1-mini',
    ],
  },
  gemini: {
    name: 'Google Gemini',
    requiresKey: true,
    keyLabel: 'Gemini API key',
    placeholder: 'AIzaSy…',
    defaultModel: 'gemini-2.0-flash',
    models: [
      'gemini-2.0-flash',
      'gemini-1.5-flash',
      'gemini-1.5-pro',
    ],
  },
  anthropic: {
    name: 'Anthropic Claude',
    requiresKey: true,
    keyLabel: 'Anthropic API key',
    placeholder: 'sk-ant-…',
    defaultModel: 'claude-3-5-sonnet-20241022',
    models: [
      'claude-3-5-sonnet-20241022',
      'claude-3-5-haiku-20241022',
      'claude-3-opus-20240229',
    ],
  },
  ollama: {
    name: 'Ollama',
    requiresKey: false,
    keyLabel: '',
    placeholder: '',
    defaultModel: 'qwen3:1.7b',
    models: [
      'qwen3:1.7b',
      'llama3.2',
      'mistral',
      'deepseek-r1',
    ],
  },
  huggingface: {
    name: 'Hugging Face Inference Providers',
    requiresKey: true,
    keyLabel: 'Hugging Face API token',
    placeholder: 'hf_…',
    defaultModel: 'meta-llama/Llama-3.1-8B-Instruct',
    models: [
      'meta-llama/Llama-3.1-8B-Instruct',
      'Qwen/Qwen2.5-7B-Instruct',
      'mistralai/Mistral-7B-Instruct-v0.3',
    ],
  },
  extractive: {
    name: 'Extractive (Local Offline)',
    requiresKey: false,
    keyLabel: '',
    placeholder: '',
    defaultModel: 'extractive',
    models: ['extractive'],
  },
}

function isExternalProvider(provider: GenerationProvider): boolean {
  return provider !== 'ollama' && provider !== 'extractive'
}
type TraceChunk = { chunk_id: string; document_id: string; document_name: string; page_number?: number | null; section_path: string[]; excerpt: string; text: string; distance?: number | null; distance_metric?: 'cosine' | null; rank?: number | null; retrieval_strategy?: string | null; score?: number | null; sparse_score?: number | null; rrf_score?: number | null; weighted_score?: number | null; dense_rank?: number | null; sparse_rank?: number | null }
type QueryResult = { question: string; answer: string; abstained: boolean; status: string; execution_id?: string; generation_provider?: string; generation_model: string; embedding_model: string; collection?: string; provider_output?: string | null; generation_error?: string | null; context_preview?: string | null; context_truncated?: boolean; execution_started_at?: string | null; execution_completed_at?: string | null; citations: TraceChunk[]; retrieved_chunks: TraceChunk[] }
type ChatMessage = { id: string; role: 'user' | 'assistant'; content: string; status?: string; citations?: TraceChunk[]; retrieved_chunks?: TraceChunk[]; collection?: string; generation_provider?: string; generation_model?: string; embedding_model?: string; execution_id?: string; execution_started_at?: string | null; execution_completed_at?: string | null; provider_output?: string | null; context_preview?: string | null; generation_error?: string | null; trace?: any[]; step_count?: number }
type SavedWorkflowItem = { id: string; name: string; created_at: string; updated_at: string; graph: WorkflowGraph }
type StorageSummary = { metadata_backend: string; metadata_location: string; metadata_document_count: number; source_files_present: number; vector_backend: string; vector_persistence: string; vector_store_healthy: boolean; vector_count?: number | null; collections: { name: string; count: number }[]; checked_at: string; last_successful_check?: string | null; error?: string | null }
type ChunkRecord = { chunk_id: string; document_id: string; document_name: string; ordinal: number; strategy_id?: string | null; page_number?: number | null; section_path: string[]; source_start?: number | null; source_end?: number | null; text: string; embedding_available: boolean; embedding_dimension?: number | null }
type ChunkPage = { total: number; limit: number; offset: number; chunks: ChunkRecord[] }
type AppView = 'workflow' | 'documents' | 'chunks' | 'storage'
type ExecutionNodeStatus = 'idle' | 'queued' | 'running' | 'completed' | 'failed' | 'skipped'
type ExecutionProgress = { execution_id: string; status: 'running' | 'succeeded' | 'failed'; node_statuses: Record<string, Exclude<ExecutionNodeStatus, 'idle'>>; error?: string | null }
export type FlowNodeData = { definition: NodeDefinition; config: Record<string, unknown>; provider_instance_id?: string | null; execution_status?: ExecutionNodeStatus }
export type FlowNode = Node<FlowNodeData>
type ApiState = 'loading' | 'ready' | 'error'

const priority = ['traditional-rag', 'hybrid-rag', 'reranking-rag', 'multilingual-rag', 'agentic-rag', 'multimodal-rag', 'start-from-blank-canvas']
const icons = [FileText, Layers3, Database, Workflow, GitBranch, ShieldCheck]
const emptyReport: ValidationReport = { valid: false, executable: false, issues: [], topological_order: [] }

function FormattedAssistantMessage({ content, citations }: { content: string; citations?: TraceChunk[] }) {
  const citationMap = new Map<string, number>()
  ;(citations || []).forEach((c, idx) => {
    if (c.chunk_id) {
      citationMap.set(c.chunk_id.toLowerCase(), idx + 1)
      const parts = c.chunk_id.split(':')
      if (parts.length > 1) {
        citationMap.set(parts[1].toLowerCase(), idx + 1)
      }
    }
  })

  let processed = content.replace(/\[(?:chunk:)?([a-f0-9\-]{8,}(?::[a-f0-9\-]{8,})?)\]/gi, (match, rawId) => {
    const key = rawId.toLowerCase()
    const num = citationMap.get(key) || citationMap.get((rawId.split(':')[1] || '').toLowerCase())
    if (num) {
      const cite = (citations || [])[num - 1]
      const docLabel = cite ? `${cite.document_name} · p.${cite.page_number ?? 1}` : `Source ${num}`
      return ` <span class="citation-badge" title="${docLabel}">[${num}]</span> `
    }
    return match
  })

  processed = processed.replace(/\[(\d+)\]/g, (match, numStr) => {
    const num = parseInt(numStr, 10)
    if (citations && num > 0 && num <= citations.length) {
      const cite = citations[num - 1]
      const docLabel = `${cite.document_name} · p.${cite.page_number ?? 1}`
      return ` <span class="citation-badge" title="${docLabel}">[${num}]</span> `
    }
    return match
  })

  const html = marked.parse(processed, { breaks: true, gfm: true }) as string

  return (
    <div
      className="assistant-body formatted-markdown"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

function makeId() { return globalThis.crypto?.randomUUID?.() ?? `node-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` }
async function watchExecution(executionId: string, signal: AbortSignal, onProgress: (progress: ExecutionProgress) => void) {
  let consecutiveNotFound = 0
  while (!signal.aborted) {
    try {
      const response = await apiFetch(`/api/v1/executions/${executionId}`, { signal })
      if (response.ok) {
        consecutiveNotFound = 0
        const progress = await response.json() as ExecutionProgress
        if (progress.node_statuses && progress.status) {
          onProgress(progress)
          if (progress.status !== 'running') return progress
        }
      } else if (response.status === 404) {
        consecutiveNotFound++
        if (consecutiveNotFound > 15) return null
      } else return null
    } catch { if (signal.aborted) return null }
    await new Promise((resolve) => window.setTimeout(resolve, 200))
  }
  return null
}
export function apiErrorDetail(detail: unknown): string | undefined {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((issue: any) => {
    const message = issue.msg ?? issue.message ?? 'invalid value'
    const field = Array.isArray(issue.loc) ? issue.loc.slice(-1)[0] : null
    return field ? `${field}: ${message}` : String(message)
  }).join(' · ')
  if (detail && typeof detail === 'object') {
    const payload = detail as Record<string, unknown>
    const message = typeof payload.message === 'string' ? payload.message : undefined
    const issues = Array.isArray(payload.issues) ? apiErrorDetail(payload.issues) : undefined
    if (message && issues) return `${message} ${issues}`
    if (message) return message
    if (typeof payload.error === 'string') return payload.error
    try { return JSON.stringify(payload) } catch { return 'The server returned an unreadable error.' }
  }
  return undefined
}
function defaultsFor(definition: NodeDefinition) {
  const config: Record<string, unknown> = { ...definition.config_defaults }
  for (const [key, field] of Object.entries(definition.config_schema.properties ?? {})) if (field.default !== undefined && config[key] === undefined) config[key] = field.default
  if (definition.node_type_id === 'embedding.create') {
    if (!config.model || typeof config.model !== 'string' || !config.model.trim()) {
      config.model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    }
    if (!config.dimension) {
      config.dimension = EMBEDDING_MODEL_DIMENSIONS[config.model as string] ?? 384
    }
  }
  return config
}
export const DEFAULT_PROVIDER_INSTANCE_BY_TYPE: Record<string, string> = {
  'embedding.create': 'embedding-provider',
  'vector.index': 'chroma-local',
  'generation.answer': 'generation-provider',
  'retrieval.rerank': 'flash-reranker',
  'agent.orchestrate': 'agent-orchestrator',
}

export function arePortTypesCompatible(outputType: string, inputType: string, acceptedTypes: string[] = []) {
  const accepts = acceptedTypes.length ? acceptedTypes : [inputType]
  return outputType === 'any' || accepts.includes('any') || accepts.includes(outputType)
}

export function serializeWorkflow(name: string, architectureId: string | null, templateId: string | null, nodes: FlowNode[], edges: Edge[]): WorkflowGraph {
  return {
    schema_version: 1,
    id: 'draft-workflow',
    name,
    architecture_id: architectureId,
    template_id: templateId,
    nodes: nodes.map((node) => {
      const config = { ...node.data.config }
      if (node.data.definition.node_type_id === 'embedding.create') {
        if (!config.model || typeof config.model !== 'string' || !config.model.trim()) {
          config.model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        }
        if (!config.dimension) {
          config.dimension = EMBEDDING_MODEL_DIMENSIONS[config.model as string] ?? 384
        }
      }
      const provider_instance_id = node.data.provider_instance_id || DEFAULT_PROVIDER_INSTANCE_BY_TYPE[node.data.definition.node_type_id] || null
      return { id: node.id, node_type_id: node.data.definition.node_type_id, config, provider_instance_id }
    }),
    connections: edges.map((edge) => ({ source_node_id: edge.source, source_port_id: edge.sourceHandle ?? '', target_node_id: edge.target, target_port_id: edge.targetHandle ?? '', kind: 'data', label: null })),
    execution: { mode: 'deterministic_dag' },
  }
}

function instantiateTemplate(template: Template, registry: Registry) {
  if (!template.graph) return { nodes: [] as FlowNode[], edges: [] as Edge[] }
  const definitions = new Map(registry.node_types.map((definition) => [definition.node_type_id, definition]))
  const positionsByTemplate: Record<string, Record<string, { x: number; y: number }>> = {
    'traditional-rag': {
      source: { x: 60, y: 50 },
      parser: { x: 380, y: 50 },
      chunker: { x: 700, y: 50 },
      embedder: { x: 1020, y: 50 },
      index: { x: 1340, y: 50 },
      query: { x: 60, y: 320 },
      retriever: { x: 480, y: 320 },
      context: { x: 900, y: 320 },
      answer: { x: 1320, y: 320 },
    },
    'hybrid-rag': {
      source: { x: 60, y: 50 },
      parser: { x: 380, y: 50 },
      chunker: { x: 700, y: 50 },
      embedder: { x: 1020, y: 50 },
      index: { x: 1340, y: 50 },
      query: { x: 60, y: 350 },
      dense_retriever: { x: 400, y: 220 },
      sparse_retriever: { x: 400, y: 480 },
      fuse: { x: 740, y: 350 },
      context: { x: 1060, y: 350 },
      answer: { x: 1380, y: 350 },
    },
    'reranking-rag': {
      source: { x: 60, y: 50 },
      parser: { x: 380, y: 50 },
      chunker: { x: 700, y: 50 },
      embedder: { x: 1020, y: 50 },
      index: { x: 1340, y: 50 },
      query: { x: 60, y: 320 },
      retriever: { x: 380, y: 320 },
      reranker: { x: 700, y: 320 },
      context: { x: 1020, y: 320 },
      answer: { x: 1340, y: 320 },
    },
    'multilingual-rag': {
      source: { x: 60, y: 50 },
      parser: { x: 380, y: 50 },
      chunker: { x: 700, y: 50 },
      embedder: { x: 1020, y: 50 },
      index: { x: 1340, y: 50 },
      query: { x: 60, y: 320 },
      retriever: { x: 480, y: 320 },
      context: { x: 900, y: 320 },
      answer: { x: 1320, y: 320 },
    },
  }
  const templatePositions = positionsByTemplate[template.id] || positionsByTemplate['traditional-rag']
  const nodes: FlowNode[] = template.graph.nodes.flatMap((item, index) => {
    const definition = definitions.get(item.node_type_id)
    return definition ? [{ id: item.id, type: 'capability', position: templatePositions[item.id] ?? { x: 120 + index * 260, y: 130 }, data: { definition, config: { ...defaultsFor(definition), ...item.config }, provider_instance_id: item.provider_instance_id }, className: '' }] : []
  })
  const edges: Edge[] = template.graph.connections.map((edge) => ({ id: makeId(), source: edge.source_node_id, sourceHandle: edge.source_port_id, target: edge.target_node_id, targetHandle: edge.target_port_id, type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed, color: '#9dbf7b' } }))
  return { nodes, edges }
}

function compatible(connection: { source: string | null; target: string | null; sourceHandle?: string | null; targetHandle?: string | null }, nodes: FlowNode[]) {
  const source = nodes.find((node) => node.id === connection.source)?.data.definition
  const target = nodes.find((node) => node.id === connection.target)?.data.definition
  const output = source?.outputs.find((port) => port.id === connection.sourceHandle)
  const input = target?.inputs.find((port) => port.id === connection.targetHandle)
  if (!output || !input) return false
  return arePortTypesCompatible(output.data_type, input.data_type, input.accepts)
}

interface ConnectionDragContextType {
  dragInfo: {
    nodeId: string | null
    handleId: string | null
    handleType: 'source' | 'target' | null
  } | null
  nodes: FlowNode[]
}

const ConnectionDragContext = createContext<ConnectionDragContextType>({ dragInfo: null, nodes: [] })

function CapabilityNode({ id, data, selected }: NodeProps<FlowNode>) {
  const { dragInfo, nodes } = useContext(ConnectionDragContext)
  const { definition, config } = data
  const executionStatus = data.execution_status ?? 'idle'
  const canvasStatus = executionStatus === 'idle'
    ? definition.implementation_status === 'implemented' || definition.implementation_status === 'tested' ? 'READY' : 'PLANNED'
    : ({ queued: 'WAITING', running: 'RUNNING', completed: 'DONE', failed: 'FAILED', skipped: 'SKIPPED' } as const)[executionStatus]
  const portRows = Array.from({ length: Math.max(definition.inputs.length, definition.outputs.length) }, (_, index) => ({
    input: definition.inputs[index], output: definition.outputs[index],
  }))

  const sourceNode = dragInfo?.nodeId ? nodes.find((n) => n.id === dragInfo.nodeId) : null
  const sourcePort = (sourceNode && dragInfo?.handleType === 'source')
    ? sourceNode.data.definition.outputs.find((p) => p.id === dragInfo.handleId)
    : null
  const targetPort = (sourceNode && dragInfo?.handleType === 'target')
    ? sourceNode.data.definition.inputs.find((p) => p.id === dragInfo.handleId)
    : null

  return <article className={`cap-node ${selected ? 'selected' : ''} execution-${executionStatus}`}>
    <div className="cap-node-head"><span className="node-icon"><Workflow size={15} /></span><div><strong>{definition.display_name}</strong><code>{definition.node_type_id}</code></div></div>
    {definition.node_type_id === 'agent.orchestrate' && (
      <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', margin: '4px 0 6px 0' }}>
        <span style={{ background: '#3b0764', border: '1px solid #7e22ce', color: '#e9d5ff', fontSize: '9px', padding: '1px 5px', borderRadius: '3px' }}>
          {String(config.mode || 'bounded_loop')}
        </span>
        <span style={{ background: '#2e1065', border: '1px solid #6b21a8', color: '#d8b4fe', fontSize: '9px', padding: '1px 5px', borderRadius: '3px' }}>
          max {String(config.max_steps || 5)} steps
        </span>
        <span style={{ background: '#1e1b4b', border: '1px solid #4338ca', color: '#c7d2fe', fontSize: '9px', padding: '1px 5px', borderRadius: '3px' }}>
          {String(config.approval_policy || 'on_uncertainty')}
        </span>
      </div>
    )}
    <p>{definition.description}</p>
    {portRows.length > 0 && <div className="cap-node-ports">{portRows.map(({ input, output }, index) => {
      let inputClass = ''
      if (dragInfo && input) {
        if (dragInfo.handleType === 'source') {
          if (dragInfo.nodeId === id) {
            inputClass = 'port-incompatible'
          } else if (sourcePort) {
            const isMatch = arePortTypesCompatible(sourcePort.data_type, input.data_type, input.accepts)
            inputClass = isMatch ? 'port-valid-target' : 'port-incompatible'
          }
        } else {
          inputClass = 'port-incompatible'
        }
      }

      let outputClass = ''
      if (dragInfo && output) {
        if (dragInfo.handleType === 'target') {
          if (dragInfo.nodeId === id) {
            outputClass = 'port-incompatible'
          } else if (targetPort) {
            const isMatch = arePortTypesCompatible(output.data_type, targetPort.data_type, targetPort.accepts)
            outputClass = isMatch ? 'port-valid-target' : 'port-incompatible'
          }
        } else {
          outputClass = (dragInfo.handleType === 'source' && dragInfo.nodeId === id && dragInfo.handleId === output.id)
            ? 'port-active-source'
            : ''
        }
      }

      return (
        <div className="cap-node-port-row" key={`${input?.id ?? 'no-input'}-${output?.id ?? 'no-output'}-${index}`}>
          <span className={`port-name input-name ${input ? '' : 'empty'}`} title={input ? `${input.display_name}: ${input.data_type}` : undefined}>{input?.display_name}</span>
          <span className={`port-name output-name ${output ? '' : 'empty'}`} title={output ? `${output.display_name}: ${output.data_type}` : undefined}>{output?.display_name}</span>
          {input && <Handle type="target" position={Position.Left} id={input.id} className={inputClass} title={`${input.display_name}: ${input.data_type}`} />}
          {output && <Handle type="source" position={Position.Right} id={output.id} className={outputClass} title={`${output.display_name}: ${output.data_type}`} />}
        </div>
      )
    })}</div>}
    <div className="cap-node-foot"><span className={`status-pill ${executionStatus === 'idle' ? definition.implementation_status : `execution-${executionStatus}`}`} title={`Component implementation: ${definition.implementation_status}; run state: ${executionStatus}`}>{canvasStatus}</span><span>{Object.keys(config).length} settings</span></div>
  </article>
}

const nodeTypes = { capability: CapabilityNode }

function FieldEditor({ name, field, defaultValue, value, error, onChange }: { name: string; field: ConfigField; defaultValue?: unknown; value: unknown; error?: string; onChange: (value: unknown) => void }) {
  const [jsonDraft, setJsonDraft] = useState(value === undefined ? '' : JSON.stringify(value, null, 2))
  const [jsonError, setJsonError] = useState('')
  useEffect(() => { setJsonDraft(value === undefined ? '' : JSON.stringify(value, null, 2)); setJsonError('') }, [value])
  const label = name.replaceAll('_', ' ')
  const fieldId = `config-${name}`
  const constraints = [
    defaultValue !== undefined ? `Default: ${JSON.stringify(defaultValue)}` : '',
    field.minimum != null || field.maximum != null ? `Range: ${field.minimum ?? '−∞'} to ${field.maximum ?? '∞'}` : '',
    field.min_length != null || field.max_length != null ? `Length: ${field.min_length ?? 0} to ${field.max_length ?? '∞'}` : '',
    field.pattern ? `Pattern: ${field.pattern}` : '',
  ].filter(Boolean)
  const effectiveValue = (value !== undefined && value !== null && value !== '') ? value : (defaultValue ?? '')

  const isMultilineText = field.value_type === 'string' && (name === 'goal' || name === 'instructions' || name === 'prompt' || name === 'system_prompt')
  const defaultRows = name === 'instructions' ? 6 : name === 'goal' ? 3 : 4
  const minHeight = name === 'instructions' ? '140px' : name === 'goal' ? '75px' : '65px'
  const placeholder = name === 'goal'
    ? 'Define primary objective (e.g. Answer queries factually based on retrieved evidence, citing statutory provisions and section numbers)...'
    : name === 'instructions'
    ? 'Provide explicit directives, behavior guidelines, boundaries, or fallback instructions:\n- Strict grounding policy: never fabricate sections or laws\n- If evidence lacks exact statutory definitions, abstain clearly\n- Format key dates and statutory titles in bold'
    : undefined

  const control = field.enum?.length ? <select id={fieldId} value={String(effectiveValue)} onChange={(event) => onChange(field.enum?.find((item) => String(item) === event.target.value))}><option value="">Select…</option>{field.enum.map((item) => <option key={String(item)} value={String(item)}>{String(item)}</option>)}</select>
    : field.value_type === 'boolean' ? <label className="toggle"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} /><span>{value ? 'Enabled' : 'Disabled'}</span></label>
      : ['array', 'object'].includes(field.value_type) ? <textarea id={fieldId} aria-label={label} value={jsonDraft} onChange={(event) => { setJsonDraft(event.target.value); try { onChange(JSON.parse(event.target.value)); setJsonError('') } catch { setJsonError('Enter valid JSON to save this value.') } }} rows={4} />
        : isMultilineText ? (
          <div className="multiline-editor-container" style={{ width: '100%' }}>
            <textarea
              id={fieldId}
              aria-label={label}
              value={value === undefined || value === null ? '' : String(value)}
              placeholder={placeholder}
              rows={defaultRows}
              style={{
                width: '100%',
                minHeight,
                resize: 'vertical',
                padding: '8px 10px',
                borderRadius: '4px',
                fontFamily: 'inherit',
                fontSize: '11px',
                lineHeight: '1.45',
                boxSizing: 'border-box',
                background: '#0d1310',
                border: error || jsonError ? '1px solid #ef4444' : '1px solid #2d4233',
                color: '#d4ebd4',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
              onChange={(event) => onChange(event.target.value)}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#64748b', marginTop: '2px', padding: '0 2px' }}>
              <span>{String(value || '').length} characters · resizable</span>
              <span>newlines preserved</span>
            </div>
          </div>
        )
        : <input id={fieldId} aria-label={label} type={['number', 'integer'].includes(field.value_type) ? 'number' : 'text'} value={value === undefined || value === null ? '' : String(value)} min={field.minimum ?? undefined} max={field.maximum ?? undefined} minLength={field.min_length ?? undefined} maxLength={field.max_length ?? undefined} pattern={field.pattern ?? undefined} onChange={(event) => onChange(field.value_type === 'number' ? (event.target.value === '' ? undefined : Number(event.target.value)) : field.value_type === 'integer' ? (event.target.value === '' ? undefined : Number.parseInt(event.target.value, 10)) : event.target.value)} />
  return <div className={`schema-field ${error || jsonError ? 'has-error' : ''}`}><label htmlFor={fieldId}>{label}{field.required && <em> required</em>}</label>{control}<small>{field.description}{constraints.length ? ` · ${constraints.join(' · ')}` : ''}</small>{(error || jsonError) && <span className="field-error">{jsonError || error}</span>}</div>
}

function VisualBuilderContent() {
  const { screenToFlowPosition, fitView } = useReactFlow()
  const [registry, setRegistry] = useState<Registry | null>(null)
  const [apiState, setApiState] = useState<ApiState>('loading')
  const [apiError, setApiError] = useState('')
  const [paletteCollapsed, setPaletteCollapsed] = useState(false)
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false)
  const [nodes, setNodes] = useState<FlowNode[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('All')
  const [report, setReport] = useState<ValidationReport>(emptyReport)
  const [validationState, setValidationState] = useState<'idle' | 'checking' | 'done' | 'error'>('idle')
  const [validationError, setValidationError] = useState('')
  const [activeTemplateId, setActiveTemplateId] = useState('start-from-blank-canvas')
  const [workflowName, setWorkflowName] = useState('Untitled workflow')
  const [templateSearch, setTemplateSearch] = useState('')
  const [templateMenu, setTemplateMenu] = useState(false)
  const [providerStatus, setProviderStatus] = useState<ProviderStatus | null>(null)
  const [, setGenerationConfig] = useState<GenerationProviderConfig | null>(null)
  const [generationProvider, setGenerationProvider] = useState<GenerationProvider>('groq')
  const [generationModel, setGenerationModel] = useState('')
  const [keyConfiguredMap, setKeyConfiguredMap] = useState<Record<string, boolean>>({})
  const [providerApiKey, setProviderApiKey] = useState('')
  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [groqModels, setGroqModels] = useState<string[]>([])
  const [openaiModels, setOpenaiModels] = useState<string[]>([])
  const [credentialBusy, setCredentialBusy] = useState(false)
  const [credentialMessage, setCredentialMessage] = useState('')
  const [providerTest, setProviderTest] = useState<{ ready: boolean; message: string } | null>(null)

  const isKeyConfigured = !PROVIDER_INFO[generationProvider]?.requiresKey || Boolean(keyConfiguredMap[generationProvider])
  const [document, setDocument] = useState<UploadedDocument | null>(null)
  const [documents, setDocuments] = useState<UploadedDocument[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState('')
  const [chunkSize, setChunkSize] = useState(1000)
  const [chunkOverlap, setChunkOverlap] = useState(120)
  const [ragQuestion, setRagQuestion] = useState('')
  const [, setRagResult] = useState<QueryResult | null>(null)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [savedWorkflows, setSavedWorkflows] = useState<SavedWorkflowItem[]>([])
  const [sidebarMode, setSidebarMode] = useState<'palette' | 'history'>('palette')
  const [savedPipelinesExpanded, setSavedPipelinesExpanded] = useState(true)
  const [saveStatusMessage, setSaveStatusMessage] = useState('')
  const [ragBusy, setRagBusy] = useState(false)
  const [ragError, setRagError] = useState('')
  const [connectionDrag, setConnectionDrag] = useState<{ nodeId: string | null; handleId: string | null; handleType: 'source' | 'target' | null } | null>(null)
  const [executionStatuses, setExecutionStatuses] = useState<Record<string, ExecutionNodeStatus>>({})
  const [executionMode, setExecutionMode] = useState<'index' | 'query' | null>(null)
  const [executionOutcome, setExecutionOutcome] = useState<'running' | 'succeeded' | 'failed' | null>(null)
  const [executionProgressSupported, setExecutionProgressSupported] = useState(false)
  const [activeView, setActiveView] = useState<AppView>('workflow')
  const [authorizeExternal, setAuthorizeExternal] = useState(false)
  const [queryScope, setQueryScope] = useState<'active' | 'all'>('active')
  const [playgroundHeight, setPlaygroundHeight] = useState(300)
  const [storageSummary, setStorageSummary] = useState<StorageSummary | null>(null)
  const [storageError, setStorageError] = useState('')
  const [storageMessage, setStorageMessage] = useState('')
  const [chunkPage, setChunkPage] = useState<ChunkPage | null>(null)
  const [chunkOffset, setChunkOffset] = useState(0)
  const [chunkFilter, setChunkFilter] = useState('')
  const [selectedExplorerDoc, setSelectedExplorerDoc] = useState<string | null>(null)
  const [documentPreview, setDocumentPreview] = useState<{ filename: string; text: string; truncated: boolean } | null>(null)
  const [documentSearch, setDocumentSearch] = useState('')
  const validationSequence = useRef(0)
  const workspaceRef = useRef<HTMLElement | null>(null)
  const playgroundDrag = useRef<{ pointerId: number; y: number; height: number } | null>(null)

  function selectOrAddNode(nodeTypeId: string) {
    const existing = nodes.find((node) => node.data.definition.node_type_id === nodeTypeId)
    if (existing) { setSelectedNodeId(existing.id); return }
    const definition = registry?.node_types.find((nodeType) => nodeType.node_type_id === nodeTypeId)
    if (definition) addNode(definition)
  }

  function resizePlayground(clientY: number) {
    const drag = playgroundDrag.current
    if (!drag) return
    setPlaygroundHeight(Math.max(250, Math.min(maxPlaygroundHeight(), drag.height + drag.y - clientY)))
  }

  function maxPlaygroundHeight() {
    const workspace = workspaceRef.current
    if (!workspace || !workspace.clientHeight) return 500
    const children = Array.from(workspace.children) as HTMLElement[]
    const canvas = children.find((child) => child.classList.contains('flow-canvas'))
    const divider = children.find((child) => child.classList.contains('playground-resizer'))
    const playground = children.find((child) => child.classList.contains('playground'))
    const fixedHeight = children
      .filter((child) => child !== canvas && child !== divider && child !== playground)
      .reduce((total, child) => total + child.getBoundingClientRect().height, 0)
    const gap = Number.parseFloat(window.getComputedStyle(workspace).rowGap) || 0
    const totalGaps = Math.max(0, children.length - 1) * gap
    const dividerHeight = divider?.getBoundingClientRect().height ?? 20
    const minimumCanvasHeight = 150
    return Math.max(250, Math.floor(workspace.clientHeight - fixedHeight - dividerHeight - totalGaps - minimumCanvasHeight))
  }

  useEffect(() => {
    let cancelled = false
    void apiFetch('/api/v1/workflows').then(async (res) => {
      if (res.ok && !cancelled) setSavedWorkflows(await res.json())
    }).catch(() => {})
    apiFetch('/api/v1/registry').then(async (response) => { if (!response.ok) throw new Error(response.status === 404 ? 'Registry route returned 404. The frontend now expects this workspace backend on port 8001.' : `Registry request failed (${response.status})`); return response.json() as Promise<Registry> })
      .then((result) => {
        if (!cancelled) {
          setRegistry(result); setApiState('ready')
          const baseline = result.templates.find((template) => template.id === 'traditional-rag')
          if (baseline?.graph) {
            const instantiated = instantiateTemplate(baseline, result)
            setNodes(instantiated.nodes); setEdges(instantiated.edges)
            setActiveTemplateId('traditional-rag'); setWorkflowName('Traditional RAG')
            setTimeout(() => { try { fitView({ padding: 0.15, duration: 300 }) } catch {} }, 100)
          }
        }
      })
      .catch((error: Error) => { if (!cancelled) { setApiError(error.message || 'Could not load the backend registry.'); setApiState('error') } })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (activeView === 'documents') {
      void apiFetch('/api/v1/documents').then(async (response) => {
        if (!response.ok) throw new Error(`Document list failed (${response.status}).`)
        setDocuments(await response.json() as UploadedDocument[])
      }).catch((error) => setStorageError(error instanceof Error ? error.message : 'Document list unavailable.'))
    }
    if (activeView === 'storage') {
      setStorageError('')
      void apiFetch('/api/v1/storage/summary').then(async (response) => {
        if (!response.ok) throw new Error(`Storage check failed (${response.status}).`)
        setStorageSummary(await response.json() as StorageSummary)
      }).catch((error) => setStorageError(error instanceof Error ? error.message : 'Storage check unavailable.'))
    }
    if (activeView === 'chunks') {
      const params = new URLSearchParams({ limit: '20', offset: String(chunkOffset) })
      if (selectedExplorerDoc) params.set('document_id', selectedExplorerDoc)
      void apiFetch(`/api/v1/chunks?${params}`).then(async (response) => {
        if (!response.ok) throw new Error(`Chunk listing failed (${response.status}).`)
        setChunkPage(await response.json() as ChunkPage)
      }).catch((error) => setStorageError(error instanceof Error ? error.message : 'Chunk listing unavailable.'))
    }
  }, [activeView, chunkOffset, selectedExplorerDoc])

  useEffect(() => {
    window.localStorage.removeItem('vrs-generation-config')
    void apiFetch('/api/v1/providers/health').then(async (response) => { if (!response.ok) throw new Error('Provider status unavailable'); setProviderStatus(await response.json()) }).catch(() => setProviderStatus(null))
    void apiFetch('/api/v1/documents').then(async (response) => {
      if (!response.ok) throw new Error('Document list unavailable')
      const rows = await response.json()
      if (Array.isArray(rows)) {
        setDocuments(rows)
        const readyDoc = rows.find((item: UploadedDocument) => item.status === 'ready')
        if (readyDoc) {
          setDocument(readyDoc)
          if (readyDoc.recommendation) {
            setSelectedStrategy(readyDoc.recommendation.recommended_strategy_id)
            setChunkSize(readyDoc.recommendation.proposed_settings.chunk_size ?? 1000)
            setChunkOverlap(readyDoc.recommendation.proposed_settings.overlap ?? 120)
          }
        } else if (rows.length > 0) {
          setDocument(rows[0])
        }
      }
    }).catch(() => undefined)
    void apiFetch('/api/v1/providers/config').then(async (response) => {
      if (!response.ok) throw new Error('Generation provider configuration unavailable')
      const config = await response.json() as GenerationProviderConfig
      setGenerationConfig(config)
      setKeyConfiguredMap({
        groq: Boolean(config.groq_api_key_present),
        openai: Boolean(config.openai_api_key_present),
        gemini: Boolean(config.gemini_api_key_present),
        anthropic: Boolean(config.anthropic_api_key_present),
        huggingface: Boolean(config.huggingface_token_present),
      })
      setExecutionProgressSupported(config.execution_progress_supported === true)
      setGenerationProvider(config.provider); setGenerationModel(config.model ?? '')
    }).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (generationProvider === 'ollama') {
      void apiFetch('/api/v1/providers/ollama/models')
        .then((res) => res.ok ? res.json() : [])
        .then((data) => {
          const list = Array.isArray(data) ? data : Array.isArray(data?.models) ? data.models : []
          if (list.length > 0) setOllamaModels(list)
        })
        .catch(() => undefined)
    }
  }, [generationProvider])

  useEffect(() => {
    if (generationProvider === 'groq') {
      void apiFetch('/api/v1/providers/groq/models')
        .then((res) => res.ok ? res.json() : [])
        .then((data) => {
          const list = Array.isArray(data) ? data : Array.isArray(data?.models) ? data.models : []
          if (list.length > 0) setGroqModels(list)
        })
        .catch(() => undefined)
    }
  }, [generationProvider, keyConfiguredMap.groq])

  useEffect(() => {
    if (generationProvider === 'openai') {
      void apiFetch('/api/v1/providers/openai/models')
        .then((res) => res.ok ? res.json() : [])
        .then((data) => {
          const list = Array.isArray(data) ? data : Array.isArray(data?.models) ? data.models : []
          if (list.length > 0) setOpenaiModels(list)
        })
        .catch(() => undefined)
    }
  }, [generationProvider, keyConfiguredMap.openai])

  const allAvailableModels = useMemo(() => {
    if (generationProvider === 'ollama') {
      return Array.from(new Set([...ollamaModels, ...PROVIDER_INFO.ollama.models]))
    }
    if (generationProvider === 'groq' && groqModels.length > 0) {
      return Array.from(new Set([...groqModels, ...PROVIDER_INFO.groq.models]))
    }
    if (generationProvider === 'openai' && openaiModels.length > 0) {
      return Array.from(new Set([...openaiModels, ...PROVIDER_INFO.openai.models]))
    }
    return PROVIDER_INFO[generationProvider]?.models || []
  }, [generationProvider, ollamaModels, groqModels, openaiModels])

  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null
  const categories = useMemo(() => ['All', ...new Set((registry?.node_types ?? []).map((definition) => definition.category))], [registry])
  const filteredNodes = useMemo(() => (registry?.node_types ?? []).filter((definition) => (category === 'All' || definition.category === category) && `${definition.display_name} ${definition.node_type_id} ${definition.description}`.toLowerCase().includes(query.toLowerCase())), [registry, category, query])
  const catalog = registry?.catalog ?? []
  const visibleCatalog = useMemo(() => catalog.filter((entry) => `${entry.name} ${entry.id} ${entry.category}`.toLowerCase().includes(templateSearch.toLowerCase())), [catalog, templateSearch])
  const templateById = new Map((registry?.templates ?? []).map((template) => [template.id, template]))
  const activeCatalogEntry = catalog.find((entry) => entry.id === activeTemplateId)
  const activeTemplate = templateById.get(activeTemplateId)
  const invalidNodeIds = new Set(report.issues.filter((issue) => issue.severity === 'error' && issue.node_id).map((issue) => issue.node_id))
  const graphSignature = JSON.stringify({ nodes: nodes.map(({ id, data }) => ({ id, type: data.definition.node_type_id, config: data.config })), edges: edges.map(({ id, source, sourceHandle, target, targetHandle }) => ({ id, source, sourceHandle, target, targetHandle })) })
  const graphForValidation = useMemo(() => {
    const archId = activeCatalogEntry?.kind === 'architecture'
      ? activeCatalogEntry.id
      : (activeCatalogEntry?.id || 'start-from-blank-canvas')
    const tempId = activeTemplate?.graph
      ? activeTemplate.id
      : (activeCatalogEntry?.id || 'start-from-blank-canvas')
    return serializeWorkflow(
      workflowName,
      archId,
      tempId,
      nodes,
      edges,
    )
  }, [nodes, edges, workflowName, activeCatalogEntry, activeTemplate])

  const runValidation = useCallback(async (graph: WorkflowGraph) => {
    const requestNumber = ++validationSequence.current
    setValidationState('checking'); setValidationError('')
    try {
      const response = await apiFetch('/api/v1/workflows/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ graph, generation_provider: generationProvider, generation_model: generationModel.trim() || undefined, require_executable: false }) })
      if (!response.ok) {
        const payload = await response.json().catch(() => null) as any
        const detail = apiErrorDetail(payload?.detail)
        throw new Error(detail || `Validation API returned ${response.status}. Restart the backend from this workspace if its API is out of date.`)
      }
      const result = await response.json() as ValidationReport
      if (requestNumber === validationSequence.current) { setReport(result); setValidationState('done') }
    } catch (error) {
      if (requestNumber === validationSequence.current) { setValidationError(error instanceof Error ? error.message : 'Validation API unavailable.'); setValidationState('error') }
    }
  }, [generationProvider, generationModel, isKeyConfigured])

  useEffect(() => {
    if (!registry) return
    const timer = window.setTimeout(() => { void runValidation(graphForValidation) }, 550)
    return () => window.clearTimeout(timer)
  }, [registry, graphSignature, runValidation])

  const onNodesChange = useCallback((changes: NodeChange<FlowNode>[]) => {
    const removed = new Set(changes.filter((change) => change.type === 'remove').map((change) => change.id))
    if (removed.size) {
      setEdges((current) => current.filter((edge) => !removed.has(edge.source) && !removed.has(edge.target)))
      setSelectedNodeId((current) => current && removed.has(current) ? null : current)
    }
    setNodes((current) => applyNodeChanges(changes, current))
  }, [])
  const onEdgesChange = useCallback((changes: EdgeChange[]) => setEdges((current) => applyEdgeChanges(changes, current)), [])
  const onConnect = useCallback((connection: Connection) => {
    if (!compatible(connection, nodes)) { setValidationError('Those ports have incompatible data types. Choose a matching output and input.'); setValidationState('error'); return }
    setValidationError(''); setEdges((current) => addEdge({ ...connection, id: makeId(), animated: false, className: '', markerEnd: { type: MarkerType.ArrowClosed, color: '#9dbf7b' } }, current))
  }, [nodes])

  const edgeReconnectSuccessful = useRef(true)

  const onReconnectStart = useCallback(() => {
    edgeReconnectSuccessful.current = false
  }, [])

  const onReconnect = useCallback((oldEdge: Edge, newConnection: Connection) => {
    if (!compatible(newConnection, nodes)) {
      setValidationError('Those ports have incompatible data types. Choose a matching output and input.')
      setValidationState('error')
      return
    }
    setValidationError('')
    edgeReconnectSuccessful.current = true
    setEdges((els) => reconnectEdge(oldEdge, newConnection, els))
  }, [nodes])

  const onReconnectEnd = useCallback((_: MouseEvent | TouchEvent, edge: Edge) => {
    if (!edgeReconnectSuccessful.current) {
      setEdges((eds) => eds.filter((e) => e.id !== edge.id))
    }
    edgeReconnectSuccessful.current = true
  }, [])

  const onEdgeDoubleClick = useCallback((_: React.MouseEvent, edge: Edge) => {
    setEdges((eds) => eds.filter((e) => e.id !== edge.id))
  }, [])

  const onEdgeContextMenu = useCallback((event: React.MouseEvent, edge: Edge) => {
    event.preventDefault()
    setEdges((eds) => eds.filter((e) => e.id !== edge.id))
  }, [])

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    const nodeTypeId = event.dataTransfer.getData('application/reactflow')
    if (!nodeTypeId || !registry) return
    const definition = registry.node_types.find((nt) => nt.node_type_id === nodeTypeId)
    if (!definition) return

    const position = screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    })

    const id = makeId()
    const defaultProviderId = DEFAULT_PROVIDER_INSTANCE_BY_TYPE[definition.node_type_id] ?? null
    setNodes((nds) => [
      ...nds,
      {
        id,
        type: 'capability',
        position,
        data: { definition, config: defaultsFor(definition), provider_instance_id: defaultProviderId },
        className: '',
      },
    ])
    setSelectedNodeId(id)
  }, [registry, screenToFlowPosition])
  const addNode = (definition: NodeDefinition) => {
    const id = makeId()
    const defaultProviderId = DEFAULT_PROVIDER_INSTANCE_BY_TYPE[definition.node_type_id] ?? null
    setNodes((current) => [...current, { id, type: 'capability', position: { x: 180 + (current.length % 3) * 50, y: 110 + current.length * 42 }, data: { definition, config: defaultsFor(definition), provider_instance_id: defaultProviderId }, className: '' }])
    setSelectedNodeId(id)
  }
  const updateConfig = (key: string, value: unknown) => {
    if (!selectedNode) return
    if (selectedNode.data.definition.node_type_id === 'chunk.split') {
      if (key === 'strategy' && typeof value === 'string') setSelectedStrategy(value)
      if (key === 'chunk_size' && typeof value === 'number') setChunkSize(value)
      if (key === 'overlap' && typeof value === 'number') setChunkOverlap(value)
    }
    setNodes((current) => current.map((node) => {
      if (node.id !== selectedNode.id) return node
      const config = { ...node.data.config }
      if (value === undefined) delete config[key]
      else config[key] = value

      // Automatically calibrate vector dimension when embedding model changes
      if (node.data.definition.node_type_id === 'embedding.create' && key === 'model' && typeof value === 'string') {
        const dim = EMBEDDING_MODEL_DIMENSIONS[value]
        if (dim) {
          config.dimension = dim
        }
      }

      // Automatically populate standard host URL & defaults when vector DB changes
      if (node.data.definition.node_type_id === 'vector.index' && key === 'backend' && typeof value === 'string') {
        const defaults = VECTOR_BACKEND_DEFAULTS[value]
        if (defaults) {
          config.url = defaults.url
          config.api_key = defaults.api_key ?? ''
        }
      }

      return { ...node, data: { ...node.data, config } }
    }))
  }
  const updateChunkerConfig = (key: 'strategy' | 'chunk_size' | 'overlap', value: unknown) => {
    setNodes((current) => current.map((node) => node.data.definition.node_type_id === 'chunk.split'
      ? { ...node, data: { ...node.data, config: { ...node.data.config, [key]: value } } }
      : node))
  }
  const clearCanvas = () => {
    if ((nodes.length || edges.length) && !window.confirm('Clear this draft workflow? This cannot be undone.')) return false
    setNodes([]); setEdges([]); setSelectedNodeId(null); setActiveTemplateId('start-from-blank-canvas'); setWorkflowName('Untitled workflow'); setReport(emptyReport); setValidationState('idle')
    return true
  }
  const nodeIssues = report.issues.filter((issue) => issue.node_id === selectedNodeId)
  const fieldErrors = useMemo(() => {
    const result: Record<string, string> = {}
    for (const issue of nodeIssues) {
      const match = issue.message.match(/configuration '([^']+)'/i) ?? issue.message.match(/input '([^']+)'/i)
      if (match) result[match[1]] = issue.message
    }
    return result
  }, [nodeIssues])
  const flowNodes = useMemo(() => nodes.map((node) => ({ ...node, data: { ...node.data, execution_status: executionStatuses[node.id] ?? 'idle' }, className: invalidNodeIds.has(node.id) ? 'invalid' : '' })), [nodes, report, executionStatuses])
  const flowEdges = useMemo(() => edges.map((edge, index) => {
    const failed = report.issues.some((issue) => issue.severity === 'error' && issue.connection_index === index)
    const sourceStatus = executionStatuses[edge.source] ?? 'idle'
    const targetStatus = executionStatuses[edge.target] ?? 'idle'
    const active = sourceStatus === 'completed' && targetStatus === 'running'
    const completed = sourceStatus === 'completed' && targetStatus === 'completed'
    return { ...edge, animated: active, className: failed ? 'invalid-edge' : active ? 'execution-active' : completed ? 'execution-completed' : '' }
  }), [edges, report, executionStatuses])

  function beginTrackedExecution(mode: 'index' | 'query', nodeIds: string[], completedNodeIds: string[]) {
    const executionId = makeId().replaceAll('-', '')
    const initial: Record<string, ExecutionNodeStatus> = Object.fromEntries(nodes.map((node) => [node.id, 'idle' as ExecutionNodeStatus]))
    for (const id of nodeIds) initial[id] = 'queued'
    for (const id of completedNodeIds) initial[id] = 'completed'
    setExecutionStatuses(initial); setExecutionMode(mode); setExecutionOutcome('running')
    const controller = new AbortController()
    const watcher = executionProgressSupported ? watchExecution(executionId, controller.signal, (progress) => {
      setExecutionStatuses((current) => ({ ...current, ...progress.node_statuses }))
    }) : Promise.resolve(null)
    return { executionId, controller, watcher }
  }

  async function settleTrackedExecution(run: ReturnType<typeof beginTrackedExecution> | null): Promise<ExecutionProgress | null> {
    if (!run) return null
    let finalProgress: ExecutionProgress | null = null
    try {
      if (!executionProgressSupported) throw new Error('Execution progress is not supported by the running backend.')
      const response = await apiFetch(`/api/v1/executions/${run.executionId}`)
      if (response.ok) {
        const progress = await response.json() as ExecutionProgress
        finalProgress = progress
        setExecutionStatuses((current) => ({ ...current, ...progress.node_statuses }))
      }
    } catch { /* The main request reports execution errors when progress cannot be read. */ }
    run.controller.abort()
    void run.watcher
    return finalProgress
  }
  const chooseTemplate = (entry: CatalogEntry) => {
    if (entry.kind === 'blank_canvas') { if (!clearCanvas()) return; setActiveTemplateId(entry.id); setWorkflowName('Untitled workflow') }
    else {
      if ((nodes.length || edges.length) && !window.confirm('Replace this canvas with the selected template? Unsaved graph changes will be lost.')) return
      const template = templateById.get(entry.template_id ?? entry.id)
      if (template?.graph && registry) {
        const instantiated = instantiateTemplate(template, registry)
        setNodes(instantiated.nodes)
        setEdges(instantiated.edges)
        setSelectedNodeId(null)
        setTimeout(() => { try { fitView({ padding: 0.15, duration: 300 }) } catch {} }, 100)
      } else { setNodes([]); setEdges([]); setSelectedNodeId(null) }
      setActiveTemplateId(entry.id); setWorkflowName(entry.name); setTemplateMenu(false)
    }
  }

  async function handleDocumentUpload(filesInput?: FileList | File[] | File | null) {
    if (!filesInput) return
    const files = filesInput instanceof FileList || Array.isArray(filesInput)
      ? Array.from(filesInput)
      : [filesInput]
    if (files.length === 0) return

    setRagBusy(true); setRagError(''); setRagResult(null)
    try {
      if (files.length === 1) {
        const form = new FormData(); form.set('file', files[0])
        const response = await apiFetch('/api/v1/documents', { method: 'POST', body: form })
        const payload = await response.json().catch(() => null) as any
        if (!response.ok) throw new Error(payload?.detail?.message ?? payload?.detail ?? `Upload failed (${response.status})`)
        setDocument(payload); setDocuments((current) => [payload, ...current.filter((item) => item.document_id !== payload.document_id)])
        if (payload.recommendation) {
          setSelectedStrategy(payload.recommendation.recommended_strategy_id)
          setChunkSize(payload.recommendation.proposed_settings.chunk_size ?? 1000)
          setChunkOverlap(payload.recommendation.proposed_settings.overlap ?? 120)
        }
      } else {
        const form = new FormData()
        for (const file of files) {
          form.append('files', file)
        }
        const response = await apiFetch('/api/v1/documents/batch', { method: 'POST', body: form })
        const payload = await response.json().catch(() => null) as any
        if (!response.ok) throw new Error(payload?.detail?.message ?? payload?.detail ?? `Batch upload failed (${response.status})`)
        const uploadedList: UploadedDocument[] = Array.isArray(payload) ? payload : []
        if (uploadedList.length > 0) {
          setDocument(uploadedList[0])
          setDocuments((current) => {
            const uploadedIds = new Set(uploadedList.map((d) => d.document_id))
            return [...uploadedList, ...current.filter((d) => !uploadedIds.has(d.document_id))]
          })
          const firstWithRec = uploadedList.find((d) => d.recommendation)
          if (firstWithRec?.recommendation) {
            setSelectedStrategy(firstWithRec.recommendation.recommended_strategy_id)
            setChunkSize(firstWithRec.recommendation.proposed_settings.chunk_size ?? 1000)
            setChunkOverlap(firstWithRec.recommendation.proposed_settings.overlap ?? 120)
          }
        }
      }
    } catch (error) { setRagError(error instanceof Error ? error.message : 'Upload failed.') }
    finally { setRagBusy(false) }
  }

  async function handleDeleteDocument(docId: string, filename: string) {
    if (!window.confirm(`Delete "${filename}" and permanently remove its source file and vector index?`)) return
    try {
      const res = await apiFetch(`/api/v1/documents/${docId}`, { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({})) as any
        throw new Error(err.detail?.message || err.detail || `Delete failed (${res.status})`)
      }
      const data = await res.json() as any
      setDocuments((curr) => {
        const updated = curr.filter((d) => d.document_id !== docId)
        if (document?.document_id === docId) {
          setDocument(updated.length > 0 ? updated[0] : null)
        }
        return updated
      })
      if (selectedExplorerDoc === docId) {
        setSelectedExplorerDoc(null)
        setDocumentPreview(null)
      }
      setStorageMessage(`Deleted document "${filename}" (${data.chunks_removed ?? 0} vectors removed).`)
      setStorageError('')
      if (activeView === 'chunks') setChunkOffset(0)
    } catch (err) {
      setStorageError(err instanceof Error ? err.message : 'Could not delete document.')
    }
  }

  async function handleDeleteDocumentChunks(docId: string, filename: string) {
    if (!window.confirm(`Delete indexed chunks for "${filename}"? The source file will be preserved.`)) return
    try {
      const res = await apiFetch(`/api/v1/documents/${docId}/chunks`, { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({})) as any
        throw new Error(err.detail?.message || err.detail || `Chunk clear failed (${res.status})`)
      }
      const data = await res.json() as any
      setDocuments((curr) => curr.map((d) => d.document_id === docId ? (data.document ?? { ...d, status: 'uploaded', chunk_count: 0 }) : d))
      if (document?.document_id === docId) {
        setDocument((curr) => curr ? { ...curr, status: 'uploaded', chunk_count: 0 } : null)
      }
      setStorageMessage(`Cleared ${data.chunks_removed ?? 0} chunks for "${filename}". Status reset to uploaded.`)
      setStorageError('')
      if (activeView === 'chunks') {
        const params = new URLSearchParams({ limit: '20', offset: '0' })
        if (selectedExplorerDoc) params.set('document_id', selectedExplorerDoc)
        void apiFetch(`/api/v1/chunks?${params}`).then((r) => r.json()).then(setChunkPage).catch(() => undefined)
      }
    } catch (err) {
      setStorageError(err instanceof Error ? err.message : 'Could not clear chunks.')
    }
  }

  async function handleDeleteAllDocuments() {
    if (!window.confirm('Are you sure you want to delete ALL uploaded documents, source files, and indexed vectors? This action cannot be undone.')) return
    try {
      const res = await apiFetch('/api/v1/documents', { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({})) as any
        throw new Error(err.detail?.message || err.detail || `Bulk delete failed (${res.status})`)
      }
      const data = await res.json() as any
      setDocuments([])
      setDocument(null)
      setSelectedExplorerDoc(null)
      setDocumentPreview(null)
      setChunkPage(null)
      setStorageMessage(data.message || `Deleted ${data.deleted_count} documents.`)
      setStorageError('')
      if (activeView === 'storage') {
        void apiFetch('/api/v1/storage/summary').then((r) => r.json()).then(setStorageSummary)
      }
    } catch (err) {
      setStorageError(err instanceof Error ? err.message : 'Could not delete documents.')
    }
  }

  async function handleDeleteSingleChunk(chunkId: string) {
    if (!window.confirm(`Delete chunk ${chunkId.slice(0, 10)}… from the persistent vector store?`)) return
    try {
      const res = await apiFetch(`/api/v1/chunks/${chunkId}`, { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({})) as any
        throw new Error(err.detail?.message || err.detail || `Chunk delete failed (${res.status})`)
      }
      setChunkPage((curr) => {
        if (!curr) return null
        return {
          ...curr,
          total: Math.max(0, curr.total - 1),
          chunks: curr.chunks.filter((c) => c.chunk_id !== chunkId),
        }
      })
      setStorageMessage(`Chunk ${chunkId.slice(0, 10)}… removed from vector store.`)
      setStorageError('')
      void apiFetch('/api/v1/documents').then((r) => r.json()).then(setDocuments).catch(() => undefined)
    } catch (err) {
      setStorageError(err instanceof Error ? err.message : 'Could not delete chunk.')
    }
  }

  async function handleClearAllChunks() {
    const isScoped = Boolean(selectedExplorerDoc)
    const prompt = isScoped
      ? 'Delete all chunks for this selected document? Source file will remain.'
      : 'Delete all indexed chunks across all documents? Source files will remain.'
    if (!window.confirm(prompt)) return
    try {
      const url = isScoped ? `/api/v1/chunks?document_id=${selectedExplorerDoc}` : '/api/v1/chunks'
      const res = await apiFetch(url, { method: 'DELETE' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({})) as any
        throw new Error(err.detail?.message || err.detail || `Clear chunks failed (${res.status})`)
      }
      const data = await res.json() as any
      setChunkOffset(0)
      setChunkPage(null)
      setStorageMessage(data.message || `Cleared chunks (${data.chunks_removed ?? 0} vectors removed).`)
      setStorageError('')
      void apiFetch('/api/v1/documents').then((r) => r.json()).then(setDocuments).catch(() => undefined)
      const params = new URLSearchParams({ limit: '20', offset: '0' })
      if (selectedExplorerDoc) params.set('document_id', selectedExplorerDoc)
      void apiFetch(`/api/v1/chunks?${params}`).then((r) => r.json()).then(setChunkPage).catch(() => undefined)
    } catch (err) {
      setStorageError(err instanceof Error ? err.message : 'Could not clear chunks.')
    }
  }

  async function handleStorageCleanup() {
    if (!window.confirm('Scan for and remove unreferenced / orphaned files in the uploads folder?')) return
    try {
      const res = await apiFetch('/api/v1/storage/cleanup', { method: 'POST' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({})) as any
        throw new Error(err.detail?.message || err.detail || `Cleanup failed (${res.status})`)
      }
      const data = await res.json() as any
      setStorageMessage(`Orphan cleanup finished: ${data.cleaned_count} file(s) removed${data.cleaned_files?.length ? ` (${data.cleaned_files.join(', ')})` : ''}.`)
      setStorageError('')
      void apiFetch('/api/v1/storage/summary').then((r) => r.json()).then(setStorageSummary).catch(() => undefined)
    } catch (err) {
      setStorageError(err instanceof Error ? err.message : 'Storage cleanup failed.')
    }
  }

  function handleProviderChange(newProvider: GenerationProvider) {
    setGenerationProvider(newProvider)
    if (newProvider === 'extractive') {
      setProviderTest({ ready: true, message: 'Local extractive generator is ready.' })
    } else {
      setProviderTest(null)
    }
    setAuthorizeExternal(false)
    setProviderApiKey('')
    setCredentialMessage('')
    const preset = PROVIDER_INFO[newProvider]
    const nextModel = preset?.defaultModel ?? ''
    if (nextModel) {
      setGenerationModel(nextModel)
    }
    updateConfig('provider', newProvider)
    if (nextModel) {
      updateConfig('model', nextModel)
    }
    if (preset?.requiresKey) {
      void apiFetch(`/api/v1/providers/credentials/${newProvider}`)
        .then((res) => res.ok ? res.json() : null)
        .then((data) => {
          if (data && typeof data.configured === 'boolean') {
            setKeyConfiguredMap((prev) => ({ ...prev, [newProvider]: data.configured }))
          }
        })
        .catch(() => undefined)
    }
  }

  async function handleProviderTest() {
    if (!generationModel.trim() || !isKeyConfigured) return
    setRagBusy(true); setProviderTest(null); setRagError('')
    try {
      const response = await apiFetch('/api/v1/providers/test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: generationProvider, model: generationModel.trim() }) })
      const payload = await response.json().catch(() => null) as any
      if (!response.ok) throw new Error(payload.detail?.message ?? payload.detail ?? `Provider test failed (${response.status})`)
      setProviderTest({ ready: true, message: payload.message })
    } catch (error) {
      setProviderTest({ ready: false, message: error instanceof Error ? error.message : 'Provider test failed.' })
    } finally { setRagBusy(false) }
  }

  async function handleSaveKey() {
    if (!providerApiKey.trim()) return
    setCredentialBusy(true); setCredentialMessage('')
    const provider = generationProvider
    const providerName = PROVIDER_INFO[provider]?.name || provider
    try {
      const response = await apiFetch(`/api/v1/providers/credentials/${provider}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ api_key: providerApiKey.trim() }) })
      const payload = await response.json().catch(() => null) as any
      if (!response.ok) {
        setProviderApiKey('')
        const detail = apiErrorDetail(payload?.detail)
        throw new Error(response.status === 404 ? `This backend does not have the ${providerName} key endpoint yet. Restart it from this workspace on port 8001, then enter the key again.` : detail || `Could not save ${providerName} API key (${response.status}).`)
      }
      setProviderApiKey(''); setKeyConfiguredMap((current) => ({ ...current, [provider]: true }))
      if (provider === 'groq') {
        setGenerationConfig((current) => current ? { ...current, groq_api_key_present: true, credential_present: true, configured: Boolean(generationModel.trim()) } : current)
      }
      setCredentialMessage(`${providerName} key is held in backend memory for this session.`)
      setProviderTest(null)
    } catch (error) { setProviderApiKey(''); setCredentialMessage(error instanceof Error ? error.message : `Could not save ${providerName} API key.`) }
    finally { setCredentialBusy(false) }
  }

  async function handleClearKey() {
    setCredentialBusy(true); setCredentialMessage('')
    const provider = generationProvider
    const providerName = PROVIDER_INFO[provider]?.name || provider
    try {
      const response = await apiFetch(`/api/v1/providers/credentials/${provider}`, { method: 'DELETE' })
      const payload = await response.json().catch(() => null) as any
      if (!response.ok) throw new Error(apiErrorDetail(payload?.detail) || `Could not clear ${providerName} API key (${response.status}).`)
      const configured = Boolean(payload.configured)
      setKeyConfiguredMap((current) => ({ ...current, [provider]: configured }))
      if (provider === 'groq') {
        setGenerationConfig((current) => current ? { ...current, groq_api_key_present: configured, credential_present: configured, configured: configured && Boolean(generationModel.trim()) } : current)
      }
      setCredentialMessage(configured ? 'Using the backend environment key.' : `${providerName} key cleared from backend memory.`)
      setProviderTest(null)
    } catch (error) { setCredentialMessage(error instanceof Error ? error.message : `Could not clear ${providerName} API key.`) }
    finally { setCredentialBusy(false) }
  }

  const fetchWorkflows = useCallback(async () => {
    try {
      const response = await apiFetch('/api/v1/workflows')
      if (response.ok) {
        setSavedWorkflows(await response.json())
      }
    } catch {
      // ignore
    }
  }, [])

  const autoSaveWorkflowSession = useCallback(async (customName?: string) => {
    if (!nodes.length) return
    try {
      const name = customName || workflowName || `Pipeline Session ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
      const response = await apiFetch('/api/v1/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          graph: graphForValidation,
        }),
      })
      if (response.ok) {
        void fetchWorkflows()
      }
    } catch {
      // ignore background auto-save failure
    }
  }, [nodes.length, workflowName, graphForValidation, fetchWorkflows])

  async function handleIndex(scope: 'all' | 'single' = 'single') {
    if (!graphForValidation) return
    const targetDoc = document
    if (scope === 'single' && !targetDoc) return
    if (scope === 'all' && documents.length === 0) return

    setRagBusy(true); setRagError(''); setRagResult(null)
    let tracked: ReturnType<typeof beginTrackedExecution> | null = null
    let succeeded = false
    try {
      tracked = beginTrackedExecution('index', ['source', 'parser', 'chunker', 'embedder', 'index'], ['source'])
      const executionGraph = {
        ...graphForValidation,
        nodes: graphForValidation.nodes.map((node) => {
          if (node.node_type_id === 'chunk.split') {
            return { ...node, config: { ...node.config, strategy: selectedStrategy, chunk_size: chunkSize, overlap: chunkOverlap } }
          }
          if (node.node_type_id === 'generation.answer') {
            return { ...node, config: { ...node.config, provider: generationProvider, model: generationModel.trim() || node.config.model } }
          }
          return node
        }),
      }

      if (scope === 'all' || !targetDoc) {
        const response = await apiFetch('/api/v1/documents/ingest-batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            graph: executionGraph,
            ...(executionProgressSupported ? { execution_id: tracked.executionId } : {}),
            generation_provider: generationProvider,
            generation_model: generationModel.trim() || undefined,
            document_ids: documents.map((d) => d.document_id),
          }),
        })
        const payload = await response.json().catch(() => null) as any
        if (!response.ok) throw new Error(apiErrorDetail(payload?.detail) || `Batch indexing failed (${response.status}).`)
        const refreshedList = await apiFetch('/api/v1/documents').then((r) => r.ok ? r.json() : [])
        if (Array.isArray(refreshedList)) {
          setDocuments(refreshedList)
          if (targetDoc) {
            const updated = refreshedList.find((d: UploadedDocument) => d.document_id === targetDoc.document_id)
            if (updated) setDocument(updated)
          } else if (refreshedList.length > 0) {
            setDocument(refreshedList[0])
          }
        }
        succeeded = true
      } else {
        const selection = {
          document_id: targetDoc.document_id,
          strategy_id: selectedStrategy,
          choice_mode: selectedStrategy === targetDoc.recommendation?.recommended_strategy_id ? 'accept_recommendation' : 'explicit_override',
          source_recommendation_id: selectedStrategy === targetDoc.recommendation?.recommended_strategy_id ? targetDoc.recommendation.recommendation_id : null,
          settings: { chunk_size: chunkSize, overlap: chunkOverlap },
        }
        const choose = await apiFetch(`/api/v1/documents/${targetDoc.document_id}/chunking-selection`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ selection }),
        })
        const chosen = await choose.json().catch(() => null) as any
        if (!choose.ok) throw new Error(apiErrorDetail(chosen?.detail) || `Chunk selection failed (${choose.status}).`)
        setDocument(chosen)

        const response = await apiFetch(`/api/v1/documents/${targetDoc.document_id}/ingest`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            graph: executionGraph,
            ...(executionProgressSupported ? { execution_id: tracked.executionId } : {}),
            generation_provider: generationProvider,
            generation_model: generationModel.trim() || undefined,
          }),
        })
        const payload = await response.json().catch(() => null) as any
        if (!response.ok) throw new Error(apiErrorDetail(payload?.detail) || `Indexing failed (${response.status}).`)
        const refreshedResponse = await apiFetch(`/api/v1/documents/${targetDoc.document_id}`)
        if (!refreshedResponse.ok) throw new Error(`Indexing completed, but document status refresh failed (${refreshedResponse.status}).`)
        const refreshed = await refreshedResponse.json()
        setDocument(refreshed)
        setDocuments((current) => [refreshed, ...current.filter((item) => item.document_id !== refreshed.document_id)])
        succeeded = true
      }
    } catch (error) {
      setRagError(error instanceof Error ? error.message : 'Indexing failed.')
      if (targetDoc) {
        try {
          const latestResponse = await apiFetch(`/api/v1/documents/${targetDoc.document_id}`)
          if (latestResponse.ok) {
            const latest = await latestResponse.json() as UploadedDocument
            setDocument(latest)
            setDocuments((current) => [latest, ...current.filter((item) => item.document_id !== latest.document_id)])
          }
        } catch { /* The indexing error remains visible */ }
      }
    } finally {
      const progress = await settleTrackedExecution(tracked)
      if (progress) setExecutionOutcome(progress.status === 'succeeded' ? 'succeeded' : progress.status === 'failed' ? 'failed' : succeeded ? 'succeeded' : 'failed')
      else {
        setExecutionOutcome(succeeded ? 'succeeded' : 'failed')
        if (succeeded) {
          const allCompleted: Record<string, ExecutionNodeStatus> = { source: 'completed', parser: 'completed', chunker: 'completed', embedder: 'completed', index: 'completed' }
          for (const n of nodes) allCompleted[n.id] = 'completed'
          setExecutionStatuses((current) => ({ ...current, ...allCompleted }))
          void autoSaveWorkflowSession()
        }
      }
      setRagBusy(false)
    }
  }

  async function handlePlanIngestion(docId: string) {
    try {
      setRagBusy(true); setRagError('')
      const response = await apiFetch(`/api/v1/documents/${docId}/plan-ingestion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'user_approved_auto' }),
      })
      const planPayload = await response.json()
      if (!response.ok) throw new Error(apiErrorDetail(planPayload?.detail) || 'Planning failed.')
      const refreshedResponse = await apiFetch(`/api/v1/documents/${docId}`)
      if (refreshedResponse.ok) {
        const refreshed = await refreshedResponse.json() as UploadedDocument
        setDocument(refreshed)
        setDocuments((current) => current.map((item) => (item.document_id === docId ? refreshed : item)))
      }
    } catch (err) {
      setRagError(err instanceof Error ? err.message : 'Ingestion planning failed.')
    } finally {
      setRagBusy(false)
    }
  }

  async function handleApprovePlan(docId: string) {
    try {
      setRagBusy(true); setRagError('')
      const response = await apiFetch(`/api/v1/documents/${docId}/approve-plan`, {
        method: 'POST',
      })
      const docPayload = await response.json()
      if (!response.ok) throw new Error(apiErrorDetail(docPayload?.detail) || 'Plan approval failed.')
      setDocument(docPayload)
      setDocuments((current) => current.map((item) => (item.document_id === docId ? docPayload : item)))
      if (docPayload.selection) {
        setSelectedStrategy(docPayload.selection.strategy_id)
        if (docPayload.selection.settings?.chunk_size) setChunkSize(docPayload.selection.settings.chunk_size)
        if (docPayload.selection.settings?.overlap) setChunkOverlap(docPayload.selection.settings.overlap)
        updateChunkerConfig('strategy', docPayload.selection.strategy_id)
      }
    } catch (err) {
      setRagError(err instanceof Error ? err.message : 'Plan approval failed.')
    } finally {
      setRagBusy(false)
    }
  }

  async function handleSaveCurrentWorkflow() {
    setSaveStatusMessage('Saving…')
    try {
      const response = await apiFetch('/api/v1/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: workflowName || 'Custom Workflow',
          graph: graphForValidation,
        }),
      })
      if (!response.ok) throw new Error(`Save failed (${response.status})`)
      setSaveStatusMessage('Saved!')
      void fetchWorkflows()
      setTimeout(() => setSaveStatusMessage(''), 2500)
    } catch (err) {
      setSaveStatusMessage(err instanceof Error ? err.message : 'Save failed')
      setTimeout(() => setSaveStatusMessage(''), 3000)
    }
  }

  async function handleDeleteWorkflow(id: string, name: string) {
    if (!window.confirm(`Delete saved pipeline "${name}"?`)) return
    try {
      const response = await apiFetch(`/api/v1/workflows/${id}`, { method: 'DELETE' })
      if (response.ok) {
        setSavedWorkflows((prev) => prev.filter((item) => item.id !== id))
        setSaveStatusMessage('Deleted')
        setTimeout(() => setSaveStatusMessage(''), 2000)
      }
    } catch {
      // ignore
    }
  }

  const handleNewWork = () => {
    if ((nodes.length || edges.length) && !window.confirm('Start a new workflow session? Any unsaved changes in current canvas will be lost.')) return
    setNodes([])
    setEdges([])
    setSelectedNodeId(null)
    setActiveTemplateId('start-from-blank-canvas')
    setWorkflowName(`Pipeline Session ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`)
    setReport(emptyReport)
    setValidationState('idle')
  }

  function handleRestoreWorkflow(wf: SavedWorkflowItem) {
    if (!registry) return
    const definitions = new Map(registry.node_types.map((d) => [d.node_type_id, d]))
    const baselinePositions: Record<string, { x: number; y: number }> = {
      source: { x: 30, y: 45 }, parser: { x: 280, y: 45 }, chunker: { x: 530, y: 45 },
      embedder: { x: 780, y: 45 }, index: { x: 1030, y: 45 }, query: { x: 180, y: 285 },
      retriever: { x: 500, y: 400 }, context: { x: 770, y: 400 }, answer: { x: 1040, y: 285 },
    }
    const newNodes: FlowNode[] = wf.graph.nodes.flatMap((item, index) => {
      const definition = definitions.get(item.node_type_id)
      return definition ? [{
        id: item.id,
        type: 'capability',
        position: baselinePositions[item.id] ?? { x: 120 + index * 260, y: 130 },
        data: { definition, config: { ...defaultsFor(definition), ...item.config }, provider_instance_id: item.provider_instance_id || DEFAULT_PROVIDER_INSTANCE_BY_TYPE[item.node_type_id] || null },
        className: '',
      }] : []
    })
    const newEdges: Edge[] = wf.graph.connections.map((edge) => ({
      id: makeId(),
      source: edge.source_node_id,
      sourceHandle: edge.source_port_id,
      target: edge.target_node_id,
      targetHandle: edge.target_port_id,
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed, color: '#9dbf7b' },
    }))
    setNodes(newNodes)
    setEdges(newEdges)
    setWorkflowName(wf.name)
    setActiveTemplateId(wf.graph.architecture_id || 'custom-rag')
    setTimeout(() => { try { fitView({ padding: 0.15, duration: 300 }) } catch {} }, 100)
  }

  async function handleQuery(event: React.FormEvent) {
    event.preventDefault()
    if (!ragQuestion.trim() || ragBusy) return
    const currentQ = ragQuestion.trim()
    setRagBusy(true)
    setRagError('')
    setRagQuestion('')

    // Append user question turn immediately to conversational stream
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: currentQ,
    }
    setChatMessages((prev) => [...prev, userMsg])

    const queryTypes = new Set(['query.input', 'retrieval.dense', 'retrieval.sparse', 'context.assemble', 'generation.answer', 'agent.orchestrate', 'merge.fuse', 'retrieval.rerank'])
    const queryNodes = nodes.filter((n) => queryTypes.has(n.data.definition.node_type_id))
    const queryNodeIds = queryNodes.map((n) => n.id)
    const initialCompleted = queryNodes.filter((n) => n.data.definition.node_type_id === 'query.input').map((n) => n.id)
    const tracked = beginTrackedExecution('query', queryNodeIds.length ? queryNodeIds : ['query', 'retriever', 'context', 'answer'], initialCompleted.length ? initialCompleted : ['query'])
    let succeeded = false
    try {
      const topK = Number(graphForValidation.nodes.find((node) => node.node_type_id === 'retrieval.dense')?.config.top_k ?? 5)
      const targetDocIds = (queryScope === 'active' && document?.status === 'ready')
        ? [document.document_id]
        : undefined
      const executionGraph = {
        ...graphForValidation,
        nodes: graphForValidation.nodes.map((node) => {
          if (node.node_type_id === 'generation.answer') {
            return {
              ...node,
              config: {
                ...node.config,
                provider: generationProvider,
                model: generationModel.trim() || node.config.model,
              },
            }
          }
          return node
        }),
      }
      const hasAgentNode = executionGraph.nodes.some((node) => node.node_type_id === 'agent.orchestrate')
      const agentNode = executionGraph.nodes.find((node) => node.node_type_id === 'agent.orchestrate')
      const endpoint = hasAgentNode ? '/api/v1/agent/run' : '/api/v1/query'
      const reqBody = hasAgentNode
        ? {
            query: currentQ,
            graph: executionGraph,
            execution_id: tracked.executionId,
            document_ids: targetDocIds,
            generation_provider: generationProvider,
            generation_model: generationModel.trim() || undefined,
            initial_strategy: 'hybrid',
            allow_external_processing: authorizeExternal,
            goal: typeof agentNode?.config?.goal === 'string' ? agentNode.config.goal : undefined,
            instructions: typeof agentNode?.config?.instructions === 'string' ? agentNode.config.instructions : undefined,
            mode: typeof agentNode?.config?.mode === 'string' ? agentNode.config.mode : undefined,
            max_steps: agentNode?.config?.max_steps ? Number(agentNode.config.max_steps) : undefined,
            max_retrieval_iterations: agentNode?.config?.max_retrieval_iterations ? Number(agentNode.config.max_retrieval_iterations) : undefined,
            token_budget: agentNode?.config?.token_budget ? Number(agentNode.config.token_budget) : undefined,
            timeout_seconds: agentNode?.config?.timeout_seconds ? Number(agentNode.config.timeout_seconds) : undefined,
            approval_policy: typeof agentNode?.config?.approval_policy === 'string' ? agentNode.config.approval_policy : undefined,
          }
        : {
            question: currentQ,
            graph: executionGraph,
            ...(executionProgressSupported ? { execution_id: tracked.executionId } : {}),
            top_k: topK,
            document_ids: targetDocIds,
            generation_provider: generationProvider,
            generation_model: generationModel.trim(),
            allow_external_processing: authorizeExternal,
          }
      const response = await apiFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody),
      })
      const payload = (await response.json().catch(() => null)) as any
      if (!response.ok) throw new Error(apiErrorDetail(payload?.detail) || `Query failed (${response.status}).`)
      setRagResult(payload)
      succeeded = true

      const assistantMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: payload.answer,
        status: payload.status,
        citations: payload.citations,
        retrieved_chunks: payload.retrieved_chunks,
        collection: payload.collection,
        generation_provider: payload.generation_provider,
        generation_model: payload.generation_model,
        embedding_model: payload.embedding_model,
        execution_id: payload.execution_id,
        execution_started_at: payload.execution_started_at,
        execution_completed_at: payload.execution_completed_at,
        provider_output: payload.provider_output,
        context_preview: payload.context_preview,
        generation_error: payload.generation_error,
        trace: payload.trace,
        step_count: payload.step_count,
      }
      setChatMessages((prev) => [...prev, assistantMsg])
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Query failed.'
      setRagError(msg)
      setChatMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: msg,
          status: 'generation_failed',
        },
      ])
    } finally {
      // Preserve session external authorization so follow-up queries work without re-checking
      const progress = await settleTrackedExecution(tracked)
      if (progress) {
        setExecutionOutcome(progress.status === 'succeeded' ? 'succeeded' : progress.status === 'failed' ? 'failed' : succeeded ? 'succeeded' : 'failed')
      } else {
        setExecutionOutcome(succeeded ? 'succeeded' : 'failed')
      }
      if (succeeded) {
        const allCompleted: Record<string, ExecutionNodeStatus> = { query: 'completed', retriever: 'completed', context: 'completed', answer: 'completed', agent: 'completed' }
        for (const n of nodes) {
          if (queryTypes.has(n.data.definition.node_type_id)) {
            allCompleted[n.id] = 'completed'
          }
        }
        setExecutionStatuses((current) => ({ ...current, ...allCompleted }))
        void autoSaveWorkflowSession()
      } else {
        setExecutionStatuses((current) => {
          const updated = { ...current }
          for (const id of Object.keys(updated)) {
            if (updated[id] === 'queued' || updated[id] === 'running') {
              updated[id] = 'failed'
            }
          }
          return updated
        })
      }
      setRagBusy(false)
    }
  }

  const activeExecutionNode = nodes.find((node) => executionStatuses[node.id] === 'running')
  const executionLabel = executionOutcome === 'running'
    ? `${executionMode === 'index' ? 'Indexing' : 'Querying'} · ${activeExecutionNode?.data.definition.display_name ?? 'Starting workflow'}`
    : executionOutcome === 'failed' ? `${executionMode === 'index' ? 'Indexing' : 'Query'} failed`
      : executionOutcome === 'succeeded' ? `${executionMode === 'index' ? 'Indexing' : 'Query'} complete` : ''

  return <main className="builder-shell">
    <header className="builder-topbar">
      <div className="brand"><span className="brand-mark">V</span><span>VISUAL RAG STUDIO</span></div>
      <div className="top-divider" />
      <div className="workflow-name"><Workflow size={16} /><input aria-label="Workflow title" value={workflowName} onChange={(event) => setWorkflowName(event.target.value)} /><span className="draft-label">DRAFT</span></div>
      <nav className="view-nav" aria-label="Workspace views">{([['workflow', 'Workflow'], ['documents', 'Documents'], ['chunks', 'Chunks'], ['storage', 'Storage']] as [AppView, string][]).map(([view, label]) => <button key={view} className={activeView === view ? 'active' : ''} onClick={() => { setActiveView(view); setStorageError(''); setStorageMessage('') }}>{label}</button>)}</nav>
      <div className="top-actions">
        <span className={`api-indicator ${apiState}`}><i />{apiState === 'loading' ? 'Connecting' : apiState === 'ready' ? 'API connected' : 'API unavailable'}</span>
        <button className="button-secondary" onClick={() => { void runValidation(graphForValidation) }} disabled={!registry || validationState === 'checking'}><ShieldCheck size={15} /> Validate</button>
        <button className="button-primary" onClick={() => void handleIndex()} disabled={ragBusy || !document?.recommendation || !providerStatus?.online || !providerStatus.embedding_model_available || !generationModel.trim() || !isKeyConfigured || !report.valid || !report.executable} title="Validate and index the selected document with this workflow graph"><Play size={14} /> Run workflow</button>
      </div>
    </header>

    {activeView === 'workflow' ? <div className={`builder-layout${paletteCollapsed ? ' palette-collapsed' : ''}${inspectorCollapsed ? ' inspector-collapsed' : ''}`}>
      {!paletteCollapsed && <aside className="palette panel">
        <div className="sidebar-tab-strip">
          <button
            type="button"
            className={`sidebar-tab ${sidebarMode === 'palette' ? 'active' : ''}`}
            onClick={() => setSidebarMode('palette')}
          >
            <Layers3 size={12} />
            Components
          </button>
          <button
            type="button"
            className={`sidebar-tab ${sidebarMode === 'history' ? 'active' : ''}`}
            onClick={() => {
              setSidebarMode('history')
              void fetchWorkflows()
            }}
          >
            <Clock size={12} />
            Work History
            <span className="tab-count">{savedWorkflows.length}</span>
          </button>
        </div>

        {sidebarMode === 'palette' ? (
          <>
            <div className="panel-heading"><div><span className="eyebrow">WORKFLOW DESIGN</span><h2>Node palette</h2></div><span className="count-badge">{registry?.node_types.length ?? '—'}</span></div>
            <label className="search-box"><Search size={15} /><input placeholder="Find a node…" value={query} onChange={(event) => setQuery(event.target.value)} aria-label="Search nodes" />{query && <button aria-label="Clear node search" onClick={() => setQuery('')}><X size={13} /></button>}</label>
            <div className="category-strip" aria-label="Filter node category">{categories.map((item) => <button key={item} className={category === item ? 'active' : ''} onClick={() => setCategory(item)}>{item}</button>)}</div>
            <div className="palette-list">
              {apiState === 'loading' && <div className="empty-state"><span className="spinner" />Loading node definitions…</div>}
              {apiState === 'error' && <div className="inline-error"><AlertCircle size={15} />Can't load node definitions.<button onClick={() => window.location.reload()}>Retry</button></div>}
              {apiState === 'ready' && filteredNodes.map((definition, index) => { const Icon = icons[index % icons.length]; return <button className="palette-card" key={definition.node_type_id} draggable onDragStart={(event) => { event.dataTransfer.setData('application/reactflow', definition.node_type_id); event.dataTransfer.effectAllowed = 'move' }} onClick={() => addNode(definition)} title="Click or drag to canvas">
                <span className="palette-icon"><Icon size={16} /></span><span className="palette-copy"><strong>{definition.display_name}</strong><code>{definition.node_type_id}</code><small>{definition.description}</small></span><span className={`status-pill ${definition.implementation_status}`}>{definition.implementation_status}</span><Plus className="palette-add" size={15} />
              </button> })}
              {apiState === 'ready' && !filteredNodes.length && <div className="empty-state">No nodes match this search.</div>}
            </div>
            <div className="palette-footer"><CircleHelp size={14} /> Planned nodes are editable contracts, not runnable components.</div>
          </>
        ) : (
          <div className="work-history-panel">
            <div className="work-history-header">
              <span>WORK HISTORY ({savedWorkflows.length})</span>
            </div>

            <button
              type="button"
              className="button-new-work"
              onClick={handleNewWork}
              title="Create a new workflow session"
            >
              <Plus size={14} strokeWidth={2.5} /> New Pipeline
            </button>

            <div className="work-history-card">
              <button
                type="button"
                className="saved-pipelines-accordion-btn"
                onClick={() => setSavedPipelinesExpanded((expanded) => !expanded)}
              >
                <div className="accordion-title">
                  <Clock size={13} />
                  <span>Saved Pipelines</span>
                </div>
                <ChevronDown size={14} className={savedPipelinesExpanded ? 'expanded' : ''} />
              </button>

              {savedPipelinesExpanded && (
                <div className="saved-pipelines-content">
                  {savedWorkflows.length === 0 ? (
                    <p className="empty-history-notice">
                      No saved pipelines yet. Click '+ New Pipeline' or run an execution to save sessions.
                    </p>
                  ) : (
                    <div className="saved-pipelines-list">
                      {savedWorkflows.map((wf) => (
                        <div key={wf.id} className="saved-pipeline-item">
                          <div className="saved-pipeline-meta">
                            <strong title={wf.name}>{wf.name}</strong>
                            <div className="saved-pipeline-sub">
                              <span className="arch-badge">{wf.graph?.architecture_id || 'rag'}</span>
                              <small>{new Date(wf.updated_at).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</small>
                            </div>
                          </div>
                          <div className="saved-pipeline-actions">
                            <button
                              type="button"
                              className="button-load-mini"
                              onClick={() => handleRestoreWorkflow(wf)}
                              title="Load pipeline into canvas"
                            >
                              Load
                            </button>
                            <button
                              type="button"
                              className="button-delete-mini"
                              onClick={() => void handleDeleteWorkflow(wf.id, wf.name)}
                              title="Delete saved pipeline"
                            >
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="studio-brand-footer">
              <div className="brand-logo-row">
                <span className="brand-mini-mark">V</span>
                <strong>VISUAL RAG STUDIO</strong>
              </div>
              <small>Visual RAG Architecture & Execution</small>
            </div>
          </div>
        )}
      </aside>}

      <section ref={workspaceRef} className="workspace" style={{ '--playground-height': `${playgroundHeight}px` } as React.CSSProperties}>
        <div className="canvas-toolbar">
          <div className="template-picker-wrap">
            <button className="template-picker" onClick={() => setTemplateMenu((open) => !open)} disabled={!registry}><span className="template-mark"><Workflow size={15} /></span><span><small>WORKFLOW TEMPLATE</small><strong>{activeCatalogEntry?.name ?? 'Choose template'}</strong></span><ChevronDown size={15} /></button>
          {templateMenu && <div className="template-menu"><label className="search-box"><Search size={14} /><input aria-label="Search architectures" placeholder="Search architecture catalog" value={templateSearch} onChange={(event) => setTemplateSearch(event.target.value)} /></label><div className="template-list">{priority.map((id) => { const entry = catalog.find((item) => item.id === id); return entry && visibleCatalog.some((visible) => visible.id === id) ? <button key={id} onClick={() => chooseTemplate(entry)}><span>{entry.name}</span><span className={`status-pill ${entry.implementation_status}`}>{entry.kind === 'blank_canvas' ? 'blank canvas' : (entry.implementation_status === 'implemented' ? 'available' : entry.implementation_status)}</span></button> : null })}{visibleCatalog.filter((entry) => !priority.includes(entry.id)).map((entry) => <button key={entry.id} onClick={() => chooseTemplate(entry)}><span>{entry.name}<small>{entry.category}</small></span><span className={`status-pill ${entry.implementation_status}`}>{entry.implementation_status === 'implemented' ? 'available' : entry.implementation_status}</span></button>)}</div></div>}
          </div>
          <div className="canvas-actions">
            <button className="workflow-save-btn" onClick={() => fitView({ padding: 0.15, duration: 300 })} title="Fit entire architecture into one frame">
              <Maximize2 size={13} /> Fit to Frame
            </button>
            <button className="workflow-save-btn" onClick={() => void handleSaveCurrentWorkflow()} title="Save workflow to backend">
              <Save size={13} /> {saveStatusMessage || 'Save'}
            </button>
            <button
              className="workflow-save-btn"
              onClick={() => {
                setPaletteCollapsed(false)
                setSidebarMode('history')
                void fetchWorkflows()
              }}
              title="Open Work History & Saved Pipelines"
            >
              <FolderOpen size={13} /> Workflows ({savedWorkflows.length})
            </button>
            <button className="panel-toggle" onClick={() => setPaletteCollapsed((collapsed) => !collapsed)} aria-label={paletteCollapsed ? 'Show node palette' : 'Hide node palette'} title={paletteCollapsed ? 'Show node palette' : 'Hide node palette'}>{paletteCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}</button>
            <button className="panel-toggle" onClick={() => setInspectorCollapsed((collapsed) => !collapsed)} aria-label={inspectorCollapsed ? 'Show inspector' : 'Hide inspector'} title={inspectorCollapsed ? 'Show inspector' : 'Hide inspector'}>{inspectorCollapsed ? <PanelRightOpen size={15} /> : <PanelRightClose size={15} />}</button>
            {executionLabel && <span className={`execution-indicator ${executionOutcome}`} role="status">{executionLabel}</span>}
            <span className="canvas-navigation-hint">Drag from palette · Drag handle to connect · Drag wire, double-click or right-click to disconnect</span>
            <span className="canvas-count">{nodes.length} nodes <i /> {edges.length} connections</span>
            <button className="icon-button" onClick={clearCanvas} aria-label="Clear canvas" title="Start with blank canvas"><Trash2 size={15} /></button>
          </div>
        </div>
        {apiError && apiState === 'error' && <div className="api-error-banner"><AlertCircle size={15} />{apiError} Start this workspace's backend on port 8001, then use Retry.</div>}
        {activeTemplate?.graph === null && activeTemplate && <div className="template-notice"><span className="notice-dot" /><div><strong>{activeTemplate.name} is planned</strong><span>No template graph is available yet. You can compose a draft from registry nodes; it will not be executable.</span></div><button onClick={() => { setActiveTemplateId('start-from-blank-canvas'); setWorkflowName('Untitled workflow') }} aria-label="Dismiss template notice"><X size={14} /></button></div>}
        <div className="flow-canvas" aria-label="Workflow canvas" onDragOver={onDragOver} onDrop={onDrop}>
          {apiState === 'ready' && (
            <ConnectionDragContext.Provider value={{ dragInfo: connectionDrag, nodes }}>
              <ReactFlow
                proOptions={{ hideAttribution: true }}
                nodes={flowNodes}
                edges={flowEdges}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={(connection) => {
                  setConnectionDrag(null)
                  onConnect(connection)
                }}
                onConnectStart={(_, params) => {
                  setConnectionDrag({
                    nodeId: params.nodeId,
                    handleId: params.handleId,
                    handleType: params.handleType,
                  })
                }}
                onConnectEnd={(_, state) => {
                  setConnectionDrag(null)
                  if (state?.fromHandle && !state.isValid) {
                    setValidationError('Connection rejected. Check that the selected output and input types are compatible.')
                    setValidationState('error')
                  }
                }}
                connectionRadius={32}
                edgesReconnectable={true}
                edgesFocusable={true}
                onReconnectStart={onReconnectStart}
                onReconnect={onReconnect}
                onReconnectEnd={onReconnectEnd}
                onEdgeDoubleClick={onEdgeDoubleClick}
                onEdgeContextMenu={onEdgeContextMenu}
                isValidConnection={(connection) => compatible(connection, nodes)}
                onNodeClick={(_, node) => setSelectedNodeId(node.id)}
                onPaneClick={() => setSelectedNodeId(null)}
                onSelectionChange={({ nodes: selected }) => {
                  if (selected.length) setSelectedNodeId(selected[0].id)
                }}
                deleteKeyCode={['Backspace', 'Delete']}
                fitView
                fitViewOptions={{ padding: 0.15 }}
                minZoom={0.25}
                maxZoom={1.5}
                panOnDrag
                panOnScroll
                panOnScrollMode={PanOnScrollMode.Free}
                zoomOnScroll={false}
                zoomActivationKeyCode="Control"
                defaultEdgeOptions={{ type: 'smoothstep', style: { stroke: '#8eb961', strokeWidth: 2.2 } }}
              >
                <Background color="#39414b" gap={22} size={1} />
                <Controls position="bottom-left" showInteractive={false} />
              </ReactFlow>
            </ConnectionDragContext.Provider>
          )}
          {apiState === 'ready' && nodes.length === 0 && <div className="canvas-empty"><div className="empty-canvas-icon"><Workflow size={25} /></div><strong>Start building your workflow</strong><p>Choose a node from the palette or select a planned template to compose a draft.</p><button className="button-secondary" onClick={() => setQuery('')}>Browse components</button><small>Changes stay in this browser session.</small></div>}
        </div>
        <div className="playground-resizer" role="separator" aria-label="Resize playground; double-click to maximize or restore" aria-orientation="horizontal" aria-valuemin={250} aria-valuemax={maxPlaygroundHeight()} aria-valuenow={playgroundHeight} title="Drag up or down to resize. Double-click to maximize or restore." tabIndex={0} onDoubleClick={() => setPlaygroundHeight((height) => height >= maxPlaygroundHeight() - 8 ? 300 : maxPlaygroundHeight())} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); playgroundDrag.current = { pointerId: event.pointerId, y: event.clientY, height: playgroundHeight } }} onPointerMove={(event) => { if (playgroundDrag.current?.pointerId === event.pointerId) resizePlayground(event.clientY) }} onPointerUp={(event) => { if (playgroundDrag.current?.pointerId === event.pointerId) playgroundDrag.current = null }} onPointerCancel={() => { playgroundDrag.current = null }} onKeyDown={(event) => { if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return; event.preventDefault(); setPlaygroundHeight((height) => Math.max(250, Math.min(maxPlaygroundHeight(), height + (event.key === 'ArrowUp' ? 24 : -24)))) }}><span /></div>
        <section className="playground panel">
          <div className="playground-heading">
            <span className="playground-icon"><Play size={14} /></span>
            <div>
              <strong>{nodes.some((n) => n.data.definition.node_type_id === 'agent.orchestrate') ? 'Agentic RAG Playground' : 'Traditional RAG Playground'}</strong>
              <small>Conversational chat, grounded answers & verifiable citations</small>
            </div>
            {chatMessages.length > 0 && (
              <button className="clear-chat-button" onClick={() => { setChatMessages([]); setRagResult(null) }} title="Clear conversation history">
                <RotateCcw size={11} style={{ marginRight: 3, verticalAlign: 'middle' }} /> Clear chat
              </button>
            )}
            <span className={`runtime-badge ${providerStatus?.online && providerStatus.embedding_model_available ? 'runtime-ready' : ''}`}>
              {providerStatus?.online && providerStatus.embedding_model_available ? 'EMBEDDER CONFIGURED' : 'EMBEDDER UNAVAILABLE'}
            </span>
          </div>
          <div className="playground-content">
            <div className={`validation-summary ${validationState}`}>
              <span className="validation-icon">{validationState === 'error' ? <AlertCircle size={15} /> : validationState === 'checking' ? <span className="spinner" /> : report.executable ? <Check size={15} /> : report.valid ? <ShieldCheck size={15} /> : <AlertCircle size={15} />}</span>
              <div>
                <strong>{validationState === 'checking' ? 'Checking graph…' : validationState === 'error' ? 'Validation unavailable' : report.executable ? 'Workflow executable' : report.valid ? 'Graph structure valid' : 'Draft workflow'}</strong>
                <small>{validationState === 'error' ? validationError : `${report.issues.filter((issue) => issue.severity === 'error').length} errors · ${report.issues.filter((issue) => issue.severity === 'warning').length} warnings · ${report.executable ? 'executable' : report.valid ? 'not executable' : 'draft'}`}</small>
              </div>
            </div>
          </div>
          <div className="rag-runtime">
            {!documents.some((item) => item.status === 'ready') && (
              <p className="query-prerequisite" role="status">
                Select the <strong>Document Source</strong> node to upload and index a source before asking questions. There are no indexed sources available on this backend right now. <button onClick={() => selectOrAddNode('document.source')}>Open document source</button>
              </p>
            )}

            <div className="chat-thread">
              {chatMessages.length === 0 ? (
                <div className="chat-welcome">
                  <MessageSquare size={14} />
                  <span>Ask questions about your indexed documents. Answers are grounded in retrieved source passages with verifiable citations.</span>
                </div>
              ) : (
                chatMessages.map((msg) => (
                  <div key={msg.id} className={`chat-turn chat-turn-${msg.role}`}>
                    <div className="chat-bubble">
                      {msg.role === 'assistant' && (
                        <div className="assistant-header">
                          <span className={`answer-badge ${msg.status === 'abstained' ? 'abstained' : msg.status === 'generation_failed' ? 'failed' : ''}`}>
                            {msg.status === 'authorization_required' ? 'External Auth Required' : msg.status === 'generation_failed' ? 'Generation Failed' : msg.status === 'abstained' ? 'Abstained' : 'Grounded Answer'}
                          </span>
                          {msg.execution_id && <small style={{ color: '#7e8e84', fontSize: 8 }}>id: {msg.execution_id.slice(0, 8)}</small>}
                        </div>
                      )}
                      {msg.role === 'assistant' ? (
                        <FormattedAssistantMessage content={msg.content} citations={msg.citations} />
                      ) : (
                        <p className="assistant-body">{msg.content}</p>
                      )}

                      {msg.citations && msg.citations.length > 0 && (
                        <details className="citations-box" open={false}>
                          <summary>
                            <FileText size={12} />
                            <span>{msg.citations.length} Source citation{msg.citations.length > 1 ? 's' : ''}</span>
                          </summary>
                          <div className="citation-items">
                            {msg.citations.map((c, idx) => (
                              <article key={c.chunk_id || idx} className="citation-entry">
                                <div className="citation-entry-head">
                                  <span className="citation-pill-badge">[{idx + 1}]</span>
                                  <span>{c.document_name}{c.page_number != null ? ` · Page ${c.page_number}` : ''}</span>
                                  {c.rrf_score != null && <small>RRF score: {c.rrf_score.toFixed(5)}</small>}
                                  {c.weighted_score != null && <small>Weighted score: {c.weighted_score.toFixed(4)}</small>}
                                  {c.sparse_score != null && <small>BM25 score: {c.sparse_score.toFixed(3)}</small>}
                                  {c.distance != null && <small>Cosine vector distance: {c.distance.toFixed(4)}</small>}
                                  {(c.dense_rank != null || c.sparse_rank != null) && <small>Ranks: {c.dense_rank != null ? `semantic #${c.dense_rank}` : ''}{c.dense_rank != null && c.sparse_rank != null ? ' · ' : ''}{c.sparse_rank != null ? `BM25 #${c.sparse_rank}` : ''}</small>}
                                </div>
                                <p>{c.text || c.excerpt}</p>
                                <code>{c.chunk_id}</code>
                              </article>
                            ))}
                          </div>
                        </details>
                      )}

                      {msg.trace && msg.trace.length > 0 && (
                        <details className="citations-box agent-trace-box" open={false} style={{ marginTop: '6px' }}>
                          <summary style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', color: '#c084fc' }}>
                            <Workflow size={12} />
                            <span>Agent Reasoning Trace ({msg.trace.length} steps{msg.step_count ? ` · steps taken: ${msg.step_count}` : ''})</span>
                          </summary>
                          <div style={{ padding: '6px 0', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            {msg.trace.map((step: any, sIdx: number) => (
                              <div key={sIdx} style={{ background: '#1c1328', border: '1px solid #4a286e', borderRadius: '4px', padding: '5px 8px', fontSize: '10px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                                  <span style={{ fontWeight: 600, color: '#e9d5ff' }}>Step {step.step_number}: <code style={{ color: '#d8b4fe' }}>{step.tool}</code></span>
                                  <span style={{ color: '#a855f7' }}>{step.duration_ms}ms · {step.status}</span>
                                </div>
                                {step.reasoning && <p style={{ margin: '2px 0', color: '#d8b4fe', fontStyle: 'italic' }}>{step.reasoning}</p>}
                                <small style={{ color: '#c084fc', opacity: 0.8 }}>{step.output_summary}</small>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}

                      {msg.role === 'assistant' && (
                        <details className="trace-panel-box">
                          <summary><Activity size={11} /> Retrieval & Model Trace</summary>
                          <div className="trace-drawer">
                            <span>Status: <b>{msg.status ?? 'completed'}</b> · Collection: <code>{msg.collection ?? 'visual_rag_studio_documents'}</code></span>
                            <span>Provider: <b>{msg.generation_provider ?? 'local'} / {msg.generation_model ?? 'default'}</b> · Embeddings: <b>{msg.embedding_model ?? 'default'}</b></span>
                            {msg.execution_started_at && <small>Started: {new Date(msg.execution_started_at).toLocaleTimeString()} · Finished: {msg.execution_completed_at ? new Date(msg.execution_completed_at).toLocaleTimeString() : '—'}</small>}
                            {msg.retrieved_chunks && msg.retrieved_chunks.length > 0 && (
                              <details>
                                <summary>All retrieved candidate passages ({msg.retrieved_chunks.length})</summary>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
                                  {msg.retrieved_chunks.map((chk, i) => (
                                    <article className="trace-chunk" key={chk.chunk_id || i}>
                                      <strong>#{chk.rank ?? i + 1} · {chk.document_name}{chk.page_number ? ` · page ${chk.page_number}` : ''}</strong>
                                      <code>{chk.chunk_id}</code>
                                      {(chk.rrf_score != null || chk.weighted_score != null || chk.sparse_score != null || chk.distance != null) && <small className="trace-chunk-scores">
                                        {chk.rrf_score != null ? `RRF score ${chk.rrf_score.toFixed(5)} · ` : ''}
                                        {chk.weighted_score != null ? `weighted score ${chk.weighted_score.toFixed(4)} · ` : ''}
                                        {chk.sparse_score != null ? `BM25 score ${chk.sparse_score.toFixed(3)} · ` : ''}
                                        {chk.distance != null ? `cosine vector distance ${chk.distance.toFixed(4)}` : ''}
                                        {(chk.dense_rank != null || chk.sparse_rank != null) ? ` · ranks semantic ${chk.dense_rank ?? '—'} / BM25 ${chk.sparse_rank ?? '—'}` : ''}
                                      </small>}
                                      <p>{chk.text || chk.excerpt}</p>
                                    </article>
                                  ))}
                                </div>
                              </details>
                            )}
                            {msg.context_preview && (
                              <details>
                                <summary>Context supplied to prompt</summary>
                                <blockquote>{msg.context_preview}</blockquote>
                              </details>
                            )}
                            {msg.provider_output && (
                              <details>
                                <summary>Raw provider output</summary>
                                <blockquote>{msg.provider_output}</blockquote>
                              </details>
                            )}
                          </div>
                        </details>
                      )}
                    </div>
                  </div>
                ))
              )}
              {ragBusy && (
                <div className="loading-turn">
                  <span className="spinner" />
                  <span>Searching indexed evidence and generating grounded response…</span>
                </div>
              )}
            </div>

            {documents.some((item) => item.status === 'ready') && (
              <div className="query-scope-bar" role="toolbar" aria-label="Query target selection">
                <span className="query-scope-label">Target:</span>
                <select
                  aria-label="Switch indexed document"
                  className="query-doc-select"
                  value={queryScope === 'all' ? '__all__' : (document?.document_id ?? '')}
                  onChange={(event) => {
                    const val = event.target.value
                    if (val === '__all__') {
                      setQueryScope('all')
                    } else {
                      setQueryScope('active')
                      const found = documents.find((doc) => doc.document_id === val)
                      if (found) {
                        setDocument(found)
                        if (found.recommendation) {
                          setSelectedStrategy(found.recommendation.recommended_strategy_id)
                          setChunkSize(found.recommendation.proposed_settings.chunk_size ?? 1000)
                          setChunkOverlap(found.recommendation.proposed_settings.overlap ?? 120)
                        }
                      }
                    }
                  }}
                >
                  <option value="__all__">🌐 All Indexed Documents ({documents.filter((item) => item.status === 'ready').length})</option>
                  <optgroup label="Single Document">
                    {documents.filter((item) => item.status === 'ready').map((doc) => (
                      <option key={doc.document_id} value={doc.document_id}>
                        📄 {doc.filename} ({doc.chunk_count} chunks)
                      </option>
                    ))}
                  </optgroup>
                </select>
                <button
                  type="button"
                  className={`scope-pill ${queryScope === 'active' ? 'active' : ''}`}
                  onClick={() => {
                    setQueryScope('active')
                    if (!document && documents.some((item) => item.status === 'ready')) {
                      const firstReady = documents.find((item) => item.status === 'ready')
                      if (firstReady) setDocument(firstReady)
                    }
                  }}
                  disabled={document?.status !== 'ready' && !documents.some((item) => item.status === 'ready')}
                  title={document?.status === 'ready' ? `Query current document: ${document.filename}` : 'Select a ready document'}
                >
                  Current: {document?.status === 'ready' ? (document.filename.length > 20 ? `${document.filename.slice(0, 18)}…` : document.filename) : 'None'}
                </button>
                <button
                  type="button"
                  className={`scope-pill ${queryScope === 'all' ? 'active' : ''}`}
                  onClick={() => setQueryScope('all')}
                  title="Search across all indexed documents"
                >
                  All ({documents.filter((item) => item.status === 'ready').length})
                </button>
              </div>
            )}

            {!providerTest?.ready && documents.some((item) => item.status === 'ready') && (
              <div className="provider-test-prompt" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px', background: '#202621', borderRadius: '6px', border: '1px solid #33442e', marginBottom: '8px' }}>
                <ShieldCheck size={14} style={{ color: '#9dbf7b', flexShrink: 0 }} />
                {isKeyConfigured ? (
                  <>
                    <span>Provider <strong>{PROVIDER_INFO[generationProvider]?.name || generationProvider}</strong> needs verification before asking.</span>
                    <button
                      type="button"
                      className="button-secondary"
                      style={{ padding: '3px 10px', fontSize: '11px', marginLeft: 'auto', whiteSpace: 'nowrap' }}
                      disabled={ragBusy || !generationModel.trim()}
                      onClick={() => void handleProviderTest()}
                    >
                      {ragBusy ? 'Testing…' : 'Test provider now'}
                    </button>
                  </>
                ) : (
                  <>
                    <span style={{ fontSize: '11px', color: '#c5cec7' }}>Provider <strong>{PROVIDER_INFO[generationProvider]?.name || generationProvider}</strong> requires an API key.</span>
                    <div style={{ display: 'flex', gap: '6px', marginLeft: 'auto', alignItems: 'center' }}>
                      <button
                        type="button"
                        className="button-secondary"
                        style={{ padding: '3px 8px', fontSize: '11px', whiteSpace: 'nowrap' }}
                        onClick={() => selectOrAddNode('generation.answer')}
                        title="Open the Inspector on the right to enter your API key"
                      >
                        Enter Key in Inspector
                      </button>
                      <button
                        type="button"
                        className="button-primary"
                        style={{ padding: '3px 8px', fontSize: '11px', whiteSpace: 'nowrap' }}
                        onClick={() => handleProviderChange('extractive')}
                        title="Switch to built-in extractive generation that runs 100% locally with zero API keys required"
                      >
                        ⚡ Use Free Offline Extractive
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            <form className="rag-query" onSubmit={(event) => void handleQuery(event)}>
              <input
                aria-label="Ask a question"
                placeholder={
                  document?.status === 'ready' && queryScope === 'active'
                    ? `Ask about ${document.filename}…`
                    : documents.some((item) => item.status === 'ready')
                    ? 'Ask about indexed sources…'
                    : 'Index a document to enable questions'
                }
                value={ragQuestion}
                onChange={(event) => setRagQuestion(event.target.value)}
                disabled={ragBusy || !documents.some((item) => item.status === 'ready')}
                required
              />
              <button
                type="submit"
                className="button-primary"
                disabled={ragBusy || !documents.some((item) => item.status === 'ready') || !ragQuestion.trim() || !providerTest?.ready || (isExternalProvider(generationProvider) && !authorizeExternal) || !report.valid || !report.executable}
                title={!documents.some((item) => item.status === 'ready') ? 'Index at least one document first.' : !providerTest?.ready ? 'Click "Test provider now" before asking.' : isExternalProvider(generationProvider) && !authorizeExternal ? `Allow this question to send retrieved excerpts to ${PROVIDER_INFO[generationProvider]?.name || 'the provider'}.` : undefined}
              >
                <Play size={13} />
                Ask
              </button>
            </form>

            {isExternalProvider(generationProvider) && (
              <label className="privacy-consent">
                <input type="checkbox" checked={authorizeExternal} onChange={(event) => setAuthorizeExternal(event.target.checked)} />
                <span>
                  <strong>Allow {PROVIDER_INFO[generationProvider]?.name || 'the provider'} to answer</strong>
                  <small>Retrieved excerpts from indexed documents will be sent to the external provider for queries in this session. Stays enabled across follow-ups.</small>
                </span>
              </label>
            )}

            {ragError && <div role="alert" className="rag-error"><AlertCircle size={14} />{ragError}</div>}
          </div>
          {validationState === 'error' && validationError && <div className="validation-api-error"><AlertCircle size={14} />{validationError}</div>}
          {report.issues.length > 0 && <div className="issue-list" aria-label="Validation issues">{report.issues.slice(0, 8).map((issue, index) => <button key={`${issue.code}-${index}`} className={`issue-row ${issue.severity}`} onClick={() => issue.node_id && setSelectedNodeId(issue.node_id)}><span className="issue-type">{issue.severity === 'error' ? <AlertCircle size={13} /> : <CircleHelp size={13} />}</span><span>{issue.message}</span>{issue.node_id && <code>{issue.node_id}</code>}</button>)}{report.issues.length > 8 && <small className="more-issues">+{report.issues.length - 8} more issues</small>}</div>}
        </section>
      </section>

      {!inspectorCollapsed && <aside className="inspector panel">
        <div className="panel-heading inspector-heading"><div><span className="eyebrow">DETAILS</span><h2>Inspector</h2></div><Settings2 size={17} /></div>
        {selectedNode ? <div className="inspector-scroll"><div className="inspected-node"><span className="node-icon large"><Workflow size={17} /></span><div><strong>{selectedNode.data.definition.display_name}</strong><code>{selectedNode.data.definition.node_type_id}</code></div></div>
          {selectedNode.data.definition.node_type_id === 'generation.answer' && <section className="inspector-section generation-node-settings"><div className="section-title">GENERATION PROVIDER</div>
            <label>Provider<select aria-label="Generation provider" value={generationProvider} onChange={(event) => handleProviderChange(event.target.value as GenerationProvider)}>
              <option value="groq">Groq</option>
              <option value="openai">OpenAI</option>
              <option value="gemini">Google Gemini</option>
              <option value="anthropic">Anthropic Claude</option>
              <option value="ollama">Ollama (Local)</option>
              <option value="huggingface">Hugging Face Inference Providers</option>
              <option value="extractive">Extractive (Local Offline)</option>
            </select></label>
            <label>Model selection<select aria-label="Recommended models" value={allAvailableModels.includes(generationModel) ? generationModel : ''} onChange={(event) => { if (event.target.value) { setGenerationModel(event.target.value); updateConfig('model', event.target.value); setProviderTest(null) } }}>
              <option value="">Custom model…</option>
              {allAvailableModels.map((m) => <option key={m} value={m}>{m}</option>)}
            </select></label>
            <label>Model ID<input aria-label="Generation model" value={generationModel} placeholder={PROVIDER_INFO[generationProvider]?.defaultModel ?? 'model:tag'} onChange={(event) => { setGenerationModel(event.target.value); updateConfig('model', event.target.value); setProviderTest(null) }} /></label>
            {PROVIDER_INFO[generationProvider]?.requiresKey && <><label>{PROVIDER_INFO[generationProvider].keyLabel}<input aria-label={PROVIDER_INFO[generationProvider].keyLabel} type="password" value={providerApiKey} autoComplete="new-password" spellCheck={false} placeholder={isKeyConfigured ? 'Key saved on backend' : PROVIDER_INFO[generationProvider].placeholder} onChange={(event) => setProviderApiKey(event.target.value)} /></label><div className="credential-actions"><button className="button-secondary" disabled={credentialBusy || !providerApiKey.trim()} onClick={() => void handleSaveKey()}>{credentialBusy ? 'Saving…' : 'Save key'}</button>{isKeyConfigured && <button className="button-secondary" disabled={credentialBusy} onClick={() => void handleClearKey()}>Clear key</button>}</div><small className="credential-note">Sent only to this loopback backend, held in backend memory, and never saved in browser storage. {credentialMessage}</small></>}
            <button className="button-secondary provider-test-button" disabled={ragBusy || !generationModel.trim() || !isKeyConfigured} onClick={() => void handleProviderTest()}>{providerTest?.ready ? 'Test again' : 'Test provider'}</button>
            <p className={`generation-readiness ${providerTest?.ready ? 'success' : providerTest ? 'failure' : ''}`} role="status">
              {generationProvider === 'groq'
                ? isKeyConfigured ? 'Groq key is configured on the backend.' : 'Save your Groq key here once to enable generation.'
                : generationProvider === 'huggingface'
                ? isKeyConfigured ? 'HF_TOKEN is configured on the backend.' : 'HF_TOKEN is missing from the backend environment.'
                : generationProvider === 'openai'
                ? isKeyConfigured ? 'OpenAI key is configured on the backend.' : 'Save your OpenAI key here once to enable generation.'
                : generationProvider === 'gemini'
                ? isKeyConfigured ? 'Gemini key is configured on the backend.' : 'Save your Gemini key here once to enable generation.'
                : generationProvider === 'anthropic'
                ? isKeyConfigured ? 'Anthropic key is configured on the backend.' : 'Save your Anthropic key here once to enable generation.'
                : generationProvider === 'extractive'
                ? 'Local extractive generator uses no external credentials.'
                : 'Ollama uses no token; test the configured model.'} {providerTest ? providerTest.message : 'Test the provider before asking a question.'}
            </p>
          </section>}
          {selectedNode.data.definition.node_type_id === 'document.source' && <section className="inspector-section source-node-settings"><div className="section-title">DOCUMENT SOURCE</div>
            <label className="source-upload">
              {documents.length > 0 ? `Upload / add documents (${documents.length} loaded)` : 'Upload TXT, Markdown, or PDF (1 to 100+ files)'}
              <input aria-label="Upload document" type="file" multiple accept=".txt,.md,.markdown,.pdf" disabled={ragBusy} onChange={(event) => { void handleDocumentUpload(event.target.files); event.currentTarget.value = '' }} />
            </label>
            {documents.length > 1 && (
              <label>Active document preview
                <select
                  aria-label="Select document"
                  value={document?.document_id ?? ''}
                  onChange={(event) => {
                    const found = documents.find((item) => item.document_id === event.target.value)
                    if (found) {
                      setDocument(found)
                      if (found.recommendation) {
                        setSelectedStrategy(found.recommendation.recommended_strategy_id)
                        setChunkSize(found.recommendation.proposed_settings.chunk_size ?? 1000)
                        setChunkOverlap(found.recommendation.proposed_settings.overlap ?? 120)
                      }
                    }
                  }}
                >
                  {documents.map((doc) => (
                    <option key={doc.document_id} value={doc.document_id}>
                      {doc.filename} ({doc.status}{doc.status === 'ready' ? ` · ${doc.chunk_count} chunks` : ''})
                    </option>
                  ))}
                </select>
              </label>
            )}
            {document && <><div className="source-document-summary">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                  <strong>{document.filename}</strong>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    {(document.status === 'ready' || document.chunk_count > 0) && (
                      <button
                        className="button-danger-outline"
                        style={{ height: '22px', padding: '0 6px', fontSize: '8px' }}
                        onClick={() => void handleDeleteDocumentChunks(document.document_id, document.filename)}
                        title="Clear chunks / un-index"
                      >
                        Clear chunks
                      </button>
                    )}
                    <button
                      className="button-danger-outline"
                      style={{ height: '22px', padding: '0 6px', fontSize: '8px' }}
                      onClick={() => void handleDeleteDocument(document.document_id, document.filename)}
                      title="Delete this uploaded file and vector index"
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>
                </div>
                <span>{document.status}{document.status === 'ready' ? ` · ${document.chunk_count} chunks indexed` : ''}</span>{document.profile?.approximate_text_length?.value ? <span>{document.profile.approximate_text_length.value.toLocaleString()} extracted characters</span> : null}</div>
              {document.recommendation && <><p className="recommendation-copy">{document.recommendation.rationale}</p><label>Chunking strategy<select aria-label="Chunking strategy" value={selectedStrategy} onChange={(event) => { setSelectedStrategy(event.target.value); updateChunkerConfig('strategy', event.target.value) }}>{(registry?.chunking_strategies ?? []).filter((strategy) => strategy.implementation_status === 'implemented' || strategy.implementation_status === 'tested').map((strategy) => <option key={strategy.id} value={strategy.id}>{strategy.display_name}</option>)}</select></label><div className="source-chunk-fields"><label>Chunk size<input aria-label="Chunk size" type="number" min={64} max={8000} value={chunkSize} onChange={(event) => { const value = Number(event.target.value); setChunkSize(value); updateChunkerConfig('chunk_size', value) }} /></label><label>Overlap<input aria-label="Chunk overlap" type="number" min={0} max={Math.max(0, chunkSize - 1)} value={chunkOverlap} onChange={(event) => { const value = Number(event.target.value); setChunkOverlap(value); updateChunkerConfig('overlap', value) }} /></label></div>
              <div className="source-index-actions" style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                <button
                  className="button-primary source-index-button"
                  style={{ flex: 1 }}
                  disabled={ragBusy || !providerStatus?.online || !providerStatus.embedding_model_available || !generationModel.trim() || !isKeyConfigured || !report.valid || !report.executable}
                  onClick={() => void handleIndex(documents.length > 1 ? 'all' : 'single')}
                >
                  <Database size={14} />
                  {ragBusy && executionMode === 'index' ? 'Indexing…' : documents.length > 1 ? `Index all (${documents.length})` : 'Index document'}
                </button>
                {documents.length > 1 && (
                  <button
                    className="button-secondary"
                    disabled={ragBusy || !providerStatus?.online || !providerStatus.embedding_model_available || !generationModel.trim() || !isKeyConfigured || !report.valid || !report.executable}
                    onClick={() => void handleIndex('single')}
                    title={`Index only ${document.filename}`}
                  >
                    Index current only
                  </button>
                )}
              </div>

              <div style={{ marginTop: '14px', background: '#121927', border: '1px solid #1e3a5f', borderRadius: '6px', padding: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 600, color: '#93c5fd' }}>
                    <Layers3 size={13} style={{ color: '#60a5fa' }} />
                    Autonomous Ingestion Planner
                  </div>
                  <button
                    className="button-secondary"
                    style={{ fontSize: '10px', padding: '2px 8px' }}
                    disabled={ragBusy}
                    onClick={() => void handlePlanIngestion(document.document_id)}
                  >
                    {document.plan ? 'Re-plan' : 'Plan Strategy'}
                  </button>
                </div>

                {document.plan ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '10px' }}>
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                      <span style={{ background: '#1e3a5f', color: '#bfdbfe', padding: '1px 6px', borderRadius: '3px' }}>
                        Mode: {document.plan.mode}
                      </span>
                      <span style={{ background: '#14532d', color: '#bbf7d0', padding: '1px 6px', borderRadius: '3px' }}>
                        Confidence: {Math.round(document.plan.confidence * 100)}%
                      </span>
                      <span style={{ background: '#312e81', color: '#c7d2fe', padding: '1px 6px', borderRadius: '3px' }}>
                        Dedup: {document.plan.deduplication_policy}
                      </span>
                    </div>

                    <div style={{ background: '#0a0f1d', padding: '6px 8px', borderRadius: '4px', border: '1px solid #172554' }}>
                      <div style={{ fontWeight: 600, color: '#93c5fd', marginBottom: '3px' }}>Strategies Selected:</div>
                      {document.plan.selected_strategies?.map((s: any, idx: number) => (
                        <div key={idx} style={{ color: '#cbd5e1' }}>
                          • <strong>{s.strategy_id}</strong> (size: {s.settings?.chunk_size ?? 1000}, overlap: {s.settings?.overlap ?? 120})
                          {s.reason && <small style={{ display: 'block', color: '#94a3b8', marginLeft: '10px' }}>{s.reason}</small>}
                        </div>
                      ))}
                    </div>

                    <div style={{ color: '#94a3b8', fontSize: '9px' }}>
                      <strong>Target Embedding:</strong> {document.plan.embedding_proposal?.model} ({document.plan.embedding_proposal?.dimension}d)
                    </div>

                    {document.plan.approved ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#4ade80', fontWeight: 600, fontSize: '10px', marginTop: '4px' }}>
                        <Check size={12} /> Plan Approved & Active for Indexing
                      </div>
                    ) : (
                      <button
                        className="button-primary"
                        style={{ marginTop: '4px', fontSize: '10px', padding: '4px 8px', background: '#2563eb' }}
                        disabled={ragBusy}
                        onClick={() => void handleApprovePlan(document.document_id)}
                      >
                        Approve Plan & Configure Chunker
                      </button>
                    )}
                  </div>
                ) : (
                  <p style={{ margin: 0, fontSize: '10px', color: '#64748b' }}>
                    Generate an autonomous multi-strategy ingestion plan based on structural heuristics and model compatibility.
                  </p>
                )}
              </div>
            </>}
            </>}
            {ragError && <p className="rag-error" role="alert">{ragError}</p>}
            <div className="source-embedding-status"><span>{providerStatus?.provider === 'sentence_transformers' ? 'Sentence Transformers' : 'Ollama'} embeddings</span><small>{providerStatus?.embedding_model_available ? providerStatus.embedding_model : providerStatus?.error ?? 'Checking embedding model…'}</small><button className="button-secondary" onClick={() => { void apiFetch('/api/v1/providers/health').then((response) => response.json()).then(setProviderStatus).catch(() => setProviderStatus(null)) }}>Refresh</button></div>
          </section>}
          <section className="inspector-section"><div className="section-title">IMPLEMENTATION</div><span className={`status-pill ${selectedNode.data.definition.implementation_status}`}>{selectedNode.data.definition.implementation_status}</span><p className="schema-description">{selectedNode.data.definition.description}</p></section>
          {(() => {
            const allProps = Object.entries(selectedNode.data.definition.config_schema.properties ?? {})
            const isGenAnswer = selectedNode.data.definition.node_type_id === 'generation.answer'
            const visibleProps = isGenAnswer
              ? allProps.filter(([name]) => name !== 'provider' && name !== 'model')
              : allProps
            return (
              <section className="inspector-section">
                <div className="section-title">CONFIGURATION <span>{visibleProps.length} fields</span></div>
                {selectedNode.data.definition.node_type_id === 'vector.index' && (
                  <div style={{ background: '#192518', border: '1px solid #3c5e29', borderRadius: '5px', padding: '7px 9px', marginBottom: '10px', fontSize: '9px', color: '#b9e88d', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Check size={13} style={{ flexShrink: 0, color: '#9bd867' }} />
                    <span><strong>Local Chroma:</strong> Defaults configured. External URL or API key is not required.</span>
                  </div>
                )}
                {selectedNode.data.definition.node_type_id === 'embedding.create' && (
                  <div style={{ background: '#192518', border: '1px solid #3c5e29', borderRadius: '5px', padding: '7px 9px', marginBottom: '10px', fontSize: '9px', color: '#b9e88d', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Check size={13} style={{ flexShrink: 0, color: '#9bd867' }} />
                    <span><strong>Free Models:</strong> Choose any model from the dropdown; dimension is set automatically.</span>
                  </div>
                )}
                {selectedNode.data.definition.node_type_id === 'agent.orchestrate' && (
                  <div style={{ background: '#221535', border: '1px solid #7e22ce', borderRadius: '5px', padding: '8px 10px', marginBottom: '10px', fontSize: '10px', color: '#e9d5ff', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
                      <Workflow size={13} style={{ color: '#c084fc' }} />
                      <span>Autonomous Agent Orchestrator</span>
                    </div>
                    <span style={{ color: '#d8b4fe', fontSize: '9px', opacity: 0.9 }}>
                      Configure goal, behavioral guidelines, step budgets, and strategy permissions. Instructions directly steer retrieval and grounded answer synthesis.
                    </span>
                  </div>
                )}
                {visibleProps.map(([name, field]) => {
                  if (selectedNode.data.definition.node_type_id === 'embedding.create' && name === 'model') {
                    const currentModel = String(selectedNode.data.config.model || field.default || 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                    const selectedModelInfo = EMBEDDING_MODELS.find((m) => m.id === currentModel)
                    const multilingualModels = EMBEDDING_MODELS.filter((m) => m.category === 'Multilingual')
                    const englishModels = EMBEDDING_MODELS.filter((m) => m.category === 'English')
                    const cloudModels = EMBEDDING_MODELS.filter((m) => m.category === 'Cloud / Ollama')
                    return (
                      <div key={name} className="schema-field">
                        <label htmlFor="config-model">
                          Model (Curated Free Models)
                        </label>
                        <select
                          id="config-model"
                          value={currentModel}
                          onChange={(e) => updateConfig('model', e.target.value)}
                        >
                          <optgroup label="🌐 Multilingual Free Models (Worldwide)">
                            {multilingualModels.map((m) => (
                              <option key={m.id} value={m.id}>
                                [{m.dimension} dim] {m.label} ({m.tag})
                              </option>
                            ))}
                          </optgroup>
                          <optgroup label="⚡ English / High-Performance Open Weights">
                            {englishModels.map((m) => (
                              <option key={m.id} value={m.id}>
                                [{m.dimension} dim] {m.label} ({m.tag})
                              </option>
                            ))}
                          </optgroup>
                          <optgroup label="☁️ Cloud Free-Tier & Local Ollama">
                            {cloudModels.map((m) => (
                              <option key={m.id} value={m.id}>
                                [{m.dimension} dim] {m.label} ({m.tag})
                              </option>
                            ))}
                          </optgroup>
                        </select>
                        {selectedModelInfo && (
                          <div style={{ marginTop: '5px', padding: '6px 8px', borderRadius: '4px', background: '#131b14', border: '1px solid #233a1e', fontSize: '10px', color: '#9bd867', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span><strong>Dimension:</strong> {selectedModelInfo.dimension}</span>
                              <span style={{ background: '#1e3818', padding: '1px 5px', borderRadius: '3px', fontSize: '9px', color: '#d2f99a' }}>{selectedModelInfo.category}</span>
                            </div>
                            <span style={{ color: '#88a380', fontSize: '9px' }}>{selectedModelInfo.description}</span>
                          </div>
                        )}
                        <small>{field.description}</small>
                        {fieldErrors[name] && <span className="field-error">{fieldErrors[name]}</span>}
                      </div>
                    )
                  }
                  return (
                    <FieldEditor
                      key={name}
                      name={name}
                      field={field}
                      defaultValue={field.default ?? selectedNode.data.definition.config_defaults[name]}
                      value={selectedNode.data.config[name] ?? field.default ?? selectedNode.data.definition.config_defaults[name]}
                      error={fieldErrors[name]}
                      onChange={(value) => updateConfig(name, value)}
                    />
                  )
                })}
                {!visibleProps.length && <p className="muted-copy">This component has no configurable fields.</p>}
              </section>
            )
          })()}
          <section className="inspector-section"><div className="section-title">CONNECTIONS</div><div className="ports-summary"><span>Inputs <b>{selectedNode.data.definition.inputs.length}</b></span><span>Outputs <b>{selectedNode.data.definition.outputs.length}</b></span></div></section>
          <section className="inspector-section"><div className="section-title">DEPENDENCIES</div>{selectedNode.data.definition.dependencies.length ? selectedNode.data.definition.dependencies.map((dep) => <code className="dependency-chip" key={dep}>{dep}</code>) : <p className="muted-copy">No dependency is declared.</p>}</section>
          <section className="inspector-section"><div className="section-title">PROVIDER REQUIREMENTS</div>{selectedNode.data.definition.provider_requirements.length ? selectedNode.data.definition.provider_requirements.map((provider) => <code className="dependency-chip" key={provider}>{provider}</code>) : <p className="muted-copy">No provider requirement is declared.</p>}</section>
          {selectedNode.data.definition.known_limitations.length > 0 && <section className="inspector-section"><div className="section-title">KNOWN LIMITATIONS</div>{selectedNode.data.definition.known_limitations.map((text, index) => <p className="limitation" key={index}>{text}</p>)}</section>}
          {nodeIssues.length > 0 && <section className="inspector-section"><div className="section-title">VALIDATION</div>{nodeIssues.map((issue, index) => <p key={index} className={`inspector-issue ${issue.severity}`}>{issue.message}</p>)}</section>}
        </div> : <div className="inspector-empty"><div className="empty-canvas-icon"><Settings2 size={22} /></div><strong>No node selected</strong><p>Select a node on the canvas to inspect its schema and edit its configuration.</p><button className="button-secondary" onClick={() => selectOrAddNode('generation.answer')}>Configure Generate Answer</button></div>}
        <details className="chunking-reserved"><summary><span className="reserved-icon"><ListFilter size={15} /></span><span><strong>Document-aware chunking</strong><small>{registry?.chunking_strategies.length ?? 0} strategy definitions · recommendation on upload</small></span><Code2 size={14} /></summary><div className="chunking-strategies">{(registry?.chunking_strategies ?? []).map((strategy) => <div key={strategy.id}><span>{strategy.display_name}</span><span className={`status-pill ${strategy.implementation_status}`}>{strategy.implementation_status}</span></div>)}<p>Recommendations are based on extracted document structure. Select a document in the Traditional RAG playground to review them.</p></div></details>
      </aside>}
    </div> : <section className="explorer-page panel">
      <div className="explorer-heading">
        <div>
          <span className="eyebrow">{activeView === 'storage' ? 'PERSISTENCE HEALTH' : activeView === 'chunks' ? 'CHROMA RECORDS' : 'SOURCE FILES'}</span>
          <h2>{activeView === 'storage' ? 'Storage overview' : activeView === 'chunks' ? 'Chunk explorer' : 'Documents'}</h2>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {activeView === 'documents' && documents.length > 0 && (
            <button className="button-danger-outline" onClick={() => void handleDeleteAllDocuments()} title="Permanently delete all documents, uploaded files, and vector embeddings">
              <Trash2 size={13} /> Delete all documents
            </button>
          )}
          {activeView === 'chunks' && (chunkPage?.total ?? 0) > 0 && (
            <button className="button-danger-outline" onClick={() => void handleClearAllChunks()} title="Clear chunks and vector embeddings">
              <Trash2 size={13} /> {selectedExplorerDoc ? 'Delete document chunks' : 'Clear all chunks'}
            </button>
          )}
          <button className="button-secondary" onClick={() => {
            setStorageError('');
            setStorageMessage('');
            if (activeView === 'storage') {
              setStorageSummary(null);
              void apiFetch('/api/v1/storage/summary').then(async (response) => {
                if (!response.ok) throw new Error(`Storage check failed (${response.status}).`);
                setStorageSummary(await response.json());
              }).catch((error) => setStorageError(error instanceof Error ? error.message : 'Storage check unavailable.'));
            } else if (activeView === 'chunks') {
              setChunkOffset(0);
              const params = new URLSearchParams({ limit: '20', offset: '0' });
              if (selectedExplorerDoc) params.set('document_id', selectedExplorerDoc);
              void apiFetch(`/api/v1/chunks?${params}`).then((r) => r.json()).then(setChunkPage).catch(() => undefined);
            } else {
              void apiFetch('/api/v1/documents').then((response) => response.json()).then(setDocuments).catch(() => setStorageError('Document list unavailable.'));
            }
          }}>Refresh</button>
        </div>
      </div>
      {storageError && <div className="explorer-error" role="alert"><AlertCircle size={15} />{storageError}</div>}
      {storageMessage && <div className="explorer-success" role="status"><Check size={15} />{storageMessage}</div>}
      {activeView === 'storage' && (storageSummary ? <>
        <div className="storage-grid">
          <article><small>METADATA</small><strong>{storageSummary.metadata_backend}</strong><span>{storageSummary.metadata_location}</span></article>
          <article><small>VECTOR STORE</small><strong>{storageSummary.vector_backend}</strong><span>{storageSummary.vector_persistence}</span><span className={storageSummary.vector_store_healthy ? 'text-good' : 'text-bad'}>{storageSummary.vector_store_healthy ? 'Connected' : 'Unavailable'}</span></article>
          <article><small>DOCUMENTS / SOURCE FILES</small><strong>{storageSummary.metadata_document_count} / {storageSummary.source_files_present}</strong><span>Records / source files present</span></article>
          <article><small>STORED VECTORS</small><strong>{storageSummary.vector_count ?? 'Unavailable'}</strong><span>Actual Chroma collection count</span></article>
        </div>
        <h3>Collections</h3>
        {storageSummary.collections.length ? storageSummary.collections.map((collection) => <div className="storage-row" key={collection.name}><code>{collection.name}</code><span>{collection.count} vectors</span></div>) : <p className="muted-copy">No Chroma collections were reported.</p>}
        <div className="storage-management">
          <div className="section-title">PERSISTENCE MANAGEMENT & CLEANUP</div>
          <p className="muted-copy">Clean up orphaned disk files in the uploads folder, clear vectors, or reset all stored documents.</p>
          <div className="storage-management-actions">
            <button className="button-secondary" onClick={() => void handleStorageCleanup()}>
              Cleanup orphaned files
            </button>
            <button className="button-danger-outline" onClick={() => void handleClearAllChunks()}>
              <Trash2 size={13} /> Clear all indexed chunks
            </button>
            <button className="button-danger" onClick={() => void handleDeleteAllDocuments()}>
              <Trash2 size={13} /> Delete all documents & source files
            </button>
          </div>
        </div>
        <p className="muted-copy" style={{ marginTop: '16px' }}>Last successful check {storageSummary.last_successful_check ? new Date(storageSummary.last_successful_check).toLocaleString() : 'not recorded'} · latest check {new Date(storageSummary.checked_at).toLocaleString()}</p>
        {storageSummary.error && <div className="explorer-error">{storageSummary.error}</div>}
      </> : <p className="muted-copy">Loading storage health…</p>)}
      {activeView === 'documents' && <>
        <label className="explorer-search"><Search size={15} /><input aria-label="Search documents" placeholder="Search filename or ID" value={documentSearch} onChange={(event) => setDocumentSearch(event.target.value)} /></label>
        <div className="document-explorer-grid">
          <div className="document-list">
            {documents.filter((item) => `${item.filename} ${item.document_id}`.toLowerCase().includes(documentSearch.toLowerCase())).map((item) => (
              <button key={item.document_id} className={selectedExplorerDoc === item.document_id ? 'selected' : ''} onClick={() => { setSelectedExplorerDoc(item.document_id); setDocumentPreview(null) }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                  <strong>{item.filename}</strong>
                </div>
                <code>{item.document_id}</code>
                <span>{item.status} · {item.chunk_count} chunks · {item.file_type.toUpperCase()}</span>
              </button>
            ))}
            {!documents.length && <p className="muted-copy">No persisted documents found.</p>}
          </div>
          {selectedExplorerDoc && (() => {
            const item = documents.find((doc) => doc.document_id === selectedExplorerDoc);
            if (!item) return null;
            return <article className="document-detail">
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span className={`status-pill ${item.status === 'ready' ? 'tested' : item.status === 'failed' ? 'planned' : 'experimental'}`}>{item.status}</span>
                  <strong>{item.filename}</strong>
                </div>
                <button
                  className="button-danger-outline"
                  onClick={() => void handleDeleteDocument(item.document_id, item.filename)}
                  title={`Delete ${item.filename} and remove source file`}
                >
                  <Trash2 size={12} /> Delete document
                </button>
              </div>
              <code>{item.document_id}</code>
              <span>Type {item.file_type} · uploaded {item.created_at ? new Date(item.created_at).toLocaleString() : 'timestamp unavailable'}</span>
              <span>Strategy {item.selection?.strategy_id ?? item.recommendation?.recommended_strategy_id ?? 'not selected'} · {item.chunk_count} stored chunks</span>
              {item.error && <p className="text-bad">{item.error}</p>}
              <div className="document-detail-actions">
                <button className="button-secondary" onClick={() => { void apiFetch(`/api/v1/documents/${item.document_id}/preview`).then(async (response) => { if (!response.ok) throw new Error('Stored source preview unavailable.'); setDocumentPreview(await response.json()) }).catch((error) => setStorageError(error instanceof Error ? error.message : 'Preview unavailable.')) }}>Preview extracted text</button>
                {(item.status === 'ready' || item.chunk_count > 0) && (
                  <button
                    className="button-danger-outline"
                    onClick={() => void handleDeleteDocumentChunks(item.document_id, item.filename)}
                    title="Delete chunks and vector embeddings for this document"
                  >
                    <Trash2 size={12} /> Clear indexed chunks
                  </button>
                )}
              </div>
              {documentPreview && <pre className="document-preview">{documentPreview.text}{documentPreview.truncated ? '\n\n[Preview truncated]' : ''}</pre>}
            </article>
          })()}
        </div>
      </>}
      {activeView === 'chunks' && <>
        <div className="chunk-filters">
          <label>Document filter
            <select aria-label="Filter chunks by document" value={selectedExplorerDoc ?? ''} onChange={(event) => { setSelectedExplorerDoc(event.target.value || null); setChunkOffset(0) }}>
              <option value="">All indexed documents</option>
              {documents.filter((item) => item.status === 'ready').map((item) => <option key={item.document_id} value={item.document_id}>{item.filename}</option>)}
            </select>
          </label>
          <label>Search this page
            <input aria-label="Search chunks on page" value={chunkFilter} onChange={(event) => setChunkFilter(event.target.value)} placeholder="Text or chunk ID" />
          </label>
        </div>
        <p className="muted-copy">{chunkPage ? `${chunkPage.total} stored chunks · page ${Math.floor(chunkOffset / 20) + 1}` : 'Loading persisted chunks…'}</p>
        <div className="chunk-explorer-list">
          {(chunkPage?.chunks ?? []).filter((chunk) => `${chunk.text} ${chunk.chunk_id} ${chunk.document_name}`.toLowerCase().includes(chunkFilter.toLowerCase())).map((chunk) => (
            <article className="stored-chunk" key={chunk.chunk_id}>
              <div>
                <strong>{chunk.document_name} · chunk {chunk.ordinal + 1}{chunk.page_number ? ` · page ${chunk.page_number}` : ''}</strong>
                <div className="chunk-actions">
                  <span>{chunk.strategy_id ?? 'strategy unavailable'} · {chunk.embedding_available ? `${chunk.embedding_dimension}-dimension vector` : 'vector unavailable'}</span>
                  <button
                    className="button-danger-outline"
                    style={{ height: '22px', padding: '0 6px', fontSize: '8px' }}
                    onClick={() => void handleDeleteSingleChunk(chunk.chunk_id)}
                    title="Delete this chunk from vector index"
                  >
                    <Trash2 size={11} /> Delete
                  </button>
                </div>
              </div>
              <code>{chunk.chunk_id}</code>
              {chunk.section_path.length > 0 && <small>{chunk.section_path.join(' / ')}</small>}
              <p>{chunk.text}</p>
            </article>
          ))}
        </div>
        <div className="page-controls">
          <button className="button-secondary" disabled={chunkOffset === 0} onClick={() => setChunkOffset(Math.max(0, chunkOffset - 20))}>Previous</button>
          <span>{chunkPage?.total ?? 0} chunks</span>
          <button className="button-secondary" disabled={!chunkPage || chunkOffset + 20 >= chunkPage.total} onClick={() => setChunkOffset(chunkOffset + 20)}>Next</button>
        </div>
      </>}
    </section>}
  </main>
}

export function VisualBuilder() {
  return (
    <ReactFlowProvider>
      <VisualBuilderContent />
    </ReactFlowProvider>
  )
}
