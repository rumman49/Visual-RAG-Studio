---
status: final
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - docs/PRD.md
  - docs/architecture.md
  - docs/UX-specification.md
  - docs/security.md
  - docs/capability-registry.md
  - docs/api-contracts.md
  - _bmad-output/planning-artifacts/architecture/architecture-Visual-RAG-Studio-2026-11-21/ARCHITECTURE-SPINE.md
---

# Visual RAG Studio - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for Visual RAG Studio, decomposing the requirements from the PRD, UX Design, and Architecture requirements into implementable stories. Status notes mark what the codebase already delivers (as of commit ebc8ec6).

## Requirements Inventory

### Functional Requirements

FR1: Gallery search plus category, capability and status filters, with previews, requirements, limitations and provider status. (Partly built: registry and API exist; no Gallery UI.)
FR2: Instantiate a blank canvas or an executable template, preserving template identity and allowing edits. (Built.)
FR3: Add, connect, configure, move and remove components, validating compatibility before execution. (Built.)
FR4: Branch a workflow, compare configurations and replace compatible nodes. (Not built.)
FR5: Save and reopen versioned workflows, and publish a graph as a reusable template. (Partly built: `/workflows` stores one graph per record; no revisions or templates.)
FR6: Strictly validated import and export of graph documents, rejecting unknown nodes and code fields. (Not built.)
FR7: Execute supported graphs with explicit providers, bounded resources, and per-node outputs and errors. (Built for Traditional, Hybrid, Reranking, Multilingual and Agentic RAG.)
FR8: Preserve provenance and citation from source to answer. (Built.)
FR9: Local and cloud model adapters with no silent provider substitution. (Built: Ollama, Groq, Hugging Face, OpenAI, Gemini, Anthropic.)
FR10: Unimplemented catalog entries stay "Planned" and cannot run. (Built in the registry.)
FR11: Record provider and integration evidence per capability. (Partly built: in `docs/library-usage.md`, not the registry.)

### NonFunctional Requirements

NFR1: No arbitrary code, shell or SQL from graphs; allow-listed node types only.
NFR2: Secrets never appear in graphs, traces, storage or responses.
NFR3: Upload type and size limits, parser timeouts, and bounded agent loops and budgets.
NFR4: Loopback only until an authentication boundary exists.
NFR5: Untrusted document text is kept separate from instructions.
NFR6: Automated tests exist; live-provider evidence is reported separately from mocks.

### Additional Requirements

- AD-11 conflict: the hosted Space binds 0.0.0.0 and CORS allows any origin with credentials; needs an API-key boundary and a CORS allow-list.
- AD-9 review: vector-node `api_key` and `url` config can end up in graph JSON.
- Dockerfile: declare `gradio`, and decide the `space_app` entrypoint versus `app.main`.
- Pin `ragspark` consistently between `pyproject.toml` and the Dockerfile.
- Update `api-contracts.md`, `capability-registry.md` and `library-usage.md` to match the code.
- Refactor `/query` dispatch so architecture families extend the registry plus a runner rather than forking the route.

### UX Design Requirements

UX-DR1: Gallery with disclosure of concept, preview or executable status, and a detail view with topology.
UX-DR2: Run view with per-node state and citations; timings only when the implementation emits them.
UX-DR3: Fork a branch for comparison, with confirmation before destructive replacement.
UX-DR4: Redact credentials and visibly mark untrusted content.
UX-DR5: Clear empty, loading, invalid and provider-error states.

### FR Coverage Map

FR1: Epic 4 - Architecture picker with search, filters, status and topology preview
FR2: Baseline (built) - blank canvas and template instantiation
FR3: Baseline (built) - editor connect/configure/validate
FR4: Epic 3 - Branch, replace compatible nodes, compare
FR5: Epic 2 - Workflow revisions and reusable templates
FR6: Epic 2 - Strict graph import/export
FR7: Baseline (built); per-node trace in Epic 3, UI in Epic 5 - Execution with per-node outputs
FR8: Baseline (built); UI in Epic 5 - Provenance and citations
FR9: Baseline (built); live verification in Epic 1
FR10: Epic 4 - Planned entries never launch
FR11: Epic 4 - Provider evidence in the registry
NFR1, NFR3: Baseline (built); regression-guarded in Epic 6
NFR2, NFR4: Epic 1
NFR5: Epic 5
NFR6: Epic 1 (CI and live-provider evidence) and Epic 6
UX-DR1: Epic 4 · UX-DR2: Epic 5 · UX-DR3: Epic 3 · UX-DR4: Epic 5 · UX-DR5: Epic 2

Target user (decided in party mode): a developer who wants a deployable tool.

## Epic List

### Epic 1: Hardened, Verified Deployment
A developer can deploy the app to a server without exposing an open API or leaking keys, and knows the live providers actually work.
**FRs covered:** FR9 (live verification) · **NFRs:** NFR2, NFR4, NFR6 · **Additional:** CI pipeline, AD-9, AD-11, Dockerfile and ragspark fixes

### Epic 2: Save, Export and Run Workflows
A developer builds a graph, keeps it, exports it as a validated file, and calls it by ID from their own app.
**FRs covered:** FR5, FR6 · **UX-DR:** UX-DR5

### Epic 3: Branch and Compare Configurations
A developer forks a workflow, swaps compatible nodes, and compares runs on the same input.
**FRs covered:** FR4, FR7 (per-node trace) · **UX-DR:** UX-DR3

### Epic 4: Architecture Picker
A developer finds the architecture that fits their documents, with honest status and evidence.
**FRs covered:** FR1, FR10, FR11 · **UX-DR:** UX-DR1

### Epic 5: Trace and Trust UI
A developer sees per-node status, citations, and redacted secrets for every run.
**FRs covered:** FR7 (run view), FR8 (UI) · **NFR:** NFR5 · **UX-DR:** UX-DR2, UX-DR4

### Epic 6: Docs and Engineering Hygiene
Maintainers get docs that match the code and a `/query` that doesn't fork per architecture.
**NFRs:** NFR1, NFR3, NFR6 · **Additional:** docs refresh, dispatch refactor, epics history

## Baseline (already built, not planned)

FR2, FR3, FR7, FR8, FR9: blank canvas and templates, editor and validation, execution for Traditional/Hybrid/Reranking/Multilingual/Agentic RAG, provenance, and six provider adapters. Details are in `docs/epics/story-1` to `story-10`.

## Epic 1: Hardened, Verified Deployment

A developer can deploy the app to a server without exposing an open API or leaking keys, and knows the live providers actually work.

### Story 1.1: CI pipeline

As a maintainer,
I want tests and builds to run automatically on every push,
So that regressions are caught before they merge.

**Acceptance Criteria:**

**Given** a push or pull request
**When** CI runs
**Then** it runs `uv run pytest`, `uv run ruff check`, and the frontend `npm test` and `npm run build`
**And** any test marked `live_providers` is excluded by default and run only when opted in

**Given** a failure
**When** CI finishes
**Then** the run is red and the failing step is named

**Given** CI
**When** it runs on Linux
**Then** the Windows-specific test-teardown fixes (Story 7 report) don't cause failures or stalls

### Story 1.2: API key required for non-loopback access

As a developer deploying the backend,
I want every `/api/v1` request to require an API key when the server isn't on loopback,
So that a hosted instance isn't an open, unauthenticated API.

**Acceptance Criteria:**

**Given** `VRS_API_KEY` is set and the app runs in production mode
**When** a request to `/api/v1/*` has no key, or a wrong key
**Then** the response is 401 with `{detail:{code,message}}`
**And** `/api/health/live` stays open

**Given** a request carrying `X-API-Key` that matches `VRS_API_KEY`
**When** it reaches `/api/v1/*`
**Then** it is accepted

**Given** the frontend talks to an instance that requires a key
**When** the user opens the app
**Then** the UI asks for the key, holds it in session memory only (never `localStorage` or a `VITE_` variable), and sends it as `X-API-Key` on every request (AD-12)
**And** a 401 shows a clear "key required or invalid" state with a retry

**Given** loopback development with no `VRS_API_KEY`
**When** the app starts
**Then** it runs without a key and the UI shows the existing "no auth" warning

**Given** production mode with no `VRS_API_KEY` set
**When** the app starts
**Then** it refuses to start and names the missing setting
**And** the key never appears in logs, responses or traces

### Story 1.3: CORS limited to configured origins

As a developer,
I want CORS restricted to an allow-list from config,
So that other sites can't call my instance with a user's credentials.

**Acceptance Criteria:**

**Given** `VRS_CORS_ORIGINS` lists my Vercel URL
**When** a browser on that origin calls the API
**Then** the request succeeds

**Given** an origin not on the list
**When** it makes a preflighted request
**Then** it is rejected
**And** `allow_origin_regex=^https?://.*` is removed

**Given** local development
**When** no list is set
**Then** `127.0.0.1` and `localhost` on port 5173 are allowed

### Story 1.4: Secrets kept out of workflow graphs

As a developer,
I want secrets in vector-store config handled outside the graph,
So that exported or saved graphs never contain credentials.

**Acceptance Criteria:**

**Given** a `vector.index` node config contains `api_key`
**When** the graph is validated or saved
**Then** validation rejects it with an actionable issue pointing to server-side credentials (AD-9)

**Given** a graph has a secret reference instead
**When** `/query` runs
**Then** the key is resolved from `RuntimeCredentials` or env
**And** no response, trace or saved workflow contains the key

### Story 1.5: Reproducible hosted image

As a developer deploying to Hugging Face Spaces,
I want the Space to build and start from one documented entrypoint,
So that deployment doesn't depend on undeclared packages.

**Acceptance Criteria:**

**Given** the repo's Dockerfile
**When** the image is built
**Then** every import used by the chosen entrypoint is declared (including `gradio` if `space_app.py` is used)
**And** `ragspark` is pinned to the same commit in `pyproject.toml` and the Dockerfile

**Given** the container starts
**When** `/api/health/live` is requested on port 7860
**Then** it returns `alive`
**And** `README.md` documents the deployment steps and required env vars

### Story 1.6: Verified live providers

As a developer,
I want recorded evidence that Ollama, Groq and Hugging Face generation work,
So that I can trust provider status in the registry.

**Acceptance Criteria:**

**Given** opt-in env vars (keys and models) are set
**When** `pytest -m live_providers` runs
**Then** each reachable provider completes a minimal prompt and one grounded RAG query against a fixture document

**Given** a provider is not configured
**When** the live tests run
**Then** they are skipped and reported separately, never passed as mocks

**Given** results
**When** the run finishes
**Then** provider status and verification date are recorded in the registry and a story report

## Epic 2: Save, Export and Run Workflows

A developer builds a graph, keeps it, exports it as a validated file, and calls it by ID from their own app.

### Story 2.1: Workflow revisions

As a developer,
I want each save to create a new revision,
So that I can reopen or restore earlier versions of a graph.

**Acceptance Criteria:**

**Given** an existing saved workflow
**When** I save changes
**Then** a new revision is stored with a number and timestamp, and the previous one is kept
**And** the workflow's `updated_at` reflects the latest revision

**Given** a workflow with several revisions
**When** I list its revisions or open one by number
**Then** I get the exact graph saved at that revision, and the latest is the default

**Given** a workflow saved before this change
**When** the app starts
**Then** it appears as revision 1 with no data loss

### Story 2.2: Save and reopen in the editor

As a developer,
I want to save, list and reopen workflows from the builder,
So that I don't rebuild graphs every session.

**Acceptance Criteria:**

**Given** a graph on the canvas
**When** I click Save
**Then** it persists and the UI confirms the name and revision, with no canvas coordinates in the payload beyond what the contract allows

**Given** saved workflows exist
**When** I open the workflow list
**Then** I can reopen one, and the canvas, configs and provider references match what I saved

**Given** the backend is unreachable or returns an error
**When** I save or open
**Then** the UI shows an error state and my unsaved canvas is unchanged (UX-DR5)

### Story 2.3: Export a graph as a validated file

As a developer,
I want to export a workflow as a declarative JSON document,
So that I can version it in git and deploy it elsewhere.

**Acceptance Criteria:**

**Given** a valid graph
**When** I export
**Then** I get a versioned JSON file containing only node IDs, `node_type_id`, config, provider references and edges
**And** it contains no secrets, canvas coordinates or UI-only data

**Given** a graph that is structurally invalid
**When** I export
**Then** the export is blocked with the validation issues listed

### Story 2.4: Import a graph with strict validation

As a developer,
I want to import a workflow file and have unsafe content rejected,
So that I can trust files from other people.

**Acceptance Criteria:**

**Given** a valid exported file
**When** I import it
**Then** the graph loads on the canvas and passes the same validation as a hand-built graph

**Given** a file with an unknown node type, an invalid schema, or an executable-code field (NFR1)
**When** I import it
**Then** it is rejected with a specific reason, and nothing is added to the canvas or saved

**Given** a file whose schema version is older
**When** I import it
**Then** it is migrated, or rejected with the version it needs

### Story 2.5: Publish a workflow as a reusable template

As a developer,
I want to turn a saved workflow into a template,
So that I can start new projects from it.

**Acceptance Criteria:**

**Given** a saved, valid workflow
**When** I publish it as a template with a name
**Then** it appears in the template picker with its source workflow and revision retained

**Given** a template
**When** I instantiate it
**Then** I get an editable copy, and the template itself is unchanged

**Given** a template that fails current validation
**When** I try to publish or instantiate it
**Then** the action is blocked and the issues are shown

### Story 2.6: Run a saved workflow by ID

As a developer,
I want to call a saved workflow over the API,
So that my own application can use it without the UI.

**Acceptance Criteria:**

**Given** a saved workflow
**When** I `POST /api/v1/workflows/{id}/runs` with a question (and optional revision)
**Then** it validates the graph, runs it, and returns the answer, citations and execution ID
**And** it requires the API key from Story 1.2

**Given** an unknown workflow ID or revision
**When** I call it
**Then** the response is 404 with `{detail:{code,message}}`

**Given** a graph that is not executable
**When** I call it
**Then** the response is 422 with the validation issues, and no model is invoked

### Story 2.7a: Download a Traditional RAG project as a Python zip

As a developer,
I want to download a zip of standalone Python code for my Traditional RAG workflow,
So that I can run it in my own app without Visual RAG Studio.

**Acceptance Criteria:**

**Given** a valid, executable Traditional RAG workflow
**When** I click Download code
**Then** I get a `.zip` with one module per stage (load, chunk, embed and index, retrieve, context, generate), a `workflow.py` that wires them in graph order, the graph's config as settings, `requirements.txt`, `README.md` and `.env.example`
**And** the project does not import any `app.*` module from Visual RAG Studio

**Given** any graph
**When** the zip is generated
**Then** it contains no API keys or tokens, only `.env.example` placeholders (NFR2, AD-9)
**And** the code is rendered from fixed server-side templates, never from code strings in the graph (NFR1)

**Given** a planned, invalid or non-executable graph
**When** I request the zip
**Then** the request is rejected with the validation issues and no file is produced

**Given** a generated Traditional RAG project
**When** it runs in a clean virtualenv against a sample document
**Then** it ingests the document and answers with citations (checked in CI)

### Story 2.7b: Hybrid, Reranking and Multilingual code in the zip

As a developer,
I want the zip to include the retrieval variants my workflow uses,
So that the generated project matches my chosen architecture.

**Acceptance Criteria:**

**Given** a workflow with BM25 and fusion nodes
**When** I download the code
**Then** the zip includes standalone BM25 and RRF (or weighted) fusion with the graph's `k1`, `b`, `rrf_k` and weights

**Given** a workflow with a rerank node
**When** I download the code
**Then** the zip includes the reranking stage with its configured top-k

**Given** a multilingual workflow
**When** I download the code
**Then** the zip includes language detection, the configured multilingual embedding model, and localized abstention messages

**Given** each of these architectures
**When** the generated project runs in a clean virtualenv against a fixture
**Then** it returns cited answers, and CI verifies this for each architecture

### Story 2.7c: Agentic RAG code in the zip

As a developer,
I want the zip to include the bounded agent loop,
So that an Agentic RAG workflow runs outside the studio with the same limits.

**Acceptance Criteria:**

**Given** a workflow with an `agent.orchestrate` node
**When** I download the code
**Then** the zip includes the planner, the `allowed_tools` allow-list, and the `max_steps`, `max_retrieval_iterations`, `token_budget` and `timeout_seconds` limits from the node config (AD-10)

**Given** the generated agent runs
**When** a limit is reached or evidence is insufficient
**Then** it stops or abstains and prints a structured step trace

**Given** the generated agent project
**When** it runs in a clean virtualenv against a fixture
**Then** it returns a cited answer or an abstention, and CI verifies this

## Epic 3: Branch and Compare Configurations

A developer forks a workflow, swaps compatible nodes, and compares runs on the same input.

### Story 3.1: Per-node trace in query responses

As a developer,
I want each query response to include a per-node trace,
So that I can see what every step did and where it failed.

**Acceptance Criteria:**

**Given** a Traditional, Hybrid, Reranking or Multilingual run
**When** it completes, fails or abstains
**Then** the response includes an ordered `trace` with node ID, node type, status, and the timing only where the implementation measures it

**Given** a node fails
**When** the run ends
**Then** the failing node, its error code and message appear in the trace, and later nodes are marked skipped

**Given** the trace
**When** it is returned or stored (Story 3.2)
**Then** it contains no secrets or full document text

### Story 3.2: Persist run history

As a developer,
I want each run's question, answer, citations and node trace saved,
So that I can reopen and compare results later.

**Acceptance Criteria:**

**Given** a workflow run (UI or `/workflows/{id}/runs`)
**When** it finishes, fails or abstains
**Then** a run record is stored with workflow ID and revision, question, status, answer, citations, the per-node trace from Story 3.1, provider and model, and timings the implementation emits

**Given** the run record
**When** it is stored or returned
**Then** it contains no secrets, and a hosted provider's use is marked as externally processed

**Given** run history
**When** I list runs for a workflow
**Then** they're paged, newest first, and I can delete one or all of them

### Story 3.3: Fork a workflow branch

As a developer,
I want to fork a workflow to try a different configuration,
So that I can experiment without losing the original.

**Acceptance Criteria:**

**Given** a saved workflow
**When** I fork it with a branch name
**Then** a new editable workflow is created from the chosen revision, linked to its parent and revision

**Given** a fork
**When** I edit and save it
**Then** the parent is unchanged

**Given** a workflow with branches
**When** I list it
**Then** I see the parent/branch relationship

### Story 3.4: Replace a node with a compatible one

As a developer,
I want to swap a node for another that fits the same ports,
So that I can try a different chunker, retriever or provider without rewiring.

**Acceptance Criteria:**

**Given** a node on the canvas
**When** I choose Replace
**Then** only node types whose ports and config pass the backend contract check are offered

**Given** a compatible replacement
**When** I confirm
**Then** connections are preserved and the node's config resets or maps with the differences shown
**And** a confirmation precedes the destructive replacement (UX-DR3)

**Given** an incompatible choice
**When** it is attempted
**Then** it is blocked with a specific type or semantic reason

### Story 3.5: Run a comparison on the same input

As a developer,
I want to run two branches on the same question and documents,
So that I can see which configuration works better.

**Acceptance Criteria:**

**Given** two workflows or branches
**When** I run a comparison with one question and a set of document IDs
**Then** each runs once and stores a run record, linked as a comparison

**Given** one branch fails or abstains
**When** the comparison finishes
**Then** the other result is still shown and the failure reason is explicit

**Given** an embedding model or dimension differs between branches
**When** they share one collection
**Then** the run is rejected with the AD-6 reason, and it proposes separate collections

### Story 3.6: Side-by-side comparison view

As a developer,
I want a view of two runs next to each other,
So that I can judge answers, citations and cost.

**Acceptance Criteria:**

**Given** a comparison
**When** I open it
**Then** I see both answers, citations, per-node status, and the config differences between the branches

**Given** timings or token counts were emitted
**When** they're shown
**Then** they're labeled as measured, and absent values are shown as unavailable, never zero

**Given** a loading, empty or error state
**When** it occurs
**Then** the view shows an explicit state (UX-DR5)

## Epic 4: Architecture Picker

A developer finds the architecture that fits their documents, with honest status and evidence.

### Story 4.1: Evidence recorded in the registry

As a developer,
I want each capability's provider, version, date and result recorded in the registry,
So that status claims are backed by data I can read.

**Acceptance Criteria:**

**Given** the five implemented architecture families and their node types
**When** the registry loads
**Then** each carries evidence entries (what was verified, how, date, test or story reference) populated from the story reports and Epic 1's live runs

**Given** an entry marked `tested`
**When** it has no evidence
**Then** registry validation fails, as `contracts.py` already requires

**Given** an entry with only mocked-provider evidence
**When** its status is computed
**Then** it can't be `tested`, and the evidence says "mock"

### Story 4.2: Browse and filter the catalog

As a developer,
I want to search and filter all 37 architectures,
So that I can narrow to ones relevant to my project.

**Acceptance Criteria:**

**Given** the registry loads
**When** I open the picker
**Then** I see all 37 entries from `/api/v1/catalog`, with no copy of the catalog in the frontend

**Given** a search term or category, status or capability filter
**When** I apply it
**Then** the list narrows accordingly, and an empty result shows an explicit empty state (UX-DR5)

**Given** the catalog fails to load
**When** the picker opens
**Then** it shows an error state with a retry

### Story 4.3: Architecture detail and topology preview

As a developer,
I want to read how an architecture works and see its graph,
So that I understand it before building with it.

**Acceptance Criteria:**

**Given** any catalog entry
**When** I open its detail
**Then** I see purpose, required nodes and config, dependencies, limitations, verified languages and modalities, and its evidence, with its status labeled as concept, preview or executable (UX-DR1)

**Given** an entry with a template graph
**When** I open its detail
**Then** a read-only topology preview renders from the registry's graph

**Given** a planned entry
**When** I view it
**Then** run and instantiate actions are unavailable, and the reason is shown (FR10)

### Story 4.4: Start a workflow from the picker

As a developer,
I want to open an executable architecture as an editable workflow,
So that I can configure and run it.

**Acceptance Criteria:**

**Given** an executable architecture
**When** I choose Use this
**Then** the editor loads its template graph with the source template and version retained

**Given** the blank "Custom RAG" entry
**When** I choose it
**Then** I get an empty canvas

**Given** a planned entry
**When** I try to instantiate it (including by direct API request)
**Then** it is refused and no graph is created

### Story 4.5: Recommend an architecture for my documents

As a developer,
I want a recommendation based on my documents and needs,
So that I don't have to know which of 37 patterns to choose.

**Acceptance Criteria:**

**Given** an uploaded document with a profile (language, structure, length)
**When** I ask for a recommendation
**Then** the picker suggests only executable architectures, each with the evidence-based reason (e.g. multilingual content points to Multilingual RAG)

**Given** a recommendation
**When** it's shown
**Then** it is advisory only, and nothing is applied until I choose it

**Given** the document profile is unavailable
**When** I ask
**Then** the picker says so and offers the unfiltered list, with no guessed result

## Epic 5: Trace and Trust UI

A developer sees per-node status, citations, and redacted secrets for every run.

### Story 5.1: Run view with per-node status

As a developer,
I want to see each node's state and result after a run,
So that I can debug a pipeline visually.

**Acceptance Criteria:**

**Given** a run
**When** it finishes
**Then** the canvas shows each node as completed, failed or skipped, using the trace, and selecting a node shows its summary and any error

**Given** a run in progress
**When** I watch the canvas
**Then** node states update from `/executions/{id}`

**Given** no run exists yet, a loading state, or an error
**When** the view renders
**Then** it shows an explicit state (UX-DR5), and absent timings read "not measured" (UX-DR2)

### Story 5.2: Citations with source provenance

As a developer,
I want each citation to show its document, page and exact excerpt,
So that I can check the answer against the source.

**Acceptance Criteria:**

**Given** an answer with citations `[1]`, `[2]`
**When** I select one
**Then** I see the document name, page or section, chunk ID and the retrieved excerpt

**Given** a citation ID that doesn't resolve to a retrieved chunk
**When** the answer renders
**Then** it is flagged as invalid and not shown as a source (AD-7)

**Given** an abstained answer
**When** it renders
**Then** it is labeled as an abstention with no fabricated sources

### Story 5.3: Agent step trace view

As a developer,
I want to see the agent's planning, retrieval, rerank and evidence steps,
So that I can understand why it answered or stopped.

**Acceptance Criteria:**

**Given** an agent run
**When** it finishes
**Then** I see each step (planning, retrieval, rerank, evaluate_evidence, reformulate_query, answer_generation, abstain) with its outcome

**Given** a budget or timeout was hit
**When** the run ends
**Then** the view shows which limit stopped it and the usage against the budget

**Given** an agent run
**When** I read the trace
**Then** `tokens_used`, `step_count` and `retrieval_iterations` match the response

### Story 5.4: Redaction and external-processing indicators

As a developer,
I want secrets hidden and hosted-provider use clearly flagged,
So that I never leak keys or send documents out by accident.

**Acceptance Criteria:**

**Given** any run view, trace, saved run or export
**When** it is rendered or generated
**Then** no API key or token appears (UX-DR4, NFR2)

**Given** a run used a hosted provider (Groq, Hugging Face, OpenAI, Gemini or Anthropic)
**When** I view it
**Then** it is marked as sent externally, naming the provider

**Given** a provider isn't configured or tested
**When** I try to run
**Then** the UI shows the actual status and blocks the run with the reason

### Story 5.5: Untrusted content kept visibly separate

As a developer,
I want retrieved document text distinguished from instructions,
So that prompt-injection attempts in documents are easy to spot.

**Acceptance Criteria:**

**Given** retrieved excerpts and context previews
**When** they are shown
**Then** they're rendered as inert text (no HTML or markdown execution) with a distinct "document content" style (NFR5, UX-DR4)

**Given** a fixture document containing "ignore previous instructions"
**When** it is retrieved and answered
**Then** the pipeline keeps it in the context block, and a regression test confirms it doesn't change the system prompt

## Epic 6: Docs and Engineering Hygiene

Maintainers get docs that match the code, a `/query` that doesn't fork per architecture, and checks that keep it that way.

### Story 6.1: One `/query` dispatch path

As a maintainer,
I want `/query` to dispatch through a runner registry,
So that adding an architecture family doesn't mean editing the route.

**Acceptance Criteria:**

**Given** `/query` today (agent branch, then the Traditional/Hybrid path, with provider selection inline in the route)
**When** it is refactored
**Then** the route resolves a runner by graph shape or architecture ID, and provider and embedding selection move to dependency helpers (AD-4)

**Given** the existing regression suite (153 tests)
**When** the refactor lands
**Then** every test still passes with no behaviour change

**Given** a graph no runner supports
**When** `/query` is called
**Then** it returns 422 `workflow_not_executable` as it does now

### Story 6.2: Guards for the security baseline

As a maintainer,
I want automated checks for the security rules,
So that NFR1 and NFR3 can't regress quietly.

**Acceptance Criteria:**

**Given** a graph with an unknown node type or a code-like field
**When** it is validated or imported
**Then** it is rejected (test)

**Given** an upload over the size limit or of an unsupported type
**When** it is posted
**Then** it is rejected with the existing error code (test)

**Given** an agent with small `max_steps`, `token_budget` or `timeout_seconds`
**When** it runs
**Then** it stops at the limit (test)

### Story 6.3: Docs match the code

As a maintainer,
I want the planning and API docs to match what is built,
So that new contributors and agents aren't misled.

**Acceptance Criteria:**

**Given** `docs/api-contracts.md`
**When** it is refreshed
**Then** it lists every route in `routes.py` and `main.py`, with planned items separated

**Given** `PRD.md`, `capability-registry.md`, `UX-specification.md` and `library-usage.md`
**When** they are refreshed
**Then** their status headers state what is built, and "Phase 0, nothing implemented" wording is gone

**Given** the Story 6 report's claim that queries return a trace
**When** the docs are reviewed
**Then** the claim is corrected or points to Story 3.1

### Story 6.4: Reconcile RagSpark status

As a maintainer,
I want one clear answer on whether RagSpark is integrated,
So that docs, dependencies and tests agree.

**Acceptance Criteria:**

**Given** `pyproject.toml`, `ragspark_adapters.py`, the reranker and the Story 7 report
**When** I audit them
**Then** the actual integration state is recorded in `docs/library-usage.md`, the registry and the epics history

**Given** RagSpark on Python 3.12 (Docker) and 3.14 (local)
**When** the integration tests run
**Then** results for both are recorded, or the unsupported version is documented

### Story 6.5: Consolidate epic history

As a maintainer,
I want the old Story 1–10 reports folded into one numbered history,
So that there's one source of truth for planning.

**Acceptance Criteria:**

**Given** `docs/epics/` with two "Epic 7" headings and two `story-7-*` files
**When** it is consolidated
**Then** each historical story has a unique ID and an index, with content preserved

**Given** `docs/epics/README.md`
**When** it is updated
**Then** it links to the new `epics.md` for planned work and to the history for delivered work
