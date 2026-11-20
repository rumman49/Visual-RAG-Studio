# UX and Interaction Specification — Phase 0

**Status:** initial requirements with the Story 3 visual builder implemented; remaining Gallery, persistence, and execution interactions are future work.

Story 3 uses a dark three-column builder: searchable backend node palette, React Flow canvas, and schema-driven Inspector, with graph validation and a restrained execution-unavailable panel. Planned templates are draft-only. See `docs/epics/story-3-visual-builder-ui.md` for behavior and limits.

## Product shell

Professional accessible dark interface, with project/workflow identity and explicit backend/provider status. Main surfaces: Architecture Gallery, searchable categorized node palette, React Flow canvas, configuration inspector, strategy guidance, bottom query playground, and execution/trace inspection. Provide clear empty, loading, invalid, planned, unavailable-provider, and execution-error states.

## Gallery

Search names, aliases, descriptions and tags. Filter by category, capability/modality/language, implementation provider, and status. Cards disclose whether the item is a catalog concept, preview, or executable template. Detail view explains use, graph topology, required nodes/configuration/dependencies, constraints, limitations, verified languages/modalities, and actual provider/integration evidence. Planned entries can be previewed and read; run/instantiate-executable actions are unavailable. Traditional RAG is the complete loader-to-grounded-generation baseline; Naive RAG is a distinct planned direct top-k retrieve-to-generate pattern with minimal query/context processing. They may share nodes but do not appear as duplicate templates. Custom RAG creates a blank canvas.

## Editing and comparison

Instantiation creates a user-editable workflow copy. Preserve provenance to source template/version. Inspector renders component schema, ports and validation. Invalid connections show specific type/semantic reasons. Users can replace a component only when contracts pass, fork an alternative configuration for comparison, and save a version as a reusable template. Confirmation precedes destructive replacement of an existing branch; recommendation application never silently overwrites.

## Trust and clarity

Show real provider health and configured dependencies; do not represent test doubles as providers. Clearly separate known facts, unsupported combinations and unknown verification. Run view shows per-node state/results/errors and source citations with provenance; display scores/timings only when implementation emits them. Redact credentials and mark untrusted document content distinctly from instructions.

## Open UX decisions

Detailed canvas layout and breakpoints, design system, accessibility target, comparison presentation, and strategy-advisor interaction require a Phase 1 design pass and user visual reference (no image was present in the accessible attachments).
