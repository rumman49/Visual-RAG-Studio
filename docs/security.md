# Security Design — Initial Requirements

Security requirements are architectural constraints, not implementation claims.

- Treat uploaded documents, retrieved text, user prompts and model outputs as untrusted data; keep them separate from system instructions and constrain tool access.
- Strictly validate workflow documents and allow-listed node types. Never execute arbitrary Python, shell, package installation, SQL, or code embedded in graph imports.
- Store credentials outside graph JSON and traces; use secret references, access controls and redaction. Never silently use a different provider or mock.
- Enforce file type/size limits, safe storage, parser isolation, timeouts, query/index/execution budgets and bounded loops.
- Validate metadata filters and generated SQL against allow-listed schemas and read-only/parameterized policies before execution.
- Preserve citation provenance to indexed source spans; reject invented document/page citations and abstain when evidence policy fails.
- Track dependency versions, licenses and advisories; inspect provider adapters and serialized state formats before use.
- Define authentication, tenant isolation, retention, encryption, deletion, and deployment threat model in Phase 1 after deployment target is chosen.

For M1, the assumed operating mode is single-user and self-hosted. Development binding stays on loopback with authentication disabled and an explicit UI/API warning. Refuse or warn on non-loopback binding until an API-key/session boundary exists. This is not suitable for shared or public deployment; define a production threat model before expanding access.
