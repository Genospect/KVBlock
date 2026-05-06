"""vLLM integration planning stub.

KVBlock should not depend on vLLM internals in the core package. A future adapter
can observe block tables, map logical ids to physical pages, and pass selected
page ids to a backend that truly skips unselected KV reads.
"""

from __future__ import annotations


class VLLMIntegrationStub:
    """Non-invasive placeholder for future vLLM page-table integration."""

    name = "vllm"
