"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic model. Checkpoints are
# serialized with msgpack, and Pydantic objects corrupt silently. Extend
# AgentState with agent-specific fields only. Do NOT add credentials, secrets,
# or Pydantic models.
#
# Structured fields (dict / list[dict]) are stored as JSON STRINGS, not bare
# Python containers — a bare container in a checkpointed State field is a
# state-safety violation. Producers serialize with to_json() on write;
# consumers deserialize with from_json() on read.
#
# RET-C2-178 — Retail On-Device Private KB Search Agent.
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph)
# + inner domain workflow (BaseGraph). Fields below cover both layers.
#
# Privacy note: raw store-staff identifiers are never persisted to State.
# PreProcessNode strips them from the query and from every caller document
# before anything is written. PrivacyFilterNode removes supplier-confidential
# signals before results are persisted. No credential or personal-data field
# is ever checkpointed.

import json
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (checkpoint safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for RET-C2-178.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode / KBSearchGraphNode.merge_output
    # ------------------------------------------------------------------

    # Screened, personal-data-stripped search query produced by
    # PreProcessNode. Raw user input is NOT persisted beyond this.
    validated_input: Optional[str]

    # JSON STRING (to_json) of the validated caller contract: the documents to
    # search, the search tuning and the category filter. Written by
    # PreProcessNode; bridged into the inner graph by KBSearchGraphNode.
    caller_contract: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # JSON STRING (to_json) of the live search defaults from config/config.yaml,
    # seeded by DomainWorkflowGraph._extra_initial_state(). Node execute()
    # methods take no config argument, so state is the only route a declared
    # value can travel into a domain node.
    search_config: Optional[str]

    # QueryNormalizeNode output.
    # Normalized query string: retail acronyms expanded, punctuation sanitized,
    # whitespace collapsed to single spaces (so no newline survives).
    normalized_query: Optional[str]

    # KBSearchNode output.
    # JSON STRING (to_json) of the ranked document chunks:
    # [{"doc_id": str, "chunk": str, "score": float, "category": str}]
    # Consumers read via from_json().
    search_results: Optional[str]

    # KBSearchNode output.
    # JSON STRING (to_json) describing the search that ran: the number of
    # searchable terms, the corpus size and source, and the effective tuning.
    search_summary: Optional[str]

    # PrivacyFilterNode output.
    # JSON STRING (to_json) of search_results with supplier-confidential
    # and internal pricing signals masked in the chunk text.
    filtered_results: Optional[str]

    # ResponseFormatNode output (terminal inner node).
    # Final Markdown KB response formatted for the retail store manager.
    # Surfaced via merge_output() and gated by PostProcessNode.
    formatted_response: Optional[str]

    # Alias written by KBSearchGraphNode.merge_output() so PostProcessNode
    # can read state["result"] in the standard backbone pattern.
    result: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
