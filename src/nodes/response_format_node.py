"""AgentCore Platform v1.0"""

# RET-C2-178 — ResponseFormatNode
# Domain node 4 (terminal): format the privacy-filtered KB search results
# into a structured Markdown response for the retail store manager.
#
# Renders the ranked results with document id, relevance and the masked
# excerpt, and says which corpus was searched. The normalized query is echoed
# on ONE line: it is whitespace-collapsed upstream, so no caller text can open
# a new line in the rendered answer and read as a retrieved step.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

logger = logging.getLogger(__name__)

_NO_RESULTS_MSG = (
    "No relevant KB documents were found for your query. "
    "Please try a more specific search term or contact your store manager."
)
_NO_TERMS_MSG = (
    "The query contained no searchable terms after screening, so no documents were ranked. "
    "Please rephrase the query using the terms you expect to find in the document."
)

_SOURCE_LABELS: Dict[str, str] = {
    "input_context": "the documents supplied with the request",
    "baseline": "the built-in baseline corpus (no documents were supplied)",
}


def _render_result(idx: int, doc: Dict[str, Any]) -> str:
    """Render a single search result as a Markdown section."""
    doc_id = str(doc.get("doc_id", "unknown"))
    chunk = str(doc.get("chunk", ""))
    score = doc.get("score", 0.0)
    score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
    category = str(doc.get("category") or "")

    category_tag = f" `[{category}]`" if category else ""
    return f"### {idx}. Document: `{doc_id}`{category_tag}\n**Relevance**: {score:.0%}\n\n{chunk.strip()}\n"


def _render_response(normalized_query: str, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    """Render the full Markdown KB response."""
    query_line = normalized_query or "(not available)"
    source = _SOURCE_LABELS.get(str(summary.get("document_source", "")), "the knowledge base")
    header = [
        "## KB Search Results\n",
        f"**Query**: {query_line}\n",
        f"**Searched**: {source}\n",
    ]

    if not summary.get("searchable_terms", 1):
        return "\n".join(header + [f"{_NO_TERMS_MSG}\n"])
    if not results:
        return "\n".join(header + [f"{_NO_RESULTS_MSG}\n"])

    lines = header + [f"**Results**: {len(results)} document(s) found\n", "---\n"]
    for idx, doc in enumerate(results, start=1):
        lines.append(_render_result(idx, doc))
    return "\n".join(lines)


class ResponseFormatNode(FunctionNode):
    """Format privacy-filtered KB results into a Markdown response.

    Terminal inner domain node. Reads filtered_results (JSON string),
    normalized_query and search_summary, renders a structured Markdown KB
    response for the retail store manager, and writes formatted_response.

    required_trust_level = ANONYMOUS: inner domain node — trust check is
    handled upstream by the outer PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        filtered_results: JSON string of the masked result chunks
        normalized_query: normalized query string (for display context)
        search_summary:   JSON string describing the search that ran

    Output state keys (partial dict):
        formatted_response: Markdown KB response string
        status:             AgentStatus.SUCCESS
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        filtered_results = from_json(state.get("filtered_results"), default=[])
        normalized_query = str(state.get("normalized_query") or "")
        summary = from_json(state.get("search_summary"), default={}) or {}

        if not isinstance(filtered_results, list):
            logger.warning(
                "ResponseFormatNode: filtered_results is not a list (type=%s)",
                type(filtered_results).__name__,
            )
            filtered_results = []
        if not isinstance(summary, dict):
            summary = {}

        formatted_response = _render_response(normalized_query, filtered_results, summary)

        emit_trace_event(
            "kb_response_formatted",
            {"result_count": len(filtered_results), "response_chars": len(formatted_response)},
            state,
        )

        logger.info(
            "ResponseFormatNode: %d results rendered, %d chars output", len(filtered_results), len(formatted_response)
        )

        return {"formatted_response": formatted_response, "status": AgentStatus.SUCCESS.value}
