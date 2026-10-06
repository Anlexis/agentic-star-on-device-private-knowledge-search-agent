"""AgentCore Platform v1.0"""

# RET-C2-178 — PrivacyFilterNode
# Domain node 3: mask supplier-confidential and internal pricing signals in
# the ranked document chunks before the response is formatted.
#
# Removes supplier-confidential pricing, internal cost data and supplier
# identifiers from document text so they are never surfaced to a store-floor
# reader who is not authorized for that data. The patterns are anchored to a
# separator or a digit run so ordinary store vocabulary ("vending machine",
# "vendor", "cost centre") is left alone.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import logging
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Patterns in chunk text that indicate supplier-confidential / internal pricing data.
_CONFIDENTIAL_PATTERNS: List["re.Pattern[str]"] = [
    # Internal cost price references (e.g. "cost: ¥123", "wholesale: 450円").
    re.compile(r"(?:cost|wholesale|仕入れ|原価)\s*[:\s]\s*[¥￥]?\s*\d[\d,]*", re.IGNORECASE),
    # Supplier ID / vendor code values (e.g. "Supplier ID: S00123", "VEND-045").
    # A separator or a digit run is required after the label, so "vending" and
    # "vendor" as ordinary words never match.
    re.compile(r"\b(?:supplier\s*(?:id|code)|vendor\s*code)\s*[:#]\s*[A-Z0-9-]{3,}\b", re.IGNORECASE),
    re.compile(r"\bVEND-\d{2,}\b"),
    # Internal margin references (e.g. "margin: 32%", "利益率: 28%").
    re.compile(r"(?:margin|利益率|粗利)\s*[:：\s]\s*\d+\s*%", re.IGNORECASE),
]

_MASKED = "[CONFIDENTIAL]"


def _filter_chunk(chunk: str) -> str:
    """Replace supplier-confidential / pricing patterns in chunk text with [CONFIDENTIAL]."""
    for pattern in _CONFIDENTIAL_PATTERNS:
        chunk = pattern.sub(_MASKED, chunk)
    return chunk


def _filter_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the privacy filter to every result chunk; other fields pass through."""
    filtered = []
    for doc in results:
        filtered.append(
            {
                "doc_id": str(doc.get("doc_id", "")),
                "chunk": _filter_chunk(str(doc.get("chunk", ""))),
                "score": doc.get("score", 0.0),
                "category": str(doc.get("category") or ""),
            }
        )
    return filtered


class PrivacyFilterNode(FunctionNode):
    """Mask supplier-confidential data in the ranked results.

    required_trust_level = ANONYMOUS: inner domain node — trust check is
    handled upstream by the outer PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        search_results: JSON string of the ranked result chunks (from KBSearchNode)

    Output state keys (partial dict):
        filtered_results: JSON string of the masked result chunks
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw_results = from_json(state.get("search_results"), default=[])

        if not isinstance(raw_results, list):
            logger.warning(
                "PrivacyFilterNode: search_results is not a list (type=%s); using empty list",
                type(raw_results).__name__,
            )
            raw_results = []

        filtered = _filter_results(raw_results)
        redacted_count = sum(
            1 for orig, filt in zip(raw_results, filtered) if str(orig.get("chunk", "")) != filt["chunk"]
        )

        emit_trace_event(
            "kb_results_privacy_filtered",
            {"input_count": len(raw_results), "output_count": len(filtered), "chunks_redacted": redacted_count},
            state,
        )

        logger.info(
            "PrivacyFilterNode: %d results in, %d out, %d chunks redacted",
            len(raw_results),
            len(filtered),
            redacted_count,
        )

        return {"filtered_results": to_json(filtered)}
