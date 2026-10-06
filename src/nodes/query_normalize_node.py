"""AgentCore Platform v1.0"""

# RET-C2-178 — QueryNormalizeNode
# Domain node 1: normalize the retail KB search query.
# Expands common retail acronyms (SKU, POS, SOH, etc.), strips runs of
# punctuation, and collapses whitespace so the normalized query is a single
# line — no newline survives into the rendered answer.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import logging
import re
from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# Retail acronym expansion map (common in Japanese retail / CVS context).
_RETAIL_ACRONYMS: Dict[str, str] = {
    r"\bSKU\b": "stock keeping unit",
    r"\bPOS\b": "point of sale",
    r"\bSOH\b": "stock on hand",
    r"\bSOP\b": "standard operating procedure",
    r"\bDC\b": "distribution center",
    r"\bFC\b": "fulfillment center",
    r"\bMD\b": "merchandise",
    r"\bVM\b": "visual merchandising",
    r"\bQC\b": "quality control",
    r"\bRTV\b": "return to vendor",
}

# Compiled acronym patterns (case-insensitive).
_COMPILED_ACRONYMS = [
    (re.compile(pattern, re.IGNORECASE), expansion) for pattern, expansion in _RETAIL_ACRONYMS.items()
]

# Strip runs of punctuation except Japanese punctuation and alphanumerics.
_EXCESS_PUNCT = re.compile(r"[!\"#%&'*+,\-/<=>@\[\\\]^_`{|}~]{2,}")


def _expand_acronyms(text: str) -> str:
    """Expand retail acronyms in the query text."""
    for pattern, expansion in _COMPILED_ACRONYMS:
        text = pattern.sub(expansion, text)
    return text


def _excess_punct_strip(text: str) -> str:
    """Replace runs of excess punctuation with a single space."""
    return _EXCESS_PUNCT.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run (newlines included) to one space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_query(text: str) -> str:
    """Apply the full normalization the pipeline uses, in order."""
    return _normalize_whitespace(_excess_punct_strip(_expand_acronyms(text.strip())))


class QueryNormalizeNode(FunctionNode):
    """Normalize the incoming KB search query for retrieval.

    Expands retail acronyms, sanitizes punctuation, and normalizes whitespace.
    Produces a clean single-line normalized_query suitable for term matching.

    required_trust_level = ANONYMOUS: inner domain node — trust check is
    handled upstream by the outer PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        user_input: the screened query handed to the inner graph

    Output state keys (partial dict):
        normalized_query: expanded and normalized query string
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        query: Any = state.get("validated_input") or state.get("user_input", "")
        query = query if isinstance(query, str) else ""

        if not query.strip():
            logger.warning("QueryNormalizeNode: query is empty")
            normalized_query = ""
        else:
            normalized_query = normalize_query(query)

        emit_trace_event(
            "kb_query_normalized",
            {"original_chars": len(query), "normalized_chars": len(normalized_query)},
            state,
        )

        logger.info("QueryNormalizeNode: original=%d chars, normalized=%d chars", len(query), len(normalized_query))

        return {"normalized_query": normalized_query}
