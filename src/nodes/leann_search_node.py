"""AgentCore Platform v1.0"""

# RET-C2-178 — KBSearchNode
# Domain node 2: rank the store's knowledge-base documents against the query.
#
# The documents come from the validated caller contract (input_context.documents),
# so the search runs over the store's own knowledge base. When the caller sends
# none, the built-in baseline corpus below is searched instead — a working
# sample, not a stand-in for a real answer, and the response says which one
# was used. An on-device vector index would slot in here in place of the
# term-overlap ranking; nothing above or below this node depends on how the
# ranking is produced.
#
# Scoring: a document's score is the fraction of the query's search terms that
# occur in its text (0.0 to 1.0). Search terms are the query's distinct words of
# three or more characters after normalization, minus a short stop-word list
# and minus the platform's masking sentinels. A query with no search terms left
# — for example one the platform masked entirely — returns no documents and is
# reported as such, rather than as a confident list ranked on nothing.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import logging
import re
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json
from src.services.caller_contract import MAX_TOP_K

logger = logging.getLogger(__name__)

# Defaults used only when neither the caller nor config/config.yaml declares
# a value (the config loader could not read the file, for example).
_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.2

# Baseline corpus, searched when the caller sends no documents.
BASELINE_KB_DOCS: List[Dict[str, Any]] = [
    {
        "doc_id": "SOP-001",
        "chunk": (
            "Store Opening Procedure: Unlock the front door at 07:45, "
            "turn on all lighting systems, and verify the POS terminal is online."
        ),
        "category": "operations",
    },
    {
        "doc_id": "SOP-002",
        "chunk": (
            "Inventory Management: Perform daily stock on hand (SOH) verification "
            "for high-velocity SKUs. Record discrepancies in the inventory log."
        ),
        "category": "inventory",
    },
    {
        "doc_id": "INC-101",
        "chunk": (
            "Incident Report Template: Record date, time, location, staff ID "
            "(anonymized), and incident description. Escalate to district manager "
            "if severity >= MEDIUM."
        ),
        "category": "incident",
    },
    {
        "doc_id": "PROD-045",
        "chunk": (
            "Product Specification — Private Label Beverages: "
            "Shelf life 12 months. Storage temperature 5-25°C. "
            "Reorder point 48 units."
        ),
        "category": "product_spec",
    },
    {
        "doc_id": "SUPP-012",
        "chunk": (
            "Supplier Communication Protocol: Submit purchase orders via the "
            "supplier portal 5 business days before required delivery date."
        ),
        "category": "supplier",
    },
]

_TERM_RE = re.compile(r"[a-z0-9]{3,}")

# Words that carry no retrieval signal in a store-operations query.
_STOP_WORDS: FrozenSet[str] = frozenset(
    {
        "the",
        "for",
        "are",
        "what",
        "how",
        "and",
        "with",
        "that",
        "this",
        "from",
        "our",
        "your",
        "which",
        "when",
        "where",
        "who",
        "does",
        "can",
        "should",
        "into",
        "about",
        "have",
        "has",
        "was",
        "were",
        "will",
        "would",
        "there",
        "their",
        "then",
        "than",
        "them",
        "they",
        "you",
        "all",
        "any",
        "each",
        "per",
        "not",
        "but",
        "its",
        "please",
        "tell",
        "steps",
        "step",
    }
)

# The platform replaces personal-data shapes in the query with a sentinel, and
# the request boundary replaces store-staff shapes with another. Neither is a
# search term: a fully masked query must not rank documents on the word
# "masked".
_MASK_SENTINELS: FrozenSet[str] = frozenset({"masked", "redacted"})


def search_terms(normalized_query: str) -> List[str]:
    """The distinct search terms of a normalized query, in first-seen order."""
    seen: List[str] = []
    for term in _TERM_RE.findall(normalized_query.lower()):
        if term in _STOP_WORDS or term in _MASK_SENTINELS or term in seen:
            continue
        seen.append(term)
    return seen


def score_document(terms: List[str], chunk: str) -> float:
    """Fraction of the search terms that occur in the chunk text."""
    if not terms:
        return 0.0
    haystack = chunk.lower()
    matched = sum(1 for term in terms if term in haystack)
    return round(matched / len(terms), 4)


def rank_documents(
    normalized_query: str,
    documents: List[Dict[str, Any]],
    top_k: int,
    min_score: float,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rank *documents* against the query; return at most top_k at or above min_score."""
    terms = search_terms(normalized_query)
    if not terms:
        return []
    ranked: List[Dict[str, Any]] = []
    for doc in documents:
        doc_category = str(doc.get("category") or "")
        if category and doc_category != category:
            continue
        score = score_document(terms, str(doc.get("chunk", "")))
        if score < min_score:
            continue
        ranked.append(
            {
                "doc_id": str(doc.get("doc_id", "")),
                "chunk": str(doc.get("chunk", "")),
                "score": score,
                "category": doc_category,
            }
        )
    ranked.sort(key=lambda d: (-d["score"], d["doc_id"]))
    return ranked[:top_k]


class KBSearchNode(FunctionNode):
    """Rank the knowledge-base documents against the normalized query.

    required_trust_level = ANONYMOUS: inner domain node — trust check is
    handled upstream by the outer PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        normalized_query: expanded and normalized query string (from QueryNormalizeNode)
        caller_contract:  JSON string — documents, top_k, min_score, category
        search_config:    JSON string — the live defaults from config/config.yaml

    Output state keys (partial dict):
        search_results: JSON string of the ranked result chunks
        search_summary: JSON string describing the search that ran
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        normalized_query = state.get("normalized_query") or ""
        contract = from_json(state.get("caller_contract"), default={}) or {}
        config = from_json(state.get("search_config"), default={}) or {}

        documents = contract.get("documents") or []
        document_source = "input_context" if documents else "baseline"
        if not documents:
            documents = BASELINE_KB_DOCS

        # The caller's bounded value wins; otherwise the declared default;
        # otherwise the code default. Config values are re-bounded here so a
        # malformed config file cannot widen the contract.
        top_k = contract.get("top_k")
        if top_k is None:
            top_k = config.get("top_k", _DEFAULT_TOP_K)
        top_k = (
            max(1, min(int(top_k), MAX_TOP_K))
            if isinstance(top_k, (int, float)) and not isinstance(top_k, bool)
            else _DEFAULT_TOP_K
        )

        min_score = contract.get("min_score")
        if min_score is None:
            min_score = config.get("min_score", _DEFAULT_MIN_SCORE)
        min_score = (
            float(min_score)
            if isinstance(min_score, (int, float)) and not isinstance(min_score, bool)
            else _DEFAULT_MIN_SCORE
        )
        min_score = max(0.0, min(min_score, 1.0))

        category = contract.get("category") or None

        terms = search_terms(str(normalized_query))
        results = rank_documents(str(normalized_query), documents, top_k, min_score, category)

        summary = {
            "searchable_terms": len(terms),
            "corpus_size": len(documents),
            "document_source": document_source,
            "top_k": top_k,
            "min_score": min_score,
            "category": category or "",
        }

        emit_trace_event(
            "kb_search_completed",
            {"result_count": len(results), **summary},
            state,
        )

        logger.info(
            "KBSearchNode: terms=%d corpus=%d source=%s results=%d",
            len(terms),
            len(documents),
            document_source,
            len(results),
        )

        return {"search_results": to_json(results), "search_summary": to_json(summary)}
