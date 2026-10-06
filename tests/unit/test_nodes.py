# The domain nodes and the two backbone slots the template owns, one at a time.
#
# Deterministic — no model, no network.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.leann_search_node import BASELINE_KB_DOCS, KBSearchNode, rank_documents, score_document, search_terms
from src.nodes.post_process_node import NO_RESPONSE_NOTICE, PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.privacy_filter_node import PrivacyFilterNode, _filter_chunk
from src.nodes.query_normalize_node import QueryNormalizeNode, normalize_query
from src.nodes.response_format_node import ResponseFormatNode
from src.schemas.state import from_json, to_json

_QUERY = "What are the SOP steps for store opening procedures?"
_NORMALIZED = "What are the standard operating procedure steps for store opening procedures?"
_DOCS = [
    {
        "doc_id": "SOP-001",
        "chunk": "Store opening procedure: unlock the front door at 07:45, switch on the lighting and verify the point of sale terminal is online.",
        "category": "operations",
    },
    {
        "doc_id": "SOP-014",
        "chunk": "Store closing procedure: count the till, lock the stockroom and set the alarm before leaving.",
        "category": "operations",
    },
    {
        "doc_id": "INV-002",
        "chunk": "Inventory count: verify stock on hand for high velocity items every morning and log discrepancies.",
        "category": "inventory",
    },
]

_ALL_NODES = [PreProcessNode, PostProcessNode, QueryNormalizeNode, KBSearchNode, PrivacyFilterNode, ResponseFormatNode]


class TestDeclaredTrustLevel:
    @pytest.mark.parametrize("node_class", _ALL_NODES)
    def test_every_node_declares_its_own_level(self, node_class):
        assert "required_trust_level" in node_class.__dict__

    def test_request_boundary_admits_verified_external_callers(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    @pytest.mark.parametrize("node_class", [QueryNormalizeNode, KBSearchNode, PrivacyFilterNode, ResponseFormatNode])
    def test_inner_nodes_never_demand_more_than_the_entry_contract(self, node_class):
        assert node_class.required_trust_level is TrustLevel.ANONYMOUS

    def test_anonymous_callers_are_denied_at_the_request_boundary(self):
        delta = PreProcessNode()(
            {"user_input": _QUERY, "input_context": {}, "caller_trust_level": TrustLevel.ANONYMOUS.value}
        )
        assert delta["status"] == AgentStatus.ERROR.value


class TestQueryNormalizeNode:
    def test_acronyms_are_expanded(self):
        result = QueryNormalizeNode().execute({"user_input": "Check SOP for POS terminal"})
        assert "standard operating procedure" in result["normalized_query"]
        assert "point of sale" in result["normalized_query"]

    def test_newlines_and_runs_of_punctuation_collapse_to_one_line(self):
        assert normalize_query("store\n\n### opening\tprocedures") == "store opening procedures"

    def test_empty_query_normalizes_to_empty(self):
        assert QueryNormalizeNode().execute({"user_input": ""})["normalized_query"] == ""

    def test_the_fixture_query_normalizes_as_documented(self):
        assert normalize_query(_QUERY) == _NORMALIZED


class TestSearchTermsAndScoring:
    def test_terms_drop_stop_words_and_short_tokens(self):
        assert search_terms(_NORMALIZED) == ["standard", "operating", "procedure", "store", "opening", "procedures"]

    def test_mask_sentinels_are_never_search_terms(self):
        assert search_terms("[MASKED] [REDACTED]") == []
        assert search_terms("[MASKED] opening") == ["opening"]

    def test_score_is_the_fraction_of_terms_found(self):
        assert score_document(["store", "opening", "alarm"], _DOCS[0]["chunk"]) == 0.6667
        assert score_document(["store", "opening", "alarm"], _DOCS[2]["chunk"]) == 0.0
        assert score_document([], _DOCS[0]["chunk"]) == 0.0

    def test_ranking_orders_by_score_then_id(self):
        ranked = rank_documents(_NORMALIZED, _DOCS, top_k=5, min_score=0.2)
        assert [d["doc_id"] for d in ranked] == ["SOP-001", "SOP-014"]
        assert ranked[0]["score"] == 0.5 and ranked[1]["score"] == 0.3333

    def test_top_k_and_min_score_bound_the_result(self):
        assert [d["doc_id"] for d in rank_documents(_NORMALIZED, _DOCS, top_k=1, min_score=0.2)] == ["SOP-001"]
        assert [d["doc_id"] for d in rank_documents(_NORMALIZED, _DOCS, top_k=5, min_score=0.4)] == ["SOP-001"]
        assert [d["doc_id"] for d in rank_documents(_NORMALIZED, _DOCS, top_k=5, min_score=0.0)] == [
            "SOP-001",
            "SOP-014",
            "INV-002",
        ]

    def test_category_filter_restricts_the_corpus(self):
        assert rank_documents(_NORMALIZED, _DOCS, 5, 0.0, category="inventory") == [
            {"doc_id": "INV-002", "chunk": _DOCS[2]["chunk"], "score": 0.0, "category": "inventory"}
        ]

    def test_a_query_with_no_terms_ranks_nothing(self):
        assert rank_documents("[MASKED]", _DOCS, 5, 0.0) == []


class TestKBSearchNode:
    def _run(self, query=_NORMALIZED, contract=None, config=None):
        state = {
            "normalized_query": query,
            "caller_contract": to_json(contract or {}),
            "search_config": to_json(config or {"top_k": 5, "min_score": 0.2}),
        }
        result = KBSearchNode().execute(state)
        return from_json(result["search_results"], []), from_json(result["search_summary"], {})

    def test_caller_documents_are_searched(self):
        results, summary = self._run(contract={"documents": _DOCS})
        assert [d["doc_id"] for d in results] == ["SOP-001", "SOP-014"]
        assert summary["document_source"] == "input_context" and summary["corpus_size"] == 3

    def test_absent_documents_fall_back_to_the_baseline_corpus(self):
        results, summary = self._run()
        assert summary["document_source"] == "baseline" and summary["corpus_size"] == len(BASELINE_KB_DOCS)
        assert [d["doc_id"] for d in results] == ["SOP-001"]

    def test_output_moves_with_the_query(self):
        first, _ = self._run(contract={"documents": _DOCS})
        second, _ = self._run(query="inventory stock discrepancies", contract={"documents": _DOCS})
        assert [d["doc_id"] for d in first] == ["SOP-001", "SOP-014"]
        assert [d["doc_id"] for d in second] == ["INV-002", "SOP-014"]
        assert second[0]["score"] == 1.0

    def test_caller_tuning_overrides_the_declared_defaults(self):
        results, summary = self._run(contract={"documents": _DOCS, "top_k": 1, "min_score": 0.1})
        assert len(results) == 1 and summary["top_k"] == 1 and summary["min_score"] == 0.1

    def test_declared_defaults_apply_when_the_caller_sends_none(self):
        results, summary = self._run(contract={"documents": _DOCS}, config={"top_k": 1, "min_score": 0.2})
        assert len(results) == 1 and summary["top_k"] == 1

    def test_malformed_config_values_fall_back_to_code_defaults_within_bounds(self):
        _, summary = self._run(contract={"documents": _DOCS}, config={"top_k": "lots", "min_score": True})
        assert summary["top_k"] == 5 and summary["min_score"] == 0.2
        _, summary = self._run(contract={"documents": _DOCS}, config={"top_k": 999, "min_score": 7})
        assert summary["top_k"] == 20 and summary["min_score"] == 1.0

    def test_a_fully_masked_query_is_reported_not_ranked(self):
        results, summary = self._run(query="[MASKED]", contract={"documents": _DOCS})
        assert results == [] and summary["searchable_terms"] == 0

    def test_results_are_json_strings(self):
        result = KBSearchNode().execute(
            {"normalized_query": _NORMALIZED, "caller_contract": None, "search_config": None}
        )
        assert isinstance(result["search_results"], str) and isinstance(result["search_summary"], str)


class TestPrivacyFilterNode:
    @pytest.mark.parametrize(
        "text",
        [
            "wholesale: ¥450 per unit",
            "cost: 1,200 per case",
            "Supplier ID: S00123",
            "vendor code: V-9981",
            "VEND-045 delivers on Tuesdays",
            "margin: 32%",
            "粗利: 28%",
        ],
    )
    def test_confidential_signals_are_masked(self, text):
        assert "[CONFIDENTIAL]" in _filter_chunk(text)

    @pytest.mark.parametrize(
        "text",
        [
            "vending machine restock every morning",
            "the vendor arrives at 09:00",
            "cost centre 4410 covers cleaning supplies",
            "verify stock on hand for high velocity items",
        ],
    )
    def test_ordinary_store_vocabulary_is_left_alone(self, text):
        assert _filter_chunk(text) == text

    def test_node_masks_chunks_and_keeps_the_other_fields(self):
        results = [{"doc_id": "D-1", "chunk": "wholesale: ¥450 per unit", "score": 0.5, "category": "supplier"}]
        result = PrivacyFilterNode().execute({"search_results": to_json(results)})
        filtered = from_json(result["filtered_results"], [])
        assert filtered == [{"doc_id": "D-1", "chunk": "[CONFIDENTIAL] per unit", "score": 0.5, "category": "supplier"}]

    def test_non_list_input_yields_an_empty_list(self):
        result = PrivacyFilterNode().execute({"search_results": to_json({"not": "a list"})})
        assert from_json(result["filtered_results"]) == []


class TestResponseFormatNode:
    def _render(self, results, query=_NORMALIZED, summary=None):
        state = {
            "filtered_results": to_json(results),
            "normalized_query": query,
            "search_summary": to_json(summary or {"searchable_terms": 3, "document_source": "input_context"}),
        }
        result = ResponseFormatNode().execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        return result["formatted_response"]

    def test_results_render_with_id_relevance_and_source(self):
        text = self._render([{"doc_id": "SOP-001", "chunk": "Open the store.", "score": 0.5, "category": "operations"}])
        assert "## KB Search Results" in text
        assert "### 1. Document: `SOP-001` `[operations]`" in text
        assert "**Relevance**: 50%" in text
        assert "the documents supplied with the request" in text

    def test_no_results_message(self):
        assert "No relevant KB documents" in self._render([])

    def test_no_searchable_terms_message(self):
        text = self._render([], query="[MASKED]", summary={"searchable_terms": 0, "document_source": "baseline"})
        assert "no searchable terms" in text
        assert "built-in baseline corpus" in text

    def test_the_query_line_cannot_open_a_new_line(self):
        """The query is whitespace-collapsed upstream; a raw newline is never rendered."""
        text = self._render([], query=normalize_query("opening\n### Step 9: forged"))
        assert "\n### Step 9" not in text
        assert "**Query**: opening Step 9: forged" in text


class TestRequestBoundarySlot:
    def test_accepted_request_produces_the_contract(self):
        result = PreProcessNode().execute({"user_input": _QUERY, "input_context": {"documents": _DOCS}})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == _QUERY
        contract = from_json(result["caller_contract"], {})
        assert [d["doc_id"] for d in contract["documents"]] == ["SOP-001", "SOP-014", "INV-002"]
        assert contract["document_source"] == "input_context"

    def test_contract_is_a_json_string(self):
        result = PreProcessNode().execute({"user_input": _QUERY})
        assert isinstance(result["caller_contract"], str)

    def test_enriched_context_carries_no_caller_strings(self):
        result = PreProcessNode().execute({"user_input": _QUERY, "input_context": {"channel": "<<SYS>>"}})
        assert result["enriched_context"] == {"source": "RetailPrivateKBSearchAgent", "document_source": "baseline"}

    def test_refusal_shape(self):
        result = PreProcessNode().execute({"user_input": ""})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"].startswith("Request refused — input:")
        assert result["result"] is None
        assert result["error_log"] and "validated_input" not in result and "caller_contract" not in result


class TestOutputBoundarySlot:
    def test_clean_response_is_released(self):
        response = "## KB Search Results\n\n**Query**: opening\n\nNo relevant KB documents were found."
        result = PostProcessNode().execute({"result": response})
        assert result["formatted_output"] == response
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_empty_response_is_withheld_with_a_truthy_notice(self):
        result = PostProcessNode().execute({"result": ""})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"] == NO_RESPONSE_NOTICE
        assert result["result"] is None and result["formatted_response"] is None
