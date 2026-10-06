# Graph composition — the two-layer wiring, and the contracts that hold it
# together.
#
# Outer backbone:  initialize -> pre_process -> main -> {route} -> post_process -> finalize
# Main slot:       KBSearchGraphNode -> DomainWorkflowGraph
# Inner pipeline:  query_normalize -> kb_search -> privacy_filter -> response_format
#
# The pieces that are easy to get wrong and impossible to see from a passing
# run are pinned here: the structured request only crosses the boundary because
# the bridge carries it, and the runtime tuning only reaches a domain node
# because the initial-state hook seeds it.
#
# Deterministic — no model, no network.

import pytest

from framework.errors import ConfigError
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.context_bridge import get_caller_contract, set_caller_contract
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import Graph, KBSearchGraphNode, RetailPrivateKBSearchAgent, runtime_config
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_QUERY = "What are the SOP steps for store opening procedures?"
_DOCS = {
    "documents": [
        {
            "doc_id": "SOP-001",
            "chunk": "Store opening procedure: unlock the front door at 07:45.",
            "category": "operations",
        },
        {"doc_id": "INV-002", "chunk": "Inventory count: verify stock on hand every morning.", "category": "inventory"},
    ]
}


class TestOuterGraph:
    def test_inherits_the_framework_base_class(self):
        from framework.graph.agent_base_graph import AgentBaseGraph

        assert issubclass(RetailPrivateKBSearchAgent, AgentBaseGraph)

    def test_state_schema_and_name(self):
        agent = RetailPrivateKBSearchAgent()
        assert agent.state_schema is State
        assert agent.name == "RetailPrivateKBSearchAgent"

    def test_compile_fills_every_backbone_slot(self):
        agent = RetailPrivateKBSearchAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["main"], KBSearchGraphNode)
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_alias_resolves_to_the_agent(self):
        assert Graph is RetailPrivateKBSearchAgent

    def test_get_output_is_the_framework_envelope(self):
        """The output gate is the single place released text is checked; no
        second layer in the envelope masks it."""
        assert "get_output" not in RetailPrivateKBSearchAgent.__dict__


class TestRuntimeConfiguration:
    def test_config_file_is_loaded(self):
        loaded = runtime_config()
        assert loaded["max_retry"] == 3
        assert loaded["search"] == {"top_k": 5, "min_score": 0.2}

    def test_declared_retry_ceiling_reaches_the_backbone(self):
        """max_retry is validated at compile time, so a bad value must fail."""
        with pytest.raises(ConfigError):
            RetailPrivateKBSearchAgent(config={"max_retry": 99}).compile()

    def test_tuning_is_forwarded_to_the_inner_graph(self):
        assert KBSearchGraphNode()._parent_config() == {"configurable": {"search": {"top_k": 5, "min_score": 0.2}}}

    def test_inner_graph_seeds_the_tuning_into_state(self):
        """Node execute() methods take no config argument, so seeding is the
        only route a declared value can travel into a domain node."""
        inner = DomainWorkflowGraph(config={"configurable": {"search": {"top_k": 2, "min_score": 0.5}}})
        assert from_json(inner._extra_initial_state()["search_config"], {}) == {"top_k": 2, "min_score": 0.5}

    @pytest.mark.parametrize(
        "search",
        [{"top_k": 0}, {"top_k": 21}, {"top_k": "five"}, {"top_k": True}, {"min_score": 2}, {"min_score": "half"}, "x"],
    )
    def test_malformed_tuning_is_refused_at_compile_time(self, search):
        with pytest.raises(ConfigError):
            DomainWorkflowGraph(config={"configurable": {"search": search}}).compile()

    def test_absent_tuning_is_not_an_error(self):
        DomainWorkflowGraph(config={}).compile()


class TestContextBridge:
    def test_contract_crosses_the_boundary(self):
        """The framework hands the inner graph only a string, so the structured
        request travels here or not at all."""
        contract = {"documents": [{"doc_id": "D-1", "chunk": "text", "category": ""}], "top_k": 2}
        KBSearchGraphNode().extract_input({"validated_input": _QUERY, "caller_contract": to_json(contract)})
        assert get_caller_contract()["top_k"] == 2

    def test_inner_initial_state_reads_the_bridge(self):
        set_caller_contract({"documents": [], "top_k": 4})
        assert from_json(DomainWorkflowGraph()._extra_initial_state()["caller_contract"], {})["top_k"] == 4

    def test_extract_input_prefers_the_validated_query(self):
        node = KBSearchGraphNode()
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"
        assert node.extract_input({}) == ""


class TestMergeOutput:
    def test_success_maps_the_response_to_both_fields(self):
        delta = KBSearchGraphNode().merge_output({}, {"formatted_response": "R", "status": AgentStatus.SUCCESS.value})
        assert delta == {"formatted_response": "R", "result": "R", "status": AgentStatus.SUCCESS.value}

    def test_non_success_publishes_no_response(self):
        delta = KBSearchGraphNode().merge_output({}, {"formatted_response": "R", "status": AgentStatus.ERROR.value})
        assert delta["result"] is None and delta["formatted_response"] is None


class TestInnerGraph:
    def test_registers_the_four_domain_steps(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes) == {"query_normalize", "kb_search", "privacy_filter", "response_format"}
        for node in inner._nodes.values():
            assert node.required_trust_level == TrustLevel.ANONYMOUS

    def test_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "ret_c2_178_private_kb_search_workflow"
        assert inner.state_schema is State

    def test_output_shape_on_success(self):
        out = DomainWorkflowGraph().get_output({"formatted_response": "R", "status": AgentStatus.SUCCESS.value})
        assert out["formatted_response"] == "R" and out["status"] == AgentStatus.SUCCESS.value

    def test_output_withholds_a_response_on_failure(self):
        out = DomainWorkflowGraph().get_output({"formatted_response": "R", "status": AgentStatus.ERROR.value})
        assert out["formatted_response"] is None

    def test_route_is_annotated_with_this_graphs_own_state(self):
        """LangGraph reads a path callable's annotation as its input schema and
        projects away every field the annotation does not declare."""
        assert DomainWorkflowGraph.route.__annotations__["state"] is State


class TestEndToEndComposition:
    """The full agent, at the trust level the manifest declares."""

    def _run(self, input_context=None):
        ctx = InvocationContext(session_id="s", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        return RetailPrivateKBSearchAgent(config=runtime_config()).invoke(_QUERY, ctx=ctx, input_context=input_context)

    def test_run_succeeds_over_caller_documents(self):
        result = self._run(_DOCS)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "### 1. Document: `SOP-001`" in result["output"]
        assert "INV-002" not in result["output"]

    def test_run_succeeds_over_the_baseline_without_documents(self):
        result = self._run()
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "built-in baseline corpus" in result["output"]

    def test_backbone_traverses_the_output_gate(self):
        assert self._run(_DOCS)["node_history"] == [
            "InitializeNode",
            "PreProcessNode",
            "KBSearchGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ]
