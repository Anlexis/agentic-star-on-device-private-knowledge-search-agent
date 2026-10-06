# PB-6 — Invoke-Order Boundary: a full agent.invoke() must execute the fixed
# AgentBaseGraph backbone in order.
#
# The Cat 2 backbone is fixed and is NEVER overridden by a template
# (add_edges() belongs to the framework):
#
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#           -> finalize -> END
#
# The framework records every executed node in `node_history` (an AgentState
# field whose reducer is operator.add, so entries accumulate in execution
# order). Each entry is the node's CLASS NAME - appended by BaseNode.__call__.
#
# For RET-C2-178 (Cat 2, two-layer nested) the `main` slot is a GraphNode
# subclass (KBSearchGraphNode) that delegates to the inner DomainWorkflowGraph.
# The inner graph runs with its own state; its inner node_history is NOT merged
# back into the outer state (merge_output() maps only formatted_response /
# result / status), so the OUTER node_history contains exactly the five backbone
# slots - never the inner domain nodes.
#
# This test drives a real end-to-end RetailPrivateKBSearchAgent().invoke() over
# a valid domain payload with an EXTERNAL caller context and asserts the
# surfaced node_history matches the canonical backbone order. A SUCCESS terminal
# status is required: on any non-SUCCESS status route() short-circuits
# main -> finalize and the post_process (S-3) slot is skipped, which would
# itself be an invoke-order violation this test would catch.
#
# The caller uses InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
# — NOT for_internal() — to exercise the real external-caller trust path. Our
# PreProcessNode is VERIFIED_EXTERNAL and inner nodes are ANONYMOUS, so the
# full domain workflow runs for an EXTERNAL caller. Using for_internal() would
# mask any inner-node trust-trap; VERIFIED_EXTERNAL exercises the same path a
# real STG invoke does.
#
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import RetailPrivateKBSearchAgent


# --- TEMPLATE-SPECIFIC ------------------------------------------------------
# The `main`-slot GraphNode class name for THIS template.
_MAIN_SLOT_NODE = "KBSearchGraphNode"

# A valid KB search query that drives the full domain workflow to a SUCCESS
# terminal status. Plain text input — PreProcessNode strips PII and writes
# validated_input; QueryNormalizeNode expands acronyms; LEANNSearchNode returns
# stub results; PrivacyFilterNode filters; ResponseFormatNode renders.
_VALID_PAYLOAD = "What are the SOP steps for store opening procedures?"
# --- END TEMPLATE-SPECIFIC --------------------------------------------------

# Canonical AgentBaseGraph backbone execution order, by node class name as
# recorded in node_history. Four entries are framework/scaffold-fixed and
# identical for every Cat 1 / Cat 2 template; only _MAIN_SLOT_NODE is
# template-specific.
_EXPECTED_ORDER = [
    "InitializeNode",  # framework default  (initialize slot)
    "PreProcessNode",  # scaffold-standard  (pre_process slot, S-1/S-2)
    _MAIN_SLOT_NODE,  # TEMPLATE-SPECIFIC  (main slot GraphNode)
    "PostProcessNode",  # scaffold-standard  (post_process slot, S-3 gate)
    "FinalizeNode",  # framework default  (finalize slot)
]


def _run() -> dict:
    """Run a full end-to-end invocation with a VERIFIED_EXTERNAL caller."""
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    return RetailPrivateKBSearchAgent().invoke(_VALID_PAYLOAD, ctx=ctx)


class TestInvokeOrderBoundary:
    """PB-6: full agent.invoke() executes the backbone in the fixed order."""

    def test_invoke_reaches_success(self):
        """The full run must terminate SUCCESS — otherwise route() short-circuits
        main -> finalize and the post_process (S-3) slot never runs."""
        result = _run()
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected SUCCESS, got {result.get('status')!r}. result={result!r}"

    def test_output_is_non_empty(self):
        """A successful run must surface a non-empty gated output."""
        assert _run().get("output"), "invoke() surfaced an empty output"

    def test_node_history_is_populated(self):
        """node_history must be a non-empty list of node class-name strings."""
        history = _run().get("node_history")
        assert isinstance(history, list) and history, f"node_history must be a non-empty list, got {history!r}"
        assert all(isinstance(n, str) for n in history), f"node_history entries must be strings, got {history!r}"

    def test_backbone_slot_order(self):
        """Core invoke-order boundary: the S-1 pre_process slot runs before the
        domain main slot, which runs before the S-3 post_process slot — as a
        strict ordered subsequence of node_history."""
        history = _run().get("node_history", [])
        ordered_slots = ["PreProcessNode", _MAIN_SLOT_NODE, "PostProcessNode"]
        for name in ordered_slots:
            assert name in history, f"Expected backbone slot {name!r} in node_history, got {history!r}"
        positions = [history.index(name) for name in ordered_slots]
        assert positions == sorted(positions), (
            f"Backbone slots executed out of order: {ordered_slots} at {positions}. " f"node_history={history!r}"
        )

    def test_full_backbone_sequence(self):
        """The complete AgentBaseGraph backbone order:
        initialize -> pre_process -> main -> post_process -> finalize."""
        history = _run().get("node_history", [])
        assert history == _EXPECTED_ORDER, (
            "node_history does not match the canonical backbone order.\n"
            f"  expected: {_EXPECTED_ORDER}\n"
            f"  actual:   {history}"
        )
