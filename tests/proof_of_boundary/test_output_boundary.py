# The output boundary — what the agent refuses to release, and what it clears
# when it refuses.
#
# The stated invariant: nothing credential-shaped and no store-staff
# personal-data shape leaves the agent. No monetary rounding grid applies —
# the response carries document ids, relevance percentages and document text,
# never a monetary aggregate — so the identifier invariant is what is enforced,
# and document ids must come out byte-identical.
#
# Two properties are pinned here that a gate can appear to have without having:
#
#   * DETECTOR PARITY. The platform scans every value of every node result for
#     credentials and RAISES when it finds one; the wrapper then discards this
#     node's whole return value, the clearing included, and the envelope falls
#     back to the un-gated response still in state. So a local pattern list
#     narrower than the platform's is not a weaker gate — it is a bypass. Every
#     shape the platform recognises is probed here.
#
#   * CLEARING, ASSERTED AS PRESENCE AND EMPTINESS. Partial state updates are
#     MERGED, so omitting a key leaves the old value in state — and
#     `assert not result.get(field)` then passes on a gate that cleared
#     nothing. Each assertion below requires the key to be in the returned
#     update AND to carry the cleared value.
#
# The replacement notice is TRUTHY on purpose: the envelope resolves the output
# as `formatted_output or result`, so a falsy replacement re-opens the exact
# fallback the clearing exists to close.
#
# Deterministic — no model, no network.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.security.credential_detector import detect_credentials

from src.nodes.leann_search_node import BASELINE_KB_DOCS
from src.nodes.post_process_node import BLOCKED_NOTICE, OUTPUT_BEARING_FIELDS, PostProcessNode, security_gate_output
from src.nodes.privacy_filter_node import _filter_chunk
from src.nodes.response_format_node import _render_response

# Simulated shapes — not real credentials.
_SHAPES = {
    "openai_key": "sk-TESTKEY1234567890abcdefghij",
    "stripe_key": "sk_live_TESTKEY1234567890abcd",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c",
    "aws_key": "AKIAIOSFODNN7EXAMPLE",
    "bearer_token": "Bearer abcdefghijklmnopqrstuvwxyz",
    "conn_string": "postgresql://db.internal.example:5432/kb",
}

_CLEAN_RESPONSE = (
    "## KB Search Results\n\n**Query**: store opening procedures\n\n"
    "**Searched**: the documents supplied with the request\n\n**Results**: 1 document(s) found\n\n---\n\n"
    "### 1. Document: `SOP-001` `[operations]`\n**Relevance**: 50%\n\n"
    "Store opening procedure: unlock the front door at 07:45.\n"
)


def _response_with(value: str) -> str:
    return f"## KB Search Results\n\n### 1. Document: `D-1`\n**Relevance**: 50%\n\nnote {value}\n"


def _block(value: str) -> dict:
    return PostProcessNode().execute({"result": _response_with(value), "formatted_response": _response_with(value)})


class TestDetectorParity:
    """Every shape the platform recognises must be refused HERE first."""

    @pytest.mark.parametrize("name,value", sorted(_SHAPES.items()))
    def test_platform_recognises_the_probe(self, name, value):
        """Guard on the probes themselves: a shape the platform ignores would
        make the parity assertion below vacuously true."""
        assert detect_credentials(value), f"probe for {name} no longer matches"

    @pytest.mark.parametrize("name,value", sorted(_SHAPES.items()))
    def test_gate_refuses_every_platform_shape(self, name, value):
        assert security_gate_output(_response_with(value)) is not None

    @pytest.mark.parametrize("name,value", sorted(_SHAPES.items()))
    def test_block_is_attributed_to_this_gate(self, name, value):
        """The block must come from the node, not from the platform raising.

        When the platform raises instead, this node's whole return value is
        discarded — so there is no error status to observe here at all.
        """
        result = _block(value)
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"], "a block must name its reason"

    @pytest.mark.parametrize(
        "value",
        [
            "password = hunter2hunter2",
            "-----BEGIN RSA PRIVATE KEY-----",
            "pk-TESTKEY1234567890",
            "ak-TESTKEY1234567890",
        ],
    )
    def test_gate_also_covers_shapes_the_platform_does_not(self, value):
        assert not detect_credentials(value), "this probe is meant to be one the platform misses"
        assert security_gate_output(_response_with(value)) is not None

    def test_reason_is_a_label_never_the_matched_text(self):
        result = _block(_SHAPES["aws_key"])
        assert _SHAPES["aws_key"] not in str(result)
        assert "aws_key" in result["error_log"][0]


class TestNestedSurface:
    def test_nested_leak_is_found(self):
        assert security_gate_output({"payload": {"args": {"note": _SHAPES["aws_key"]}}}) is not None

    def test_leak_inside_a_list_is_found(self):
        assert security_gate_output([{"note": _SHAPES["jwt"]}]) is not None

    def test_clean_nested_structure_passes(self):
        """The control that proves the scan is not simply refusing everything."""
        assert security_gate_output({"payload": {"args": {"note": "shelf life 12 months"}}}) is None


class TestPersonalDataIsRefusedOutbound:
    @pytest.mark.parametrize("value", ["staff.mgr@store.example", "03-1234-5678", "EMP-123456"])
    def test_store_staff_shapes_are_refused(self, value):
        assert security_gate_output(_response_with(value)) is not None
        assert _block(value)["status"] == AgentStatus.ERROR.value

    def test_a_real_response_passes_the_gate(self):
        assert security_gate_output(_CLEAN_RESPONSE) is None


class TestTheBaselineCorpusRendersAndReleasesUnchanged:
    """The control against a gate that refuses everything, and the identifier
    invariant: `SOP-001`, `INC-101`, `PROD-045` are the `<letters>-<digits>`
    shapes a currency precision grid misreads, and there is no such layer here."""

    def _rendered(self):
        results = [
            {"doc_id": d["doc_id"], "chunk": _filter_chunk(d["chunk"]), "score": 1.0, "category": d["category"]}
            for d in BASELINE_KB_DOCS
        ]
        return _render_response("store", results, {"searchable_terms": 1, "document_source": "baseline"})

    def test_every_baseline_document_passes_the_gate(self):
        assert security_gate_output(self._rendered()) is None

    @pytest.mark.parametrize("doc", BASELINE_KB_DOCS, ids=[d["doc_id"] for d in BASELINE_KB_DOCS])
    def test_document_ids_and_text_are_byte_identical(self, doc):
        rendered = self._rendered()
        assert f"Document: `{doc['doc_id']}`" in rendered
        assert doc["chunk"] in rendered, "the privacy filter must leave the baseline corpus untouched"

    @pytest.mark.parametrize("token", ["07:45", "5-25°C", "48 units", "12 months", "5 business days", "SKUs"])
    def test_structural_tokens_survive(self, token):
        assert token in self._rendered()


class TestClearingOnViolation:
    """Presence AND emptiness — merged updates make omission look like clearing."""

    @pytest.mark.parametrize("field", OUTPUT_BEARING_FIELDS)
    def test_every_output_bearing_field_is_present_in_the_update(self, field):
        result = _block(_SHAPES["aws_key"])
        assert field in result, (
            f"{field} missing from the returned update — a merged update leaves "
            "the previous value in state, so omission is not clearing"
        )

    def test_released_text_is_replaced_not_merely_flagged(self):
        result = _block(_SHAPES["aws_key"])
        assert result["result"] is None and result["formatted_response"] is None
        assert _SHAPES["aws_key"] not in str(result["formatted_output"])

    def test_replacement_is_truthy(self):
        """A falsy replacement re-opens the envelope's fallback to result."""
        result = _block(_SHAPES["aws_key"])
        assert result["formatted_output"] == BLOCKED_NOTICE
        assert bool(result["formatted_output"])

    def test_status_is_error(self):
        assert _block(_SHAPES["aws_key"])["status"] == AgentStatus.ERROR.value


class TestCleanAndEmptyPaths:
    def test_clean_response_passes_unchanged(self):
        result = PostProcessNode().execute({"result": _CLEAN_RESPONSE})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == _CLEAN_RESPONSE
        assert "result" not in result, "a clean release leaves the rendered response in place"

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_response_still_yields_a_truthy_notice(self, empty):
        """An empty formatted_output would let the envelope fall back to result."""
        result = PostProcessNode().execute({"result": empty})
        assert bool(result["formatted_output"])
        assert result["result"] is None
        assert result["status"] == AgentStatus.ERROR.value
