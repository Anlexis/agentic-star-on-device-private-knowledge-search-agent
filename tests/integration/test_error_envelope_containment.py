# Containment of the error envelope, driven end to end.
#
# The envelope resolves the released output as `formatted_output or result`,
# with no status check. Three properties follow, and all three are live here:
#
#   1. a falsy formatted_output ACTIVATES the fallback, so "" as a "withheld"
#      marker produces the exact disclosure it was written to prevent;
#   2. an output gate that RAISES leaks, because the wrapper turns an exception
#      into a bare error update that clears nothing;
#   3. the platform's own credential scan raises on the gate node's return
#      value and the wrapper then discards that node's whole update — the
#      clearing included — so a gate whose pattern set is narrower than the
#      platform's is a containment bypass rather than a weaker filter.
#
# The fault is injected on the DATA path, never on the gate: the main slot's
# merge_output is made to publish a drifted response — the one hop into outer
# state the platform does not scan, exactly where an index returning a
# credential-shaped chunk would surface. Patching the gate would test the
# patch, not the agent.
#
# Deterministic — no model, no network.

import os

import pytest

from tests.integration.asgi import Client

_TOKEN = "test-caller-token"
os.environ["INVOKE_AUTH_TOKEN"] = _TOKEN

from src.api.server import app  # noqa: E402
from src.graph.graph import KBSearchGraphNode  # noqa: E402
from src.nodes.post_process_node import BLOCKED_NOTICE  # noqa: E402

client = Client(app)
AUTH = {"Authorization": f"Bearer {_TOKEN}"}

_QUERY = "What are the SOP steps for store opening procedures?"
_DOCS = {
    "documents": [
        {"doc_id": "SOP-001", "chunk": "Store opening procedure: unlock the front door.", "category": "operations"}
    ]
}

# Shapes the platform's own detector recognises. Probing only with one it does
# not (an assignment form, say) would report a narrower gate as safe when it
# is not.
_LEAKED = {
    "aws_key": "AKIAIOSFODNN7EXAMPLE",
    "conn_string": "postgresql://db.internal.example:5432/kb",
    # And one the platform does NOT recognise, which only the local half catches.
    "credential_assignment": "password=hunter2hunter2",
}


def _drifted_response(value: str) -> str:
    return f"## KB Search Results\n\n### 1. Document: `SOP-001`\n**Relevance**: 50%\n\nnote {value}\n"


@pytest.fixture(params=sorted(_LEAKED), ids=sorted(_LEAKED))
def drifted_main_slot(request, monkeypatch):
    """Make the DATA path publish a response the gate must refuse."""
    value = _LEAKED[request.param]

    def _drifted(self, state, sub_result):  # noqa: ANN001
        return {
            "formatted_response": _drifted_response(value),
            "result": _drifted_response(value),
            "status": sub_result.get("status"),
        }

    monkeypatch.setattr(KBSearchGraphNode, "merge_output", _drifted)
    return value


def _invoke():
    return client.post("/invoke", json={"input": _QUERY, "input_context": _DOCS}, headers=AUTH)


class TestDriftedResponseIsContained:
    def test_envelope_carries_no_leaked_value(self, drifted_main_slot):
        assert drifted_main_slot not in _invoke().text

    def test_status_is_error(self, drifted_main_slot):
        assert _invoke().json()["status"] == "error"

    def test_output_is_the_withheld_notice(self, drifted_main_slot):
        """The notice is TRUTHY, so the fallback to result stays closed.

        It also proves the block came from the gate rather than from the
        platform raising: when the platform raises, this node's update is
        discarded and there is no notice to observe.
        """
        assert _invoke().json()["output"] == BLOCKED_NOTICE

    def test_block_happened_at_the_gate_not_upstream(self, drifted_main_slot):
        assert "PostProcessNode" in _invoke().json()["node_history"]

    def test_envelope_carries_no_traceback_or_source_path(self, drifted_main_slot):
        body = _invoke().text
        assert "Traceback" not in body
        assert "/src/" not in body

    def test_envelope_carries_no_response_content(self, drifted_main_slot):
        assert "KB Search Results" not in str(_invoke().json()["output"])


class TestCleanPathControl:
    """Without the drift the same request still produces its real answer.

    A gate that refused everything would pass every assertion above; this is
    what stops that from counting as containment.
    """

    def test_same_request_succeeds_when_the_data_path_is_clean(self):
        response = _invoke()
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "### 1. Document: `SOP-001`" in response.json()["output"]

    def test_gate_node_runs_on_the_clean_path_too(self):
        assert "PostProcessNode" in _invoke().json()["node_history"]
