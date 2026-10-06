"""AgentCore Platform v1.0"""

# RET-C2-178 — PreProcessNode (outer pre_process slot; the request boundary)
#
# Everything a caller can send is validated here, once, against the contract
# in src/services/caller_contract.py:
#   * the query is length-capped, screened for disallowed instructions
#     (chat-template control tokens included) and credential shapes, and
#     stripped of store-staff personal-data shapes before it is stored;
#   * every caller document is bounded, its id restricted to an inert
#     alphabet, its text screened and stripped the same way as the query;
#   * every number is parsed by a finite + bounded parser;
#   * a refusal names the field and never repeats the value.
#
# The refusal is CALLER-VISIBLE. AgentBaseGraph.route() sends an ERROR status
# straight to finalize, so post_process never runs on this path and
# get_output()'s `formatted_output or result` publishes whatever this node put
# there. So the notice is a closed set and nothing else: a field label from
# REFUSAL_FIELD_LABELS and a reason from the contract's REFUSAL_REASONS, both
# declared in code. The contract path (which carries a caller-controlled
# document index) and the matched pattern label go to the audit trail instead.
#
# The refusal is enforced HERE, in the node that owns the caller contract,
# rather than being left to the platform's own input screen. That screen
# scores some control-token forms and not others, and where it is absent or
# configured off the payload would reach the search path and return success.
# This node refuses on its own, which is why the boundary tests call
# execute() directly with no wrapper in front of it.
#
# Node contract:
#  - extend FunctionNode; implement execute(state) -> dict
#  - return ONLY the fields this node changes (never the full state)
#  - return AgentStatus enum values — never plain strings
#  - read input_context via state.get("input_context", {}) — read-only

from typing import Any, ClassVar, Dict, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import to_json
from src.services.caller_contract import (
    DECLARED_CONTEXT_FIELDS,
    DOCUMENT_FIELDS,
    REASON_CONTRACT_VIOLATION,
    REFUSAL_REASONS,
    ContractError,
    validate_request,
)

#: The caller-visible refusal, in full: this prefix, a label from
#: REFUSAL_FIELD_LABELS, ": ", and a reason from REFUSAL_REASONS. Nothing else
#: is ever published on this channel.
REFUSAL_PREFIX = "Request refused — "

#: Every field label a refusal may name. A ContractError path is the contract's
#: own vocabulary, but `input_context.documents[7].chunk` also carries a
#: caller-controlled index, and a path is not a label anyone declared. Each one
#: is reduced to a member of this set before it is published; the full path
#: goes to the audit event.
REFUSAL_FIELD_LABELS: Tuple[str, ...] = (
    "input",
    "input_context",
    *(f"input_context.{field}" for field in DECLARED_CONTEXT_FIELDS),
    *(f"input_context.documents.{field}" for field in DOCUMENT_FIELDS),
)


def caller_visible_field_label(path: str) -> str:
    """Reduce a ContractError field path to a label from REFUSAL_FIELD_LABELS.

    `input` stays `input`; `input_context.documents[7].chunk` becomes
    `input_context.documents.chunk`; `input_context.documents[7]` becomes
    `input_context.documents`. A path that reduces to nothing declared falls
    back to `input_context` rather than being echoed — the fallback is what
    makes this a closed set rather than a sanitiser.
    """
    segments = [segment.split("[", 1)[0] for segment in path.split(".")]
    while segments:
        label = ".".join(segments)
        if label in REFUSAL_FIELD_LABELS:
            return label
        segments.pop()
    return "input_context"


def caller_visible_reason(reason: str) -> str:
    """Hold a ContractError reason to the contract's declared set.

    Every raise site in the contract passes a declared constant, so this is a
    no-op today. It is here so that a raise site added later which composes its
    own sentence degrades to a label instead of publishing that sentence.
    """
    return reason if reason in REFUSAL_REASONS else REASON_CONTRACT_VIOLATION


def _refuse(label: str, reason: str) -> Dict[str, Any]:
    """The single non-success shape for this node.

    *label* is one of REFUSAL_FIELD_LABELS and *reason* one of the contract's
    REFUSAL_REASONS, so the published notice is drawn entirely from declared
    values. `result` is cleared in the same update: the envelope resolves
    `formatted_output or result` with no status check, so a refusal that left
    `result` alone would be one truthiness away from publishing it. The notice
    is truthy for the same reason.

    `error_log` carries this one new line and nothing read back out of state —
    the state reducer appends, so re-emitting the existing entries would
    duplicate every one of them.
    """
    message = f"{REFUSAL_PREFIX}{label}: {reason}"
    return {
        "status": AgentStatus.ERROR.value,
        "formatted_output": message,
        "result": None,
        "error_log": [message],
    }


class PreProcessNode(FunctionNode):
    """Validate the caller request and produce the contract the pipeline runs on.

    required_trust_level = VERIFIED_EXTERNAL: this is the external-facing
    backbone gate — the caller must be at least VERIFIED_EXTERNAL.

    Input state keys:
        user_input:    the search query
        input_context: structured invocation parameters (read-only)

    Output state keys (partial dict):
        validated_input:  the screened, stripped query
        caller_contract:  JSON string — the validated documents and tuning
        enriched_context: where the documents came from (no caller strings)
        status:           AgentStatus.SUCCESS, or ERROR on a refusal
        formatted_output: (on refusal) the closed-set notice, published to the
                          caller because ERROR routes straight to finalize
        result:           (on refusal) cleared, so the envelope cannot fall back
        error_log:        (on refusal) that same notice, and nothing else
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        try:
            contract = validate_request(user_input, input_context)
        except ContractError as refusal:
            # A field label is named so the caller can act; the value never
            # appears, and neither does anything else the caller supplied.
            #
            # The refusal is written to formatted_output because the envelope
            # surfaces that field and nothing else: the error log is not part
            # of it, so a refusal that lived only there would reach the caller
            # as an error with no reason at all. What travels is a declared
            # label and a declared reason — nothing is composed here.
            #
            # The exception's own attributes are NOT interpolated. `field`
            # carries the document index the caller's payload decided, and
            # `detail` names a pattern the platform detector declares rather
            # than this template; both belong in the audit trail, which is
            # where they go.
            label = caller_visible_field_label(refusal.field)
            reason = caller_visible_reason(refusal.reason)
            emit_trace_event(
                "kb_search_request_refused",
                {"field": refusal.field, "label": label, "reason": reason, "pattern": refusal.detail},
                state,
            )
            return _refuse(label, reason)

        emit_trace_event(
            "kb_search_request_accepted",
            {
                "query_chars": len(contract["query"]),
                "document_count": len(contract["documents"]),
                "document_source": contract["document_source"],
            },
            state,
        )

        return {
            "validated_input": contract["query"],
            # Structured values are stored as JSON strings: checkpoint
            # serialization does not carry bare containers safely.
            "caller_contract": to_json({key: value for key, value in contract.items() if key != "query"}),
            "enriched_context": {
                "source": "RetailPrivateKBSearchAgent",
                "document_source": contract["document_source"],
            },
            "status": AgentStatus.SUCCESS.value,
        }
