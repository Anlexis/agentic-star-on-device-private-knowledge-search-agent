"""AgentCore Platform v1.0"""

# RET-C2-178 — PostProcessNode: the output boundary of the agent.
#
# The stated output invariant of this template: nothing credential-shaped and
# no store-staff personal-data shape leaves the agent. This node enforces it
# on the rendered response with the module-level scan below, called from
# execute().
#
# The credential half takes the platform's own detector as the FLOOR and adds
# the local patterns it lacks. That is not a style choice. The platform scans
# every value of every node result for credentials and RAISES when it finds
# one; the wrapper then discards this node's whole return value, the clearing
# included, and the envelope falls back to the un-gated response still sitting
# in state. So a local list narrower than the platform's is not a weaker gate
# — it is a containment bypass. The local patterns are kept because they catch
# what the platform does not (assignment forms such as "password=…", "pk-" and
# "ak-" key prefixes, private-key headers); the union is a superset of both.
#
# Two independent layers, each with its own audit event:
#   * inbound — the request boundary strips personal-data shapes from the
#     query and every caller document and refuses credential shapes there;
#   * outbound — this gate refuses to release anything still matching. Both
#     directions read ONE pattern definition, so they cannot drift apart.
#
# Containment on violation: returning an error is not enough on its own. The
# framework's envelope resolves the output as `formatted_output or result` with
# no status check, so a gate that raised — or that set an error status without
# clearing — still ships the un-gated response inside the error envelope. This
# node therefore CLEARS every output-bearing field as it blocks, and replaces
# formatted_output with a TRUTHY notice: a falsy replacement re-opens the very
# fallback the clearing exists to close. No second layer re-checks this in the
# envelope; the gate is the single place the released text is checked, so the
# mutant that removes it can fail.
#
# Every non-success return goes through the one module-level _contain() helper,
# and what it publishes is a closed set: the notice for one of ERROR_REASONS
# and nothing else — never error_log, never the violation label, never any text
# a node composed. Those can carry an upstream string or a caller fragment, and
# truncating or redacting them is not a closed set. error_log stays the
# INTERNAL channel: the reducer appends to it and the audit trail needs it; it
# is simply not projected to the caller.

import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.caller_contract import find_credential, find_personal_data

logger = logging.getLogger(__name__)

# The replacement is TRUTHY on purpose. The framework's envelope falls back to
# state["result"] whenever formatted_output is falsy, so an empty string here
# would re-open the exact channel the clearing closes.
BLOCKED_NOTICE = (
    "[OUTPUT WITHHELD — the KB search response did not pass the output boundary. "
    "Remove credential-like strings and personal identifiers from the knowledge-base documents and retry.]"
)
NO_RESPONSE_NOTICE = "[NO RESPONSE — the search pipeline produced no response to release.]"

# Every state field AT THIS LEVEL that can carry released text. On a violation
# each one is overwritten, so no path out of the graph — including the
# framework's own fallback to state["result"] — can reach the un-gated
# response, and a checkpoint or a downstream reader cannot pick it up either.
OUTPUT_BEARING_FIELDS: Tuple[str, ...] = ("result", "formatted_output", "formatted_response")


def security_gate_output(content: Any) -> Optional[str]:
    """Scan released content and name the first violation, or None if clean.

    Walks nested mappings and sequences, scanning every leaf — and a mapping's
    KEYS as strings alongside its values, because a values-only walk reports
    zero findings on exactly the case where the credential IS the key.

    Returns the NAME of the matched pattern — never the matched text and never
    the key, which would put the leak into the record that reports it: the
    returned name travels into error_log, where the framework's own credential
    scan would raise on it and discard this node's whole update, the clearing
    included.
    """
    if content is None:
        return None
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(key, str):
                hit = find_credential(key) or find_personal_data(key)
                if hit:
                    return hit
            hit = security_gate_output(value)
            if hit:
                return hit
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            hit = security_gate_output(item)
            if hit:
                return hit
        return None
    text = str(content)
    return find_credential(text) or find_personal_data(text)


# Reason codes — the two non-success outcomes this node has, and the only keys
# into the notice table below. Declared as a set so the tests can pin it: the
# caller-visible envelope carries the notice for one of these and nothing else.
REASON_OUTPUT_WITHHELD = "output_withheld"
REASON_NO_RESPONSE = "no_response"
ERROR_REASONS: "frozenset[str]" = frozenset({REASON_OUTPUT_WITHHELD, REASON_NO_RESPONSE})

_NOTICE_FOR_REASON: Dict[str, str] = {
    REASON_OUTPUT_WITHHELD: BLOCKED_NOTICE,
    REASON_NO_RESPONSE: NO_RESPONSE_NOTICE,
}


def _contain(reason: str, new_errors: Optional[List[str]] = None) -> Dict[str, Any]:
    """The node result for ANY non-success outcome — the single error shape.

    Error status, every output-bearing field cleared, and a caller-visible
    notice that is a module constant selected by *reason* — one of
    ERROR_REASONS. Nothing is read out of state: not the response, not
    error_log. The inner entries are already in error_log and the state reducer
    APPENDS, so re-emitting them here would duplicate every line as well as
    risk publishing it.

    *new_errors* are this node's own lines for the internal channel; they never
    enter the envelope. The notice is TRUTHY, because the framework resolves
    `formatted_output or result` with no status check and a falsy replacement
    would re-open the fallback the clearing exists to close.
    """
    contained: Dict[str, Any] = {field: None for field in OUTPUT_BEARING_FIELDS}
    contained["formatted_output"] = _NOTICE_FOR_REASON[reason]
    contained["status"] = AgentStatus.ERROR.value
    contained["error_log"] = list(new_errors) if new_errors else [reason]
    return contained


class PostProcessNode(FunctionNode):
    """Output gate: refuse to release credential-like or personal-data content.

    required_trust_level = ANONYMOUS: inner backbone slot — the trust check
    is handled by PreProcessNode (VERIFIED_EXTERNAL) upstream.

    Input state keys:
        result: the rendered KB response (from KBSearchGraphNode.merge_output)

    Output state keys (partial dict):
        formatted_output:   the response when clean; a truthy withheld notice
                            on a violation — never an empty value, which would
                            re-open the envelope's fallback
        result:             cleared alongside formatted_output on a violation
        formatted_response: cleared alongside formatted_output on a violation
        status:             AgentStatus.SUCCESS or AgentStatus.ERROR
        error_log:          (on a violation) one closed-set reason label
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        result = state.get("result")
        text = result if isinstance(result, str) else ("" if result is None else str(result))

        if not text.strip():
            # Nothing was rendered. The pipeline always renders on success, so
            # this is a fault, and the notice is still truthy: an empty
            # formatted_output would let the envelope fall back to whatever
            # remains in result.
            logger.error("PostProcessNode: no response to release")
            emit_trace_event("kb_output_withheld", {"reason": REASON_NO_RESPONSE, "violation": "empty_response"}, state)
            return _contain(REASON_NO_RESPONSE, ["No response was produced by the search pipeline"])

        violation = security_gate_output(text)
        if violation:
            logger.error("PostProcessNode: output withheld — violation type: %s", violation)
            emit_trace_event("kb_output_withheld", {"reason": REASON_OUTPUT_WITHHELD, "violation": violation}, state)
            return _contain(REASON_OUTPUT_WITHHELD, [f"Output withheld at the boundary — {violation}"])

        emit_trace_event("kb_search_response_emitted", {"response_chars": len(text)}, state)

        return {"formatted_output": text, "status": AgentStatus.SUCCESS.value}
