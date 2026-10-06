"""AgentCore Platform v1.0"""

# Caller-request bridge across the outer/inner graph boundary.
#
# Why it exists: the framework invokes a nested graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)`. Only the request
# STRING crosses — neither the outer state nor the caller's structured
# invocation parameters are forwarded. So the validated documents and search
# tuning the pre_process node produces would never reach the search pipeline.
#
# The two sanctioned subclass hooks bridge it:
#
#   KBSearchGraphNode.extract_input(state)       [runs BEFORE subgraph.invoke]
#       -> set_caller_contract(<validated contract>)
#   DomainWorkflowGraph._extra_initial_state()   [runs INSIDE subgraph.invoke]
#       -> seeds the contract into the inner state
#
# What crosses is the VALIDATED contract only: every document has already
# passed its shape check, its free text has been screened and stripped of
# personal-data shapes, and every number has been parsed as finite and bounded.
# The raw request body never travels.
#
# A ContextVar keeps the hand-off correct per thread and per task, so
# concurrent invocations inside one process cannot see each other's request.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_CONTRACT: ContextVar[Optional[Dict[str, Any]]] = ContextVar("ret_c2_178_caller_contract", default=None)


def set_caller_contract(contract: Optional[Dict[str, Any]]) -> None:
    """Stash the validated caller contract for the imminent inner-graph invoke."""
    _CALLER_CONTRACT.set(dict(contract) if contract else {})


def get_caller_contract() -> Dict[str, Any]:
    """Read (without consuming) the stashed contract; {} when none was set."""
    return _CALLER_CONTRACT.get() or {}
