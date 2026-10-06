"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# On the hosted platform the gateway calls agent.invoke() directly instead.

import json
import os
import re
import secrets
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials_in_value
from shared.secrets import factory as secrets_factory
from shared.utils.audit_logger import emit_trace_event
from src.graph.graph import RetailPrivateKBSearchAgent, runtime_config

app = FastAPI(title="Agent")

# The registry loads config/config.yaml and passes it as Graph(config=...);
# this standalone server mirrors that exactly, so the declared runtime
# parameters are live in both deployments instead of only one.
agent = RetailPrivateKBSearchAgent(config=runtime_config())
agent.compile()
# The secret provider is scoped by the identity the manifest declares, so a
# secret resolves from the same place here and under the registry. The provider
# reads `env/namespaces/{namespace}/…` and `env/agents/{namespace}/{name}/…`,
# so a namespace that disagrees with `config/agent.yaml` silently splits one
# agent's secrets across two stores. `namespace` is lower(industry) — "ret" —
# not the lowercased Template ID.
# tests/integration/test_manifest_identity_alignment.py holds these two values
# to the manifest; it reads both sides rather than restating them.
agent.provision_secrets(secrets_factory(namespace="ret", agent_name="RetailPrivateKBSearchAgent"))

# Upper bound on the serialized structured parameters (bytes). The graph
# enforces per-field bounds — inert alphabets, finite numeric ranges, document
# and text caps; this is the coarse guard that keeps an oversized payload from
# reaching the graph at all.
_MAX_INPUT_CONTEXT_BYTES = 262_144

# Caller identifiers are anchored with fullmatch, never `$`: in Python `$`
# also matches before a trailing newline, so a `^…$` pattern accepts
# "deploy-smoke-001\n", and the session id is written into audit records where
# a newline opens a forged line.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_SAFE_FIELD_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")

# ── Structured-parameter credential screen ───────────────────────────────────
# Why this runs before invoke() rather than inside a node:
#
# The framework's mandatory output gate scans every value of every node result
# for credential patterns, and the backbone's first node copies the structured
# parameters verbatim into its own result. So a credential-shaped string
# anywhere in them makes the FIRST node of the graph fail, before any template
# code runs. What the caller receives is an error status with the domain result
# withheld and no explanation — nothing names the parameter, the field, or the
# reason. On a hosted conversation the same context is replayed every turn, so
# the session never recovers on its own.
#
# The request cannot succeed either way. Refusing it here changes nothing about
# what is accepted; it changes an opaque failure into an actionable one.
#
# The screen calls the SAME detector the gate calls, on the SAME assembled
# object, so what this adapter refuses and what the gate blocks are one set by
# construction — there is no local pattern list that could drift from it.
# Scanning field by field composes exactly to scanning the whole mapping (the
# detector on a mapping is the union over its values), which is what lets the
# refusal name the offending field without widening or narrowing the match.
#
# Field NAMES are caller-controlled too, so a name is repeated back only when
# it is short and inert; anything else is reported by position. The rejected
# value and the matched text are never echoed, in the response or in the audit
# record.


def _field_reference(name: object, index: int) -> str:
    """Render a caller-supplied field name safe to put in a message."""
    if isinstance(name, str) and _SAFE_FIELD_NAME_RE.fullmatch(name) and not detect_credentials_in_value(name):
        return f"input_context.{name}"
    return f"input_context field #{index}"


def screen_input_context(input_context: dict[str, Any]) -> str | None:
    """Return a reference to the first credential-bearing field, else None.

    Walks the top-level fields in caller order and hands each value to the
    framework credential detector, which recurses through nested mappings and
    lists on its own. Only the first offending field is reported: one is enough
    to act on, and the message stays bounded however many fields were sent.
    """
    for index, (name, value) in enumerate(input_context.items(), start=1):
        if detect_credentials_in_value(value):
            return _field_reference(name, index)
    return None


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured invocation parameters: the store's knowledge-base documents
    # and the search tuning — validated field by field inside the graph. This
    # is the channel the documents belong on: the platform rewrites
    # personal-data shapes out of `input` at every node boundary, and its name
    # heuristic reads title-case product and store names as personal names, so
    # document text arrives masked there and intact here.
    input_context: dict[str, Any] | None = None


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone caller auth: when INVOKE_AUTH_TOKEN is set on the server
    # environment, callers that no upstream middleware vouched for (still
    # ANONYMOUS) must present it as a Bearer token and run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary — a deployment-level caller
    # credential, not an agent secret, so the secrets provider does not apply
    # (no invocation context exists before auth).
    #
    # Required here specifically: the request boundary of this agent declares
    # VERIFIED_EXTERNAL, and nothing else sets request.state.trust_level in a
    # standalone deployment. Without this boundary every request would arrive
    # ANONYMOUS, the trust gate would deny it, and the agent would return an
    # error for every call.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was
            # absent, malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    if req.session_id and not _SESSION_ID_RE.fullmatch(req.session_id):
        # 400, not 422: pydantic owns 422 and answers there with a list of
        # error objects, so reusing it would make client handling ambiguous.
        raise HTTPException(
            status_code=400,
            detail="session_id must be 1-128 characters of letters, digits, underscore, dot, colon or hyphen.",
        )

    input_context = req.input_context or {}
    if input_context and len(json.dumps(input_context, default=str)) > _MAX_INPUT_CONTEXT_BYTES:
        raise HTTPException(status_code=413, detail="input_context exceeds the maximum allowed size.")
    offending_field = screen_input_context(input_context)
    if offending_field is not None:
        emit_trace_event(
            "input_context_credential_refused",
            {"field": offending_field},
            {"session_id": req.session_id},
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"{offending_field} contains a credential-shaped value. Remove API keys, "
                "tokens and connection strings from input_context and retry."
            ),
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return agent.invoke(req.input, ctx=ctx, input_context=input_context)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "RetailPrivateKBSearchAgent"}
