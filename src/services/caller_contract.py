"""AgentCore Platform v1.0"""

# Caller-request contract for the private knowledge-base search agent.
#
# One place validates everything a caller can send, so there is exactly one
# answer to "what is accepted?" — the pre_process node calls into here and no
# node downstream re-parses raw request data.
#
# Two request channels reach this module:
#   * the request string (the `input` field): the search query;
#   * `input_context`, the structured invocation parameters: the store's own
#     knowledge-base documents to search, plus the search tuning. Documents
#     belong here rather than in the request string — the platform rewrites
#     personal-data shapes out of the request string at every node boundary,
#     and its name heuristic reads title-case product and store names as
#     personal names, so document text embedded there would arrive masked.
#
# Rules that hold for every field:
#   * every string that renders into the released answer is restricted to an
#     inert alphabet, anchored with \A and \Z. `$` also matches before a trailing
#     newline, so `^[A-Za-z0-9_-]{1,64}$` accepts "SOP-001\n" — and the document
#     id is rendered into a Markdown heading, where a trailing newline is exactly
#     the character that opens a second line and reads as retrieved text;
#   * numbers go through a finite + bounded parser. NaN and the infinities
#     survive float() and compare False against every bound, so an unchecked
#     non-finite value silently passes the check it was meant to fail;
#   * free text (the query, a document chunk) is screened for chat-template
#     control tokens and instruction phrases — raw, and again after markup and
#     invisible characters are stripped — and for credential shapes, using the
#     platform's own detector as the floor plus the local patterns it lacks;
#   * personal-data shapes in free text are replaced by a fixed stub before the
#     text is stored;
#   * a value that fails any check REFUSES the request, naming the field but
#     never repeating the value — and the refusal WORDING is drawn from
#     REFUSAL_REASONS, a set declared in this module. The pre_process node
#     publishes that wording to the caller, so a reason assembled at the point
#     of failure (a rejected value, a caller-supplied key, a caught exception's
#     message) would be an open set on a caller-visible channel;
#   * absent documents are not an error — the search then runs over the
#     built-in baseline corpus. Undeclared top-level keys are ignored, never
#     read and never echoed: the hosted runtime adds its own keys there.

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from framework.security.credential_detector import detect_credentials

# ── Bounds ────────────────────────────────────────────────────────────────────
MAX_QUERY_CHARS = 2000
MAX_DOCUMENTS = 50
MAX_CHUNK_CHARS = 4000
MAX_TOP_K = 20
MAX_CONTEXT_DEPTH = 6

# Declared numeric ranges. Named here rather than written at the call site so
# the refusal wording they produce can be enumerated in REFUSAL_REASONS from
# the same constants: a closed set that restated the bounds would drift.
TOP_K_RANGE: Tuple[int, int] = (1, MAX_TOP_K)
MIN_SCORE_RANGE: Tuple[float, float] = (0.0, 1.0)

# Declared structured parameters. Anything else at the top level is ignored.
DECLARED_CONTEXT_FIELDS: Tuple[str, ...] = ("documents", "top_k", "min_score", "category")
DOCUMENT_FIELDS: Tuple[str, ...] = ("doc_id", "chunk", "category")

# ── Inert alphabets — anchored with \A and \Z, never ^ and $ ─────────────────
_DOC_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")
_CATEGORY_RE = re.compile(r"\A[a-z0-9_]{1,32}\Z")

# Zero-width and bidi controls: invisible in a rendered answer, so they can
# hide a directive from a human reader while a model still reads it.
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
# Markup is stripped before the second screening pass, so a directive spliced
# with tags ("ig<b>nore</b> all previous instructions") is caught once the tags
# are gone — and control tokens are caught on the first pass, before the strip
# could remove them.
_MARKUP_RE = re.compile(r"<[^>]{0,64}>")

# ── Personal-data shapes ──────────────────────────────────────────────────────
# ONE definition, used by both directions of the guarantee: the inbound strip
# that keeps store-staff identifiers out of the search pipeline, and the
# outbound gate that refuses to release anything still matching. Two lists
# would drift, and the drift would always favour the leak.
PERSONAL_DATA_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("email_address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Japanese phone numbers (loose): 0XX-XXXX-XXXX / 0XXXXXXXXXX.
    ("phone_number", re.compile(r"\b0\d{1,4}-?\d{1,4}-?\d{3,4}\b")),
    # Store-staff badge identifiers (EMP-123456, STAFF-001, STORE-2024).
    ("staff_identifier", re.compile(r"\b(?:EMP|STAFF|STORE)-?\d{4,}\b", re.IGNORECASE)),
)
REDACTION_STUB = "[REDACTED]"


def strip_personal_data(text: str) -> str:
    """Replace personal-data shapes in free text with a fixed stub."""
    for _name, pattern in PERSONAL_DATA_PATTERNS:
        text = pattern.sub(REDACTION_STUB, text)
    return text


def find_personal_data(text: str) -> Optional[str]:
    """Name the first personal-data shape present in a string, or None."""
    for name, pattern in PERSONAL_DATA_PATTERNS:
        if pattern.search(text):
            return name
    return None


# ── Credential shapes ─────────────────────────────────────────────────────────
# The platform detector is the FLOOR: every value of every node result is
# scanned with it, and the wrapper discards a node's whole update when it
# raises — so anything the platform recognises must be refused here first.
# The local patterns cover assignment forms and key prefixes the platform's
# value-shape patterns do not; the union is a superset in both directions.
_LOCAL_CREDENTIAL_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]{0,32}PRIVATE KEY-----")),
)


def find_credential(text: str) -> Optional[str]:
    """Name the first credential shape in a string, or None.

    The platform's own detector runs first, so the set refused here is never
    narrower than the set the platform's output gate blocks.
    """
    findings = detect_credentials(text)
    if findings:
        return str(findings[0]["type"])
    for name, pattern in _LOCAL_CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return name
    return None


# ── Disallowed-instruction screen ─────────────────────────────────────────────
# Chat-template control tokens are screened as a CLASS, not as a list of the
# ones seen so far. They are how a payload forges a turn boundary, and they
# carry no meaning in a store's knowledge base, so matching them cannot block
# real work. The platform's own screen scores <|im_start|> and [INST] but not
# <<SYS>> or <|system|>, so the class is covered here rather than assumed.
_CONTROL_TOKEN_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("chat_template_token", re.compile(r"<\|[^<>|]{0,64}\|>")),
    ("instruction_token", re.compile(r"\[/?INST\]", re.IGNORECASE)),
    ("system_token", re.compile(r"<</?SYS>>", re.IGNORECASE)),
    ("role_tag", re.compile(r"<\s*/?(?:system|assistant|user)\s*>", re.IGNORECASE)),
)

# Instruction-shaped phrases. Every pattern requires a verb AND its object, so
# the surrounding text has to actually be an instruction: a procedure that
# mentions rules, prompts or an assistant does not match.
_INSTRUCTION_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[\s\S]{0,40}?"
            r"\b(?:previous|prior|earlier|above|all)\b[\s\S]{0,20}?"
            r"\b(?:instruction|instructions|rule|rules|prompt|prompts|direction|directions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|dump)\b[\s\S]{0,30}?"
            r"\b(?:your|the)\b[\s\S]{0,20}?\b(?:system\s+prompt|instructions|rules)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\byou\s+are\s+(?:now|from\s+now\s+on)\b[\s\S]{0,40}?\b(?:ai|assistant|model|dan|developer\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
)


def screen_disallowed_instructions(text: str) -> Optional[str]:
    """Name the first disallowed-instruction shape in a string, or None.

    Screened twice: once on the text as received, so control tokens are seen
    before any strip could remove them, and once with invisible characters and
    markup removed, so a directive spliced with tags is seen after the text
    re-assembles. A screen that only ran after a strip would silently convert a
    detectable token attack into undetectable plain text.
    """
    for name, pattern in _CONTROL_TOKEN_PATTERNS:
        if pattern.search(text):
            return name
    stripped = _MARKUP_RE.sub("", _INVISIBLE_RE.sub("", text))
    for candidate in (text, stripped):
        for name, pattern in _INSTRUCTION_PATTERNS:
            if pattern.search(candidate):
                return name
    return None


def _walk_strings(value: Any, depth: int = 0) -> List[str]:
    """Collect every string leaf AND every mapping key, depth-first.

    Keys are collected because a hostile field NAME is caller data on the same
    footing as its value, and a screen that read values only would pass it.
    """
    if depth > MAX_CONTEXT_DEPTH:
        return []
    if isinstance(value, str):
        return [value]
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                found.append(key)
            found.extend(_walk_strings(item, depth + 1))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_walk_strings(item, depth + 1))
    return found


def screen_structure(value: Any) -> Optional[str]:
    """Screen a parsed structure depth-first, keys included.

    Runs after parsing, so a payload that escaped its control tokens as \\u
    sequences is screened in its decoded form.
    """
    for text in _walk_strings(value):
        hit = screen_disallowed_instructions(text)
        if hit:
            return hit
    return None


# ── Validation error ──────────────────────────────────────────────────────────


class ContractError(ValueError):
    """A caller value failed its check. Carries the field, never the value.

    Three attributes, ALL authored by this module:

      field   the contract path of the offending value. Declared names plus a
              `documents[N]` index — never a caller-supplied key.
      reason  the refusal wording, drawn from REFUSAL_REASONS below. It is the
              only part that reaches the caller, so it is a closed set: no
              caller value, no caller-supplied key, no text from outside here.
      detail  the pattern label behind a screening refusal, for the audit trail
              only. Empty for every other refusal.

    They exist so the refusal notice can be rebuilt from declared parts instead
    of interpolating the caught exception: the pre_process node publishes a
    notice to the caller, and an interpolated exception is exactly how text
    nobody declared ends up in it.
    """

    def __init__(self, field: str, reason: str, detail: str = "") -> None:
        self.field = field
        self.reason = reason
        self.detail = detail
        super().__init__(f"{field}: {reason}")


# ── The closed set of refusal wordings ────────────────────────────────────────
# Every reason a ContractError can carry is declared here, and the pre_process
# node holds the caller-visible notice to this set. The reason is the only part
# of a refusal that reaches the caller, so it has to be a set this module chose
# and not a sentence assembled at the point of failure: a reason that quoted the
# rejected value, or a caller-supplied field NAME, would leave the set open —
# and an open set is what "the ERROR envelope publishes composed text" means.
#
# Where a bound appears in the wording it is interpolated from the module
# constant that enforces it, so the declared set cannot drift from the check.
REASON_NOT_A_NON_EMPTY_STRING = "must be a non-empty string"
REASON_QUERY_TOO_LONG = f"must be at most {MAX_QUERY_CHARS} characters"
REASON_CHUNK_TOO_LONG = f"must be at most {MAX_CHUNK_CHARS} characters"
REASON_DISALLOWED_INSTRUCTION = "contains a disallowed instruction pattern"
REASON_CREDENTIAL_SHAPED = "contains a credential-shaped value"
REASON_NOT_AN_OBJECT = "must be an object"
# The declared field names are this module's own; the UNDECLARED name the
# caller sent is not named at all. Naming it was the one place a caller string
# travelled back out, and an inert-alphabet mask does not close that: a staff
# badge identifier or a Japanese phone number is spelled entirely in the inert
# alphabet, so it passed the mask and rode the refusal notice to the caller.
REASON_UNDECLARED_DOCUMENT_FIELD = f"must carry only the declared fields ({', '.join(DOCUMENT_FIELDS)})"
REASON_MALFORMED_DOC_ID = "must be 1-64 characters of letters, digits, underscore or hyphen"
REASON_MALFORMED_CATEGORY = "must be 1-32 characters of lowercase letters, digits or underscore"
REASON_NOT_A_NUMBER = "must be a number"
REASON_BOOLEAN_NOT_A_NUMBER = "must be a number, not a boolean"
REASON_NOT_FINITE = "must be a finite number"
REASON_NOT_A_WHOLE_NUMBER = "must be a whole number"
REASON_NOT_A_DOCUMENT_LIST = "must be a list of documents"
REASON_TOO_MANY_DOCUMENTS = f"must hold at most {MAX_DOCUMENTS} documents"
REASON_DUPLICATE_DOC_ID = "must be unique within the request"
#: What a caller sees when a reason is NOT one of the declared set. Nothing
#: below can reach it; it exists so that a raise site added later which forgets
#: the set degrades to a label instead of publishing its own sentence.
REASON_CONTRACT_VIOLATION = "does not satisfy the request contract"


def range_reason(low: float, high: float) -> str:
    """The wording for an out-of-range number, built from the bounds."""
    return f"must be between {low:g} and {high:g}"


#: The whole set, for the node that publishes it and the tests that pin it.
REFUSAL_REASONS: "frozenset[str]" = frozenset(
    {
        REASON_NOT_A_NON_EMPTY_STRING,
        REASON_QUERY_TOO_LONG,
        REASON_CHUNK_TOO_LONG,
        REASON_DISALLOWED_INSTRUCTION,
        REASON_CREDENTIAL_SHAPED,
        REASON_NOT_AN_OBJECT,
        REASON_UNDECLARED_DOCUMENT_FIELD,
        REASON_MALFORMED_DOC_ID,
        REASON_MALFORMED_CATEGORY,
        REASON_NOT_A_NUMBER,
        REASON_BOOLEAN_NOT_A_NUMBER,
        REASON_NOT_FINITE,
        REASON_NOT_A_WHOLE_NUMBER,
        REASON_NOT_A_DOCUMENT_LIST,
        REASON_TOO_MANY_DOCUMENTS,
        REASON_DUPLICATE_DOC_ID,
        REASON_CONTRACT_VIOLATION,
        # The only two ranges this contract declares (see TOP_K_RANGE /
        # MIN_SCORE_RANGE); both call sites pass them, so the set is complete.
        range_reason(*TOP_K_RANGE),
        range_reason(*MIN_SCORE_RANGE),
    }
)


# ── Scalar parsers ────────────────────────────────────────────────────────────


def finite_in_range(value: Any, field: str, low: float, high: float) -> float:
    """Parse a caller number, or refuse.

    Rejects booleans (isinstance(True, int) is True in Python, so an unchecked
    parser reads `true` as 1), non-numeric strings, NaN and both infinities,
    and magnitudes outside the declared range. NaN is the one that matters
    most: it parses through float() and then compares False against every
    bound, so a parser that only range-checked would let it through as "not
    out of range".
    """
    if isinstance(value, bool):
        raise ContractError(field, REASON_BOOLEAN_NOT_A_NUMBER)
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            # `from None` on purpose: the caught exception's message quotes the
            # rejected string, and this refusal is published to the caller.
            raise ContractError(field, REASON_NOT_A_NUMBER) from None
    else:
        raise ContractError(field, REASON_NOT_A_NUMBER)
    if not math.isfinite(parsed):
        raise ContractError(field, REASON_NOT_FINITE)
    if not (low <= parsed <= high):
        raise ContractError(field, range_reason(low, high))
    return parsed


def bounded_integer(value: Any, field: str, low: int, high: int) -> int:
    """Parse a caller integer through the finite parser, refusing fractions."""
    parsed = finite_in_range(value, field, low, high)
    if parsed != int(parsed):
        raise ContractError(field, REASON_NOT_A_WHOLE_NUMBER)
    return int(parsed)


def is_inert_doc_id(value: Any) -> bool:
    """True when *value* is a document id over the closed inert alphabet."""
    return isinstance(value, str) and bool(_DOC_ID_RE.match(value))


def is_inert_category(value: Any) -> bool:
    """True when *value* is a category code over the closed inert alphabet."""
    return isinstance(value, str) and bool(_CATEGORY_RE.match(value))


# ── Free-text screening ───────────────────────────────────────────────────────


def screen_free_text(text: str, field: str) -> str:
    """Screen one free-text field and return its stored form, or refuse.

    Order matters: control tokens and credential shapes are checked on the raw
    text, before the personal-data strip could alter them; the strip runs last
    and its output is what the pipeline stores.
    """
    hit = screen_disallowed_instructions(text)
    if hit:
        # The matched pattern's NAME travels on `detail`, for the audit trail.
        # It is not part of the reason: the credential half of this screen
        # names patterns the platform detector declares, not this module, and
        # the reason is the half that reaches the caller.
        raise ContractError(field, REASON_DISALLOWED_INSTRUCTION, hit)
    hit = find_credential(text)
    if hit:
        raise ContractError(field, REASON_CREDENTIAL_SHAPED, hit)
    return strip_personal_data(text)


# ── Request validation ────────────────────────────────────────────────────────


def _validate_document(raw: Any, index: int) -> Dict[str, Any]:
    """Validate one knowledge-base document into its stored form."""
    where = f"input_context.documents[{index}]"
    if not isinstance(raw, Mapping):
        raise ContractError(where, REASON_NOT_AN_OBJECT)

    for key in raw.keys():
        if key not in DOCUMENT_FIELDS:
            # The caller's key is NOT named back. It reaches the refusal notice
            # the pre_process node publishes, and an inert-alphabet mask is not
            # a closed set: `EMP-123456` and `080-1234-5678` are spelled in the
            # inert alphabet. The declared field names say the same thing.
            raise ContractError(where, REASON_UNDECLARED_DOCUMENT_FIELD)

    doc_id = raw.get("doc_id")
    if not is_inert_doc_id(doc_id):
        raise ContractError(f"{where}.doc_id", REASON_MALFORMED_DOC_ID)

    chunk = raw.get("chunk")
    if not isinstance(chunk, str) or not chunk.strip():
        raise ContractError(f"{where}.chunk", REASON_NOT_A_NON_EMPTY_STRING)
    if len(chunk) > MAX_CHUNK_CHARS:
        raise ContractError(f"{where}.chunk", REASON_CHUNK_TOO_LONG)
    stored_chunk = screen_free_text(chunk.strip(), f"{where}.chunk")

    category = raw.get("category")
    if category is not None and not is_inert_category(category):
        raise ContractError(f"{where}.category", REASON_MALFORMED_CATEGORY)

    return {"doc_id": doc_id, "chunk": stored_chunk, "category": category or ""}


def validate_request(user_input: Any, input_context: Any = None) -> Dict[str, Any]:
    """Validate a whole request and return the contract the pipeline runs on.

    Raises ContractError naming the offending field. The caller's raw values
    never appear in the message and never leave this function except in their
    validated, inert or stripped forms.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ContractError("input", REASON_NOT_A_NON_EMPTY_STRING)
    query = user_input.strip()
    if len(query) > MAX_QUERY_CHARS:
        raise ContractError("input", REASON_QUERY_TOO_LONG)
    stored_query = screen_free_text(query, "input")

    context: Mapping[str, Any] = input_context if isinstance(input_context, Mapping) else {}

    documents: List[Dict[str, Any]] = []
    raw_documents = context.get("documents")
    if raw_documents is not None:
        if not isinstance(raw_documents, (list, tuple)):
            raise ContractError("input_context.documents", REASON_NOT_A_DOCUMENT_LIST)
        if len(raw_documents) > MAX_DOCUMENTS:
            raise ContractError("input_context.documents", REASON_TOO_MANY_DOCUMENTS)
        seen: set[str] = set()
        for index, raw in enumerate(raw_documents):
            document = _validate_document(raw, index)
            if document["doc_id"] in seen:
                raise ContractError(f"input_context.documents[{index}].doc_id", REASON_DUPLICATE_DOC_ID)
            seen.add(document["doc_id"])
            documents.append(document)

    top_k: Optional[int] = None
    if context.get("top_k") is not None:
        top_k = bounded_integer(context["top_k"], "input_context.top_k", *TOP_K_RANGE)

    min_score: Optional[float] = None
    if context.get("min_score") is not None:
        min_score = finite_in_range(context["min_score"], "input_context.min_score", *MIN_SCORE_RANGE)

    category: Optional[str] = None
    if context.get("category") is not None:
        if not is_inert_category(context["category"]):
            raise ContractError("input_context.category", REASON_MALFORMED_CATEGORY)
        category = str(context["category"])

    return {
        "query": stored_query,
        "documents": documents,
        "top_k": top_k,
        "min_score": min_score,
        "category": category,
        "document_source": "input_context" if documents else "baseline",
    }
