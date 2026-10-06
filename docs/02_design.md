# Design Document — RET-C2-178

**Template ID:** RET-C2-178
**Template Name:** RetailPrivateKBSearchAgent
**Category:** Cat 2 (domain-specific pipeline)
**Industry:** RET (Retail)
**Inheritance:** `AgentBaseGraph` (framework base class — direct inheritance)
**Pattern:** two-layer nested (outer backbone + inner search workflow)

| Role | Class |
|---|---|
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |

## 1. Overview

RetailPrivateKBSearchAgent takes a plain-language question and the store's own
knowledge-base documents, ranks the documents by how many of the question's
search terms each one contains, masks supplier-confidential signals in the
matching text, and returns a Markdown answer. Store-staff identifiers are
stripped from the question and from every document at the request boundary;
the output boundary refuses to release anything still matching those shapes or
a credential shape.

The documents arrive with the request as structured invocation parameters and
are searched in memory. An on-device vector index would slot into the search
step in place of the term-overlap ranking; nothing above or below that step
depends on how the ranking is produced.

## 2. Architecture

```
OUTER backbone (AgentBaseGraph — fixed; add_edges() NOT overridden)
  START -> initialize -> pre_process -> main -> post_process -> finalize -> END
                                         |
                                         v  (KBSearchGraphNode.get_subgraph)
INNER search workflow (DomainWorkflowGraph : BaseGraph)
  START -> query_normalize -> kb_search -> privacy_filter -> response_format -> END
```

- The `main` slot is **`KBSearchGraphNode`**, a `GraphNode` subclass. It
  delegates the whole workflow to `DomainWorkflowGraph`.
- `merge_output()` maps the inner `formatted_response` into outer state as both
  `formatted_response` and **`result`** — the field the output gate reads and
  the envelope surfaces.
- The inner graph is constructed with the live runtime tuning; its domain
  nodes take no constructor arguments and read their tuning from seeded state.

### Directory layout

| Path | Role |
|---|---|
| `src/graph/graph.py` | outer graph, main-slot node, runtime-config loader |
| `src/graph/domain_workflow_graph.py` | inner graph: four steps, linear edges |
| `src/graph/context_bridge.py` | carries the validated request across the boundary |
| `src/services/caller_contract.py` | the single definition of what a caller may send |
| `src/nodes/pre_process_node.py` | request boundary |
| `src/nodes/post_process_node.py` | output boundary |
| `src/nodes/query_normalize_node.py` | search step 1 — acronym expansion, whitespace |
| `src/nodes/leann_search_node.py` | search step 2 — ranking (`KBSearchNode`) |
| `src/nodes/privacy_filter_node.py` | search step 3 — confidential-signal masking |
| `src/nodes/response_format_node.py` | search step 4 — Markdown rendering |
| `src/schemas/state.py` | flat `State(AgentState)` + JSON helpers |
| `config/agent.yaml` | static manifest (identity, entry point, requirements) |
| `config/config.yaml` | runtime parameters |
| `src/api/server.py` | HTTP adapter |

## 3. Request contract

| Field | Content |
|---|---|
| `input` | the question — a non-empty string of at most 2000 characters |
| `input_context.documents` | at most 50 objects of exactly `doc_id`, `chunk` and optional `category` |
| `input_context.top_k` | whole number 1–20; overrides the configured default |
| `input_context.min_score` | number 0–1; overrides the configured default |
| `input_context.category` | inert code; restricts the search to documents of that category |

Undeclared top-level keys are ignored — never read, never echoed. The hosted
runtime adds its own keys there, so a refusal on an unknown key would refuse
every hosted turn.

Rules that hold for every field:

| Rule | Detail |
|---|---|
| Inert identifiers | `doc_id` is `[A-Za-z0-9_-]{1,64}`, `category` is `[a-z0-9_]{1,32}`. Both patterns are anchored with `\A` and `\Z`, never `^` and `$`: `$` also matches before a trailing newline, so a `^…$` pattern accepts `"SOP-001\n"`, and the id is rendered into a Markdown heading where a newline opens a second line that reads as retrieved text. |
| Finite numbers | `top_k` and `min_score` — real or string-shaped — go through a finite + bounded parser. `NaN` and the infinities parse through `float()` and compare False against every bound, so an unchecked value passes exactly the check it should fail. Booleans are refused (`isinstance(True, int)` is true). |
| Structural caps | 2000 characters of question, 50 documents, 4000 characters per document, 256 KB of structured parameters at the adapter. |
| Disallowed instructions | chat-template control tokens (`<\|…\|>`, `[INST]`, `<<SYS>>`, `<system>`) are screened as a class, raw and again after markup and invisible characters are stripped; instruction-shaped phrases require a verb and its object, so ordinary store prose is unaffected. Applied to the question and to every document, keys included. |
| Credential shapes | refused at the adapter (structured parameters, using the platform's own detector) and at the request boundary (question and document text, using the platform's detector plus the local patterns it lacks). |
| Personal-data shapes | e-mail addresses, phone numbers and staff badge numbers are replaced by `[REDACTED]` in the question and in every document before anything is stored. |
| Refusals | closed-set only. The notice is written to `formatted_output`, and an ERROR status routes straight to `finalize`, so it is what the caller reads. It is assembled from a field label in `REFUSAL_FIELD_LABELS` and a reason in the contract's `REFUSAL_REASONS` — both declared in code. The rejected value, the caller's own field NAME, the document index and the matched pattern label are not part of it; the full path and the pattern go to the audit event. |

The platform's own input screen runs before the request boundary and masks
personal-data shapes in the question with `[MASKED]`; its name heuristic reads
title-case runs (product and store names) as personal names. That is why the
documents belong in the structured parameters, which the platform does not
rewrite. A question masked in its entirety yields no search terms; the answer
then says so rather than ranking documents on the word "masked".

## 4. Ranking

Search terms are the question's distinct words of three or more characters
after normalization, minus a short stop-word list and minus the masking
sentinels. A document's score is the fraction of those terms that occur in its
text, 0 to 1. Documents at or above `min_score` are returned in descending score
order (ties by id), at most `top_k` of them. A question with no search terms
returns no documents and the answer states that no searchable terms remained.

The ranking depends on the input in both directions: a different question
re-orders the same documents, and a different document set re-orders the same
question. The integration suite proves both.

## 5. Node design

| Node | Layer | Responsibility | Writes |
|---|---|---|---|
| `PreProcessNode` | outer pre_process | validate the whole request against the caller contract; refuse with the field named | `validated_input`, `caller_contract`, `enriched_context`, `status` |
| `KBSearchGraphNode` | outer main | bridge the validated contract, run the inner workflow, map its response to `result` | `formatted_response`, `result`, `status` |
| `QueryNormalizeNode` | inner 1 | expand retail acronyms, strip punctuation runs, collapse whitespace to one line | `normalized_query` |
| `KBSearchNode` | inner 2 | rank the caller's documents (or the baseline corpus) against the search terms; honour `top_k`, `min_score`, `category` | `search_results`, `search_summary` |
| `PrivacyFilterNode` | inner 3 | mask cost prices, vendor codes and margins in the matching text | `filtered_results` |
| `ResponseFormatNode` | inner 4 | render the Markdown answer; emit the terminal success status that routes the run through the output gate | `formatted_response`, `status` |
| `PostProcessNode` | outer post_process | scan the released text; on a violation clear every output-bearing field and return a truthy notice | `formatted_output`, `result`, `formatted_response`, `status` |

Every node is a `FunctionNode` subclass returning a partial update.
`PreProcessNode` declares `VERIFIED_EXTERNAL` — the level the manifest admits;
the inner nodes and the output gate declare `ANONYMOUS`, because the trust
decision is taken once at the request boundary and the inner graph runs under
the same caller context.

### Security layers

| Layer | Where | Implementation |
|---|---|---|
| S-1 trust gate | framework, every node | `caller_trust_level` checked against each node's declaration before `execute()` |
| S-2 input screen | framework, then `PreProcessNode` | platform masking and injection policy on `user_input`; the template's own contract screen refuses what the platform does not score |
| S-3 output gate | framework, then `PostProcessNode` | platform credential scan on every node result; the template's gate refuses the union of platform and local shapes and clears on violation |
| S-4 audit | every node | one domain event per node via `emit_trace_event(name, payload, state)`; payloads carry counts and labels, never caller values |
| S-5 secrets | `src/api/server.py` | `provision_secrets()` under the manifest's namespace and name; no secret is read by template code (`requires.secrets: []`) |

## 6. State

Flat `State(AgentState)` (`src/schemas/state.py`). Structured fields are stored
as JSON strings — checkpoints are serialized with msgpack, and a bare container
there corrupts silently.

| Field | Producer | Notes |
|---|---|---|
| `validated_input` | PreProcessNode | screened, stripped question |
| `caller_contract` | PreProcessNode | JSON — validated documents, tuning, category, document source |
| `search_config` | inner initial-state hook | JSON — the live defaults from `config/config.yaml` |
| `normalized_query` | QueryNormalizeNode | single-line normalized question |
| `search_results` | KBSearchNode | JSON — ranked results `[{doc_id, chunk, score, category}]` |
| `search_summary` | KBSearchNode | JSON — searchable terms, corpus size and source, effective tuning |
| `filtered_results` | PrivacyFilterNode | JSON — the same results with confidential signals masked |
| `formatted_response` | ResponseFormatNode | the Markdown answer |
| `result` | KBSearchGraphNode | alias of `formatted_response` read by the output gate |
| `trace_id` / `correlation_id` | framework | tracing only |

No credential, secret or model object is present in state.

## 7. Runtime configuration

`config/config.yaml` carries the live parameters.

| Key | Reader | Effect |
|---|---|---|
| `max_retry` | the backbone | retry ceiling; validated at compile time, an out-of-range value refuses to start |
| `timeout_s` | none in this template | declared by the template contract; neither the template code nor the framework's graph runtime reads it |
| `search.top_k` | search step | documents returned when the caller sends no `top_k` |
| `search.min_score` | search step | relevance floor when the caller sends no `min_score` |

The path a value travels: the registry (or the HTTP adapter, which mirrors it)
loads the file and passes it to the graph constructor; the main slot forwards
the search block to the inner graph; the inner graph's initial-state hook seeds
it into inner state. Node `execute()` methods take no config argument, so state
seeding is the only route a declared value can reach a domain node. A malformed
search block is refused at compile time rather than coerced to a default nobody
declared. The integration suite proves both declared values visibly change the
released answer, and that a caller's bounded value overrides them.

## 8. The output boundary

The stated invariant: **nothing credential-shaped and no store-staff
personal-data shape leaves the agent.**

The agent renders no monetary aggregates — the answer carries document ids,
relevance percentages and document text with cost figures already masked — so
a monetary rounding grid does not apply. The identifier invariant above is what
is enforced instead, and document ids such as `SOP-001` and `INC-101` — the
`<letters>-<digits>` shape a currency grid misreads — come out byte-identical.

The credential half takes **the platform's own detector as the floor** and adds
the local patterns it lacks (assignment forms such as `password=…`, `pk-`/`ak-`
key prefixes, private-key headers). The platform scans every value of every
node result and RAISES when it finds a credential; the wrapper then discards
the gate node's whole return value — the clearing included — and the envelope
falls back to the un-gated response still in state. A local list narrower than
the platform's is therefore not a weaker filter but a containment bypass.

Two independent layers, each with its own audit event:

- **inbound** — the request boundary strips personal-data shapes from the
  question and every document and refuses credential shapes there;
- **outbound** — this gate refuses to release anything still matching.

Both read ONE pattern definition (`src/services/caller_contract.py`), so they
cannot drift apart.

### Containment

`AgentBaseGraph.get_output` resolves the released output as
`formatted_output or result`, with no status check. Three consequences, all
handled:

1. A falsy `formatted_output` re-opens the fallback, so the withheld notice is
   truthy, and so is the notice for an empty response.
2. A gate that raises leaks, because the wrapper turns an exception into a bare
   error update that clears nothing — which is why the gate's detector is never
   narrower than the platform's.
3. On a violation the gate clears `result`, `formatted_response` and
   `formatted_output` in the same update, so no path out of the graph reaches
   the un-gated text.

`get_output` is deliberately NOT overridden. The output gate is the single
place the released text is checked; a second layer in the envelope would
contain a leak by itself and make the gate's own clearing unfalsifiable. The
containment suite proves the gate is load-bearing by injecting the fault on the
data path — the main slot publishing a drifted response — and the original
narrow gate fails that suite: the platform raises inside post-process, the
node's update is discarded, and the envelope surfaces the response with the
credential in it.

## 9. Design decision record

| Decision | Chosen | Rationale |
|---|---|---|
| Base class | `AgentBaseGraph` | fixed domain pipeline — not autonomous |
| Composition | `GraphNode` subgraph (nested) | multi-step workflow encapsulated in an inner `BaseGraph` |
| Inner topology | linear (4 nodes) | each step is sequential and deterministic |
| Document channel | `input_context.documents` | the platform rewrites personal-data shapes and title-case names out of the question string; structured parameters arrive intact |
| Ranking | term-overlap fraction | deterministic, explainable, input-dependent; the index integration point is one function |
| Absent documents | baseline corpus, stated in the answer | a request without documents still produces a real, labelled answer |
| Identifier anchoring | `\A…\Z` | `$` admits a trailing newline into a rendered heading |
| Output gate | union of platform and local detectors; clear-and-notice on violation; no envelope override | a narrower gate is a bypass; a second layer masks the first |
| Trust levels | request boundary `VERIFIED_EXTERNAL`, inner nodes `ANONYMOUS` | one trust decision at the boundary; inner nodes never demand more than the entry contract admits |
