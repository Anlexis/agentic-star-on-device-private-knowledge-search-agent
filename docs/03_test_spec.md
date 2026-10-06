# Test Specification — RET-C2-178

**Template:** RET-C2-178 — RetailPrivateKBSearchAgent
**Category:** Cat 2 (two-layer nested) · **Inheritance:** `AgentBaseGraph`

The tables below describe the tests that ship in this repository. Every case is
deterministic: no model call, no network, no index.

## 1. Test files

| Layer | Target | File |
|---|---|---|
| Unit | the caller contract (every accepted and refused shape) | `tests/unit/test_caller_contract.py` |
| Unit | the domain nodes and the two backbone slots | `tests/unit/test_nodes.py` |
| Unit | graph composition, config plumbing, the context bridge | `tests/unit/test_graph_composition.py` |
| Unit | framework compliance (the default gates cannot be replaced) | `tests/unit/test_framework_compliance_tc06_tc07.py` |
| Boundary | the request boundary refuses on its own account | `tests/proof_of_boundary/test_request_boundary.py` |
| Boundary | the output boundary and its clearing | `tests/proof_of_boundary/test_output_boundary.py` |
| Boundary | backbone invoke order | `tests/proof_of_boundary/test_pb_invoke_order.py` |
| Boundary | import isolation | `tests/proof_of_boundary/test_import_isolation.py` |
| Boundary | state safety | `tests/proof_of_boundary/test_state_safety.py` |
| Boundary | human-review interrupt propagation (skipped: no HITL) | `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` |
| Integration | the real HTTP entry point, end to end | `tests/integration/test_invoke_end_to_end.py` |
| Integration | error-envelope containment | `tests/integration/test_error_envelope_containment.py` |
| Integration | manifest identity vs the entrypoint's provisioned identity | `tests/integration/test_manifest_identity_alignment.py` |

`tests/integration/asgi.py` is a small synchronous driver for the application
under test, not a test module. The integration suite reads
`deploy/invoke_payload.json` as its fixture, so the smoke payload and the tests
assert the same contract.

## 2. The caller contract

| Case | Input | Expected |
|---|---|---|
| CC-01 | a question alone | accepted; the baseline corpus is searched |
| CC-02 | documents and tuning as structured parameters | accepted; every document validated |
| CC-03 | numeric strings for `top_k` / `min_score` | parsed |
| CC-04 | undeclared top-level keys (e.g. conversation history) | ignored, never refused |
| CC-05 | empty, whitespace-only or non-string question | refused, field named |
| CC-06 | question over 2000 characters, more than 50 documents, a document over 4000 characters | refused, field named |
| CC-07 | a document with an unexpected field | refused naming the DECLARED fields; the caller's key is never repeated back, whatever it is spelled with |
| CC-08 | duplicate `doc_id` | refused |
| CC-09 | `doc_id` with a trailing newline, a space, a heading marker, or over 64 characters | refused — the anchors are `\A`/`\Z`, and the newline case is pinned explicitly |
| CC-10 | `category` outside `[a-z0-9_]{1,32}` (including a trailing newline) | refused |
| CC-11 | `NaN`, `Infinity`, `-Infinity`, an overflowing literal, a boolean, a non-numeric string, a list — as a number and as a string, for both numeric fields | refused, field named |
| CC-12 | `top_k` out of 1–20 or fractional; `min_score` outside 0–1 | refused |
| CC-13 | `null` for a numeric field | treated as absent |
| CC-14 | chat-template control tokens: `<\|im_start\|>`, `<\|system\|>`, `[INST]`, `<<SYS>>`, `<system>` | refused |
| CC-15 | instruction-shaped phrases, a directive spliced with markup, a directive hidden with zero-width characters | refused |
| CC-16 | a hostile document field name | refused |
| CC-17 | ordinary store questions containing "rules", "prompt", "assistant", "act as a", "system:" | accepted |
| CC-18 | a credential shape in the question or in a document (platform-recognised and local-only shapes) | refused, field named, value never repeated |
| CC-19 | an e-mail address, phone number or staff badge in the question or a document | accepted; replaced by `[REDACTED]` before storage |
| CC-20 | store vocabulary with times, temperature ranges and counts | survives the strip unchanged |

## 3. Nodes

| Case | Target | Expected |
|---|---|---|
| ND-01 | normalize | acronyms expanded; newlines and punctuation runs collapse to one line |
| ND-02 | search terms | stop words, short tokens and the masking sentinels are excluded |
| ND-03 | score | the fraction of terms found; 0 for no terms |
| ND-04 | ranking | descending score, ties by id; `top_k` and `min_score` bound the result; the category filter restricts the corpus |
| ND-05 | search node | caller documents are searched; absent documents fall back to the baseline corpus and the summary says so |
| ND-06 | search node | a different question produces a different ranking |
| ND-07 | search node | the caller's tuning overrides the declared defaults; the declared defaults apply when the caller sends none; malformed config values fall back within bounds |
| ND-08 | search node | a fully masked question ranks nothing and reports zero searchable terms |
| ND-09 | privacy filter | cost prices, supplier ids, vendor codes and margins are masked; "vending machine", "the vendor arrives", "cost centre" are left alone |
| ND-10 | response format | id, relevance percentage, category tag and corpus source are rendered; no-results and no-terms messages; the question line cannot open a new line |
| ND-11 | request boundary slot | the contract is a JSON string; `enriched_context` carries no caller string; a refusal writes a truthy notice and clears `result` |
| ND-12 | output boundary slot | a clean response is released unchanged; an empty response is withheld with a truthy notice |
| ND-13 | trust levels | every node declares its own; the request boundary admits verified external callers; anonymous callers are denied there |

## 4. Composition and configuration

| Case | Expected |
|---|---|
| GC-01 | the outer graph fills all five backbone slots with the expected classes; the alias resolves |
| GC-02 | `get_output` is not overridden |
| GC-03 | `config/config.yaml` is loaded; `max_retry` is validated at compile time |
| GC-04 | the search block is forwarded to the inner graph and seeded into its state |
| GC-05 | a malformed search block is refused at compile time; an absent one is not an error |
| GC-06 | the validated contract crosses the boundary on the bridge and is read by the inner initial-state hook |
| GC-07 | `merge_output` maps the response to both `formatted_response` and `result`; a non-success inner status publishes nothing |
| GC-08 | the inner graph registers four steps, withholds the response on failure, and annotates `route` with its own `State` |
| GC-09 | a full invocation at the declared trust level succeeds over caller documents and over the baseline, and traverses the output gate |

## 5. The output boundary

| Case | Released content | Expected |
|---|---|---|
| OB-01 | every credential shape the platform detector recognises | refused by this gate first, with a reason label |
| OB-02 | shapes only the local patterns catch (assignment forms, key prefixes, private-key headers) | refused |
| OB-03 | a leak nested inside a mapping or a list | refused |
| OB-04 | a clean nested structure | released (the control that keeps OB-03 honest) |
| OB-05 | an e-mail address, phone number or staff badge | refused |
| OB-06 | the whole baseline corpus, rendered | released; every document id and text byte-identical; structural tokens (`07:45`, `5-25°C`, `48 units`) survive |
| OB-07 | a violation | `result`, `formatted_response` and `formatted_output` are PRESENT in the returned update; the first two cleared, the third a truthy notice |
| OB-08 | an empty response | a truthy notice and an error status, never an empty value |

## 6. End to end, through the HTTP entry point

| Case | Expected |
|---|---|
| E2E-01 | the committed smoke payload succeeds at the declared trust level with a non-empty answer |
| E2E-02 | the ranking is computed from the caller's documents; the relevance figures match the term-overlap fractions; the corpus source is stated |
| E2E-03 | a different question produces a different answer; a category filter restricts it; a request without documents searches the baseline and says so |
| E2E-04 | no staff identifier from a document reaches the answer |
| E2E-05 | the backbone reaches the output gate (`node_history` pinned) |
| E2E-06 | a missing or wrong caller credential is refused, indistinguishably |
| E2E-07 | the declared `top_k` and `min_score` visibly change the answer; a caller's `top_k` overrides the declared default |
| E2E-08 | a credential-shaped structured parameter is refused with the field named (or by position when the name is not inert) and never echoed; ordinary text on the same field passes; oversized parameters are refused |
| E2E-09 | a `session_id` or `doc_id` with a trailing newline is refused; the same value without it is accepted; no forged heading is rendered |
| E2E-10 | a control token the platform does not score is refused with a readable reason; one the platform scores is still refused; a credential-shaped question is refused naming the field |
| E2E-11 | every refusal carries no traceback and no source path |

## 7. Error-envelope containment

The envelope resolves the released output as `formatted_output or result`, with
no status check, so a gate that raised — or that set an error status without
clearing — would ship the un-gated response inside the error envelope.

| Case | Expected |
|---|---|
| EC-01 | with a drifted response on the DATA path (a platform-recognised shape, and a local-only shape), the envelope carries no leaked value |
| EC-02 | the status is an error and the output is the withheld notice |
| EC-03 | the gate node appears in `node_history`, proving the block happened there |
| EC-04 | no traceback and no source path reach the surface |
| EC-05 | without the drift, the same request still returns its real answer |

## 7-bis. The refusal envelope is a closed set

An ERROR status routes straight to `finalize`, so the request boundary's
refusal — not the output gate's notice — is what a refused caller reads. It
must therefore be assembled from declared values only.

| Case | Expected |
|---|---|
| RE-01 | with the contract refusing on a field path and a reason that both carry a sentinel, the sentinel appears nowhere in the node's returned mapping — walking every nested key and value |
| RE-02 | the same sentinel appears nowhere in the invoke body, through the real ASGI `/invoke` |
| RE-03 | the notice is still `Request refused — <label>: <reason>` with the label in `REFUSAL_FIELD_LABELS` and the reason in `REFUSAL_REASONS` — containment, not silence |
| RE-04 | an undeclared document key is never repeated back, including one spelled in the inert alphabet (`EMP-123456`, `080-1234-5678`) |
| RE-05 | over EVERY refusal path: the notice is prefix + declared label + declared reason, truthy, with `result` cleared and `error_log` carrying that one line |
| RE-06 | over EVERY refusal path: no document index reaches the caller |
| RE-07 | every declared path reduces to its declared label; anything undeclared falls back rather than being echoed |
| RE-08 | every refusal the contract can raise carries a reason drawn from `REFUSAL_REASONS`, and the declared set contains no wording the paths cannot produce (bar the unused fallback) |

The fault is injected on the data path — the main slot publishing a drifted
response — never on the gate: patching the gate would test the patch rather
than the agent.

## 8. Identity alignment

| Case | Expected |
|---|---|
| ID-01 | the manifest declares `namespace`, `name` and `industry` |
| ID-02 | the namespace the entrypoint provisions secrets under equals the manifest's `namespace` (both read from the files) |
| ID-03 | the agent name the entrypoint provisions equals the manifest's `name` |
| ID-04 | the manifest's `namespace` is `lower(industry)` |
| ID-05 | the manifest's dotted class path names the class the adapter serves |

## 9. Release thresholds

- Every case above passes under the framework wheel the build pipeline installs.
- The output invariant holds for every representation probed, in both
  directions: leak forms refused, ordinary store text released unchanged.
- No store-staff identifier and no credential shape is present in any state
  field at any stage.
