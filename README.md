# On-Device Private Knowledge Search Agent

AI agent for searching private store knowledge bases on-device, built with Agentic Star.

> **Category**: Cat 2 (a domain-specific pipeline for one job-to-be-done)
> **Industry**: Retail
> **Template ID**: RET-C2-178

## Overview

Store staff need answers from their own operating documents — opening and
closing procedures, product specifications, incident templates, supplier
protocols — without those documents, or the questions asked of them, leaving
the store's environment.

This agent takes a plain-language question and the store's knowledge-base
documents, ranks the documents by how many of the question's search terms each
one contains, and returns a short Markdown answer: the matching documents, a
relevance figure for each, and the matching text. Supplier-confidential
signals in the text — cost prices, vendor codes, margins — are masked before
anything is returned, and store-staff identifiers are stripped from both the
question and the documents before either is stored.

The documents travel with the request, so nothing has to be uploaded anywhere
in advance and nothing is retained between calls. When a request carries no
documents, a small built-in sample corpus is searched instead and the answer
says so.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Sending a request

The question goes in `input`; the documents and the search tuning go in
`input_context`. Keep the documents there rather than pasting them into the
question: the platform rewrites personal-data shapes out of `input` at every
processing step, and its name heuristic treats title-case product and store
names as personal names, so document text sent that way arrives masked.

```json
{
  "input": "What are the SOP steps for store opening procedures?",
  "input_context": {
    "documents": [
      {"doc_id": "SOP-001", "chunk": "Store opening procedure: unlock the front door at 07:45, switch on the lighting and verify the point of sale terminal is online.", "category": "operations"},
      {"doc_id": "SOP-014", "chunk": "Store closing procedure: count the till, lock the stockroom and set the alarm before leaving.", "category": "operations"}
    ],
    "top_k": 5,
    "min_score": 0.2
  }
}
```

Document ids and categories are restricted to short inert codes; `top_k` and
`min_score` must be finite numbers in range; document text is screened for
instruction-like content and credential shapes and stripped of store-staff
identifiers. A refused request names the field that failed and never repeats
the value. `docs/02_design.md` carries the full contract.

## Project Structure

```
src/          agent implementation (nodes, graphs, services, schemas)
tests/        unit, boundary and integration tests
config/       agent manifest and runtime parameters
deploy/       local deployment recipe and a smoke payload
docs/         design and operational documentation
```

`docs/` holds the design (`02_design.md`) and the test specification
(`03_test_spec.md`).

## Customising

1. Adjust `config/config.yaml` for your own defaults — the number of documents
   returned and the relevance floor both change the released answer.
2. Replace the term-overlap ranking in `src/nodes/leann_search_node.py` with a
   call to your own on-device index; the request contract, the privacy filter
   and the output boundary do not depend on how the ranking is produced.
3. Extend the confidential-signal patterns in `src/nodes/privacy_filter_node.py`
   and the accepted fields in `src/services/caller_contract.py` for your own
   documents.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
