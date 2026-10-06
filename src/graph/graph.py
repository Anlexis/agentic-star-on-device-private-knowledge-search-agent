"""AgentCore Platform v1.0"""

# RET-C2-178 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Retail On-Device Private KB Search Agent.
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, bounded by max_retry)
#                                             v
#                                          pre_process
#
#   `main` slot is a GraphNode subclass (KBSearchGraphNode) that
#   delegates the full KB search domain workflow to
#   DomainWorkflowGraph (inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- validated caller contract across the boundary
#
# Rules enforced:
#   - RetailPrivateKBSearchAgent inherits AgentBaseGraph (L1 Base — direct inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - KBSearchGraphNode assigned to self._nodes["main"]
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - No Level 0 agenticstar imports

from pathlib import Path
from typing import Any, ClassVar, Dict

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.utils.config_loader import load_agent_config
from src.graph.context_bridge import set_caller_contract
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

# Repo root: src/graph/graph.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fallbacks mirror config/config.yaml so the search tuning is never empty even
# where the config file is unreadable in an exotic deployment layout.
_FALLBACK_SEARCH: Dict[str, Any] = {"top_k": 5, "min_score": 0.2}


def runtime_config() -> Dict[str, Any]:
    """Load config/config.yaml — the live runtime parameters.

    The registry loads this file and passes it to the graph constructor; the
    standalone HTTP entry point does the same, so `max_retry` and the search
    tuning are live in both deployments rather than declared and ignored.

    Reading the static manifest (config/agent.yaml) here instead would return
    nothing: the manifest carries identity and compile-time requirements only,
    and a reader pointed at it degrades silently to defaults.
    """
    loaded = load_agent_config(_REPO_ROOT)
    return dict(loaded) if isinstance(loaded, dict) else {}


class KBSearchGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of RetailPrivateKBSearchAgent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    — instantiate DomainWorkflowGraph with the forwarded
                          runtime config (_parent_config())
      extract_input()   — hand the validated query to the inner graph and
                          stash the validated caller contract on the bridge
      merge_output()    — map sub_result fields into outer state delta (changed keys only)
      error_strategy    — "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (default — fail fast).
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the live search defaults to the inner graph.

        Returns the search block under config["configurable"] — never an empty
        dict. The inner graph republishes it into inner state
        (DomainWorkflowGraph._extra_initial_state()) so the search step reads
        live values: node execute() methods take no config parameter, so state
        seeding is the only route config can travel.
        """
        search = runtime_config().get("search")
        if not isinstance(search, dict) or not search:
            search = dict(_FALLBACK_SEARCH)
        return {"configurable": {"search": search}}

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time. The inner graph receives the
        runtime-derived config through its constructor; its domain nodes still
        take no constructor arguments and read their tuning from seeded state.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the query string, and bridge the validated caller contract.

        The framework hands only a string to the inner graph, so the structured
        part of the request travels on the bridge instead — set here, one step
        before the inner invoke, and read by the inner graph's initial-state
        hook. Only the contract the pre_process node already validated crosses.
        """
        set_caller_contract(from_json(state.get("caller_contract"), {}) or {})
        return str(state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_response", "status", ...
          This merge_output() reads -> sub_result.get("formatted_response"),
                                       sub_result.get("status")

        formatted_response is mapped to "result" as well, because the
        post_process output gate reads state["result"] — without that mapping
        the surfaced output would always be empty.

        A non-success inner status never reaches here (error_strategy is
        "propagate", so the inner error is re-raised first), but the guard is
        kept so a future strategy change cannot start publishing an un-gated
        response through this path.
        """
        status = sub_result.get("status")
        response = sub_result.get("formatted_response")
        if status != AgentStatus.SUCCESS.value:
            return {"formatted_response": None, "result": None, "status": status}
        return {"formatted_response": response, "result": response, "status": status}


class RetailPrivateKBSearchAgent(AgentBaseGraph):
    """Outer graph for RET-C2-178 (Cat 2).

    Inherits AgentBaseGraph directly (L1 Base). Domain logic is fully
    encapsulated in KBSearchGraphNode (main slot), which delegates to
    DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed — identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode (caller contract + personal-data strip)
      - main:        KBSearchGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    get_output() is NOT overridden either: the output gate is the single place
    the released text is checked, and it withholds by clearing every
    output-bearing field and returning a truthy notice, so the framework's
    envelope never has a leaked value to fall back to.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "RetailPrivateKBSearchAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = KBSearchGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Back-compat alias — config/agent.yaml declares class: "RetailPrivateKBSearchAgent".
# server.py imports RetailPrivateKBSearchAgent directly; Graph alias provided
# for scaffold back-compatibility.
Graph = RetailPrivateKBSearchAgent
