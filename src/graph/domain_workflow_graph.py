"""AgentCore Platform v1.0"""

# RET-C2-178 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full retail on-device private KB search domain workflow:
#
#   START -> query_normalize -> kb_search -> privacy_filter -> response_format -> END
#
# Called by KBSearchGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology — no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with KBSearchGraphNode.merge_output()
#   - No agenticstar imports
#   - Not placed under src/subagents/

from typing import Any, Dict

from langgraph.graph import END, START

from framework.errors import ConfigError
from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_contract
from src.nodes.leann_search_node import KBSearchNode
from src.nodes.privacy_filter_node import PrivacyFilterNode
from src.nodes.query_normalize_node import QueryNormalizeNode
from src.nodes.response_format_node import ResponseFormatNode
from src.schemas.state import State, to_json
from src.services.caller_contract import MAX_TOP_K


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for RET-C2-178.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by KBSearchGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          -> query_normalize   (QueryNormalizeNode)   — expand acronyms, normalize whitespace
          -> kb_search         (KBSearchNode)         — rank the knowledge-base documents
          -> privacy_filter    (PrivacyFilterNode)    — mask supplier-confidential signals
          -> response_format   (ResponseFormatNode)   — render Markdown KB response
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ret_c2_178_private_kb_search_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Reject a search block that would silently change the answer.

        The tuning arrives from config/config.yaml through the parent graph.
        Both values are read by the search step and shape the released
        answer, so a malformed one is refused at compile time rather than
        being coerced to a default the operator never declared.
        """
        search = (self.config.get("configurable") or {}).get("search")
        if search is None:
            return
        if not isinstance(search, dict):
            raise ConfigError(f"[{self.__class__.__name__}] 'search' must be a mapping")

        top_k = search.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K):
            raise ConfigError(
                f"[{self.__class__.__name__}] 'search.top_k' must be an integer between 1 and {MAX_TOP_K}"
            )

        min_score = search.get("min_score")
        if min_score is not None:
            if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
                raise ConfigError(f"[{self.__class__.__name__}] 'search.min_score' must be a number")
            if not (0.0 <= float(min_score) <= 1.0):
                raise ConfigError(f"[{self.__class__.__name__}] 'search.min_score' must be between 0 and 1")

    # -- Initial state ---------------------------------------------------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the bridged caller contract and the live tuning into inner state.

        Both have to arrive this way. The framework hands the inner graph only a
        string, so the validated documents travel on the context bridge; and
        node execute() methods take no config argument, so the tuning travels
        as state. Structured values are stored as JSON strings, because
        checkpoint serialization does not carry bare containers safely.
        """
        search = (self.config.get("configurable") or {}).get("search") or {}
        return {
            "caller_contract": to_json(get_caller_contract()),
            "search_config": to_json(dict(search)),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 4 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments — FunctionNode
        subclasses take no __init__; tuning arrives through seeded state.
        Every key registered here is referenced in add_edges().
        """
        self._nodes["query_normalize"] = QueryNormalizeNode()
        self._nodes["kb_search"] = KBSearchNode()
        self._nodes["privacy_filter"] = PrivacyFilterNode()
        self._nodes["response_format"] = ResponseFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear KB search domain topology.

        Each step passes its partial-dict output into the shared State.
        The topology is intentionally linear — no conditional branching
        between domain nodes. route() is implemented as required by the ABC
        but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "query_normalize")
        self._sg.add_edge("query_normalize", "kb_search")
        self._sg.add_edge("kb_search", "privacy_filter")
        self._sg.add_edge("privacy_filter", "response_format")
        self._sg.add_edge("response_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: State) -> str:
        """Conditional routing — required by the BaseGraph contract.

        add_conditional_edges() is not used for this linear topology, so this is
        never called at runtime. It is annotated with this graph's own State
        because LangGraph reads a path callable's annotation as its input schema
        and projects away every field the annotation does not declare — an
        annotation naming the base state would make the routing fields
        permanently absent if this method were ever wired.
        Returns END on error so an unexpected call does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "response_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by KBSearchGraphNode.merge_output() in graph.py
        as the `sub_result` argument. Both methods are designed together to
        guarantee field-name consistency:

            Inner get_output()   emits: "formatted_response", "status", ...
            Outer merge_output() reads: sub_result.get("formatted_response"),
                                        sub_result.get("status")

        On a non-success status the response is withheld here as well as at
        the outer boundary: a step that failed leaves the state it should have
        written empty, so anything rendered downstream describes a search that
        never ran.
        """
        status = state.get("status")
        return {
            "formatted_response": state.get("formatted_response") if status == AgentStatus.SUCCESS.value else None,
            "status": status,
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
