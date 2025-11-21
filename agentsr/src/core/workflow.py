from __future__ import annotations
from typing import Any, Dict, List, Optional

from core.node import Node


class Workflow:
    """
    A simple directed graph of nodes.

    - Nodes are instances of Node (or subclasses).
    - Edges define which node(s) follow after each node.
    - Execution is synchronous and single-threaded.

    Basic semantics of `run`:
      - Start from `start_node`.
      - For each node:
          state = node.run(state)
          visited.append(node.name)

          Let `succs = outgoing edges from this node`.

          * If len(succs) == 0:
                stop (this is a terminal node).
          * If len(succs) == 1:
                go to that successor.
          * If len(succs) > 1:
                expect the node to write `state["_next_node"]`
                indicating which successor to take.
                (If not provided, raise an error.)
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, List[str]] = {}
        self._start: Optional[str] = None

    def build_workflow(self) -> None:
        """
        Example method to build a sample workflow.
        This can be customized or replaced as needed.
        """
        pass

    # --- graph construction API ---

    def add_node(self, node: Node, is_start: bool = False) -> None:
        """
        Add a node to the workflow.

        Args:
            node: Node instance (must have unique .name).
            is_start: If True, set this node as the start node.
                      If start is not set yet, the first added node
                      becomes the start node by default.
        """
        if node.name in self._nodes:
            raise ValueError(f"Node with name {node.name!r} already exists.")

        self._nodes[node.name] = node
        self._edges.setdefault(node.name, [])

        if is_start or self._start is None:
            self._start = node.name

    def add_edge(self, src: str, dst: str) -> None:
        """
        Add a directed edge from node `src` to node `dst`.
        Both nodes must already be in the workflow.
        """
        if src not in self._nodes:
            raise KeyError(f"Source node {src!r} not found in workflow.")
        if dst not in self._nodes:
            raise KeyError(f"Destination node {dst!r} not found in workflow.")

        self._edges.setdefault(src, [])
        self._edges[src].append(dst)

    def set_start(self, node_name: str) -> None:
        """
        Explicitly set the start node by name.
        """
        if node_name not in self._nodes:
            raise KeyError(f"Start node {node_name!r} not found in workflow.")
        self._start = node_name

    # --- inspection helpers ---

    @property
    def nodes(self) -> Dict[str, Node]:
        return dict(self._nodes)

    @property
    def edges(self) -> Dict[str, List[str]]:
        # return a shallow copy to avoid accidental external mutation
        return {k: v[:] for k, v in self._edges.items()}

    @property
    def start_node(self) -> Optional[str]:
        return self._start

    # --- execution ---

    def run(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        max_steps: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run the workflow from the start node until termination.

        Args:
            initial_state: Optional initial state dict. If None, use {}.
            max_steps: Safety cap to avoid infinite loops in graphs with cycles.

        Returns:
            Final state dictionary after the workflow terminates.

        Raises:
            RuntimeError: if no start node is set, or if:
                - the step limit is exceeded (likely a cycle),
                - multiple outgoing edges but no `_next_node` is chosen,
                - `_next_node` points to an invalid or non-successor node.
        """
        if self._start is None:
            raise RuntimeError("No start node set for this workflow.")

        state: Dict[str, Any] = {} if initial_state is None else dict(initial_state)
        visited: List[str] = []

        current = self._start
        steps = 0

        while current is not None:
            if steps >= max_steps:
                raise RuntimeError(
                    f"Maximum step limit ({max_steps}) exceeded. "
                    "Possible infinite loop in workflow."
                )

            node = self._nodes[current]
            state = node.run(state)
            visited.append(current)
            steps += 1

            successors = self._edges.get(current, [])

            # No outgoing edges -> terminal node
            if not successors:
                current = None
                continue

            # Single successor -> go there
            if len(successors) == 1:
                current = successors[0]
                continue

            # Multiple successors -> expect node to choose via `_next_node`
            next_name = state.pop("_next_node", None)
            if next_name is None:
                raise RuntimeError(
                    f"Node {current!r} has multiple successors {successors}, "
                    "but `_next_node` was not set in state."
                )
            if next_name not in successors:
                raise RuntimeError(
                    f"Invalid `_next_node` {next_name!r} chosen by node {current!r}. "
                    f"Allowed successors are: {successors}."
                )

            current = next_name

        # Optionally store the execution trace
        state["_visited_nodes"] = visited
        return state
