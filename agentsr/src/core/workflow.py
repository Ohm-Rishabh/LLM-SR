from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
import os

from core.node import Node
from core.workspace import WorkspaceManager
from core.consts import ROOT_DIR

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        workspace_base_dir: Optional[Path] = None,
        enable_workspace: bool = True
    ) -> None:
        """
        Initialize the workflow.

        Args:
            workspace_base_dir: Base directory for workspaces (default: ROOT_DIR/workspaces)
            enable_workspace: Enable workspace support (default: True)
        """
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, List[str]] = {}
        self._start: Optional[str] = None

        # Workspace configuration
        self.enable_workspace = enable_workspace
        if workspace_base_dir is None:
            workspace_base_dir = Path(ROOT_DIR) / "workspaces"
            os.makedirs(workspace_base_dir, exist_ok=True)
        self.workspace_base_dir = Path(workspace_base_dir)
        self.workspace_manager: Optional[WorkspaceManager] = None

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
            logger.error(f"Attempted to add duplicate node: {node.name}")
            raise ValueError(f"Node with name {node.name!r} already exists.")

        self._nodes[node.name] = node
        self._edges.setdefault(node.name, [])

        if is_start or self._start is None:
            self._start = node.name
            logger.info(f"Added node '{node.name}' as start node")
        else:
            logger.info(f"Added node '{node.name}' to workflow")

    def add_edge(self, src: str | Node, dst: str | Node) -> None:
        """
        Add a directed edge from node `src` to node `dst`.
        Both nodes must already be in the workflow.
        """
        if isinstance(src, Node):
            src = src.name
        if isinstance(dst, Node):
            dst = dst.name
        if src not in self._nodes:
            logger.error(f"Source node '{src}' not found in workflow")
            raise KeyError(f"Source node {src!r} not found in workflow.")
        if dst not in self._nodes:
            logger.error(f"Destination node '{dst}' not found in workflow")
            raise KeyError(f"Destination node {dst!r} not found in workflow.")

        self._edges.setdefault(src, [])
        self._edges[src].append(dst)
        logger.debug(f"Added edge: {src} -> {dst}")

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
        cleanup_old_workspaces: bool = False,
        keep_last_n_workspaces: int = 20,
    ) -> Dict[str, Any]:
        """
        Run the workflow from the start node until termination.

        Args:
            initial_state: Optional initial state dict. If None, use {}.
            max_steps: Safety cap to avoid infinite loops in graphs with cycles.
            cleanup_old_workspaces: If True, clean up old workspaces after run.
            keep_last_n_workspaces: Number of recent workspaces to keep during cleanup.

        Returns:
            Final state dictionary after the workflow terminates.

        Raises:
            RuntimeError: if no start node is set, or if:
                - the step limit is exceeded (likely a cycle),
                - multiple outgoing edges but no `_next_node` is chosen,
                - `_next_node` points to an invalid or non-successor node.
        """
        if self._start is None:
            logger.error("Attempted to run workflow with no start node")
            raise RuntimeError("No start node set for this workflow.")

        logger.info(f"Starting workflow execution from node '{self._start}'")
        state: Dict[str, Any] = {} if initial_state is None else dict(initial_state)

        # Initialize workspace if enabled
        if self.enable_workspace:
            self.workspace_manager = WorkspaceManager(self.workspace_base_dir)
            workspace_path = self.workspace_manager.create_workspace()
            logger.info(f"Created workspace: {workspace_path}")

            # Inject workspace manager into all nodes
            for node in self._nodes.values():
                node.set_workspace_manager(self.workspace_manager)
            logger.debug(f"Injected workspace manager into {len(self._nodes)} node(s)")

            # Add workspace information to state
            state["_workspace_root"] = str(workspace_path)
            state["_workspace_manager"] = self.workspace_manager

            # Copy input data file to workspace if provided
            if "input_file" in state:
                original_path = Path(state["input_file"])
                file_name = original_path.name
                if original_path.exists():
                    try:
                        workspace_data_path = self.workspace_manager.copy_input_file(
                            original_path,
                            file_name
                        )
                        # Update state to point to workspace copy
                        state["input_file"] = str(workspace_data_path)
                        logger.info(f"Copied input file to workspace: {workspace_data_path}")
                    except Exception as e:
                        logger.warning(f"Failed to copy input file to workspace: {e}")
                        # Keep original path in state
                else:
                    logger.warning(f"Input file not found: {original_path}")

        visited: List[str] = []
        current = self._start
        steps = 0

        while current is not None:
            if steps >= max_steps:
                logger.error(f"Workflow exceeded maximum step limit ({max_steps})")
                raise RuntimeError(
                    f"Maximum step limit ({max_steps}) exceeded. "
                    "Possible infinite loop in workflow."
                )

            logger.info(f"Executing node '{current}' (step {steps + 1})")
            node = self._nodes[current]
            state = node.run(state)
            visited.append(current)
            steps += 1

            successors = self._edges.get(current, [])

            # No outgoing edges -> terminal node
            if not successors:
                logger.info(f"Node '{current}' is terminal, ending workflow")
                current = None
                continue

            # Single successor -> go there
            if len(successors) == 1:
                logger.debug(f"Node '{current}' has single successor '{successors[0]}'")
                current = successors[0]
                continue

            # Multiple successors -> expect node to choose via `_next_node`
            next_name = state.pop("_next_node", None)
            if next_name is None:
                logger.error(f"Node '{current}' has multiple successors but _next_node not set")
                raise RuntimeError(
                    f"Node {current!r} has multiple successors {successors}, "
                    "but `_next_node` was not set in state."
                )
            if next_name not in successors:
                logger.error(f"Invalid _next_node '{next_name}' chosen by node '{current}'")
                raise RuntimeError(
                    f"Invalid `_next_node` {next_name!r} chosen by node {current!r}. "
                    f"Allowed successors are: {successors}."
                )

            logger.info(f"Node '{current}' chose successor '{next_name}' from {successors}")
            current = next_name

        # Optionally store the execution trace
        state["_visited_nodes"] = visited

        # Store workspace path in state for reference
        if self.enable_workspace and self.workspace_manager:
            state["_workspace_path"] = str(self.workspace_manager.workspace_root)
            logger.info(f"Workspace: {self.workspace_manager.workspace_root}")

        # Optional cleanup of old workspaces
        if cleanup_old_workspaces and self.enable_workspace:
            try:
                removed_count = WorkspaceManager.cleanup_old_workspaces(
                    self.workspace_base_dir,
                    keep_last_n=keep_last_n_workspaces
                )
                logger.info(f"Cleaned up {removed_count} old workspace(s)")
            except Exception as e:
                logger.warning(f"Failed to cleanup old workspaces: {e}")

        logger.info(f"Workflow completed successfully. Visited nodes: {visited}")
        return state
