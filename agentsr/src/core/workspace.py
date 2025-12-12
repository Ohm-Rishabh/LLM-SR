"""
Workspace Manager - Manages isolated workspaces for workflow execution.

This module provides the WorkspaceManager class which creates and manages
isolated workspace directories for each workflow run. Each workspace provides
organized subdirectories for inputs, outputs, logs, and scratch files.
"""

from pathlib import Path
import datetime
import secrets
import shutil
import logging
import json
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages isolated workspace directories for workflow runs.

    Each workspace provides:
    - Isolated file system for a workflow run
    - Organized directories (input/, output/, logs/, scratch/)
    - Path resolution utilities
    - Cleanup mechanisms

    Example:
        >>> manager = WorkspaceManager(base_dir="/path/to/workspaces")
        >>> workspace = manager.create_workspace()
        >>> input_file = manager.copy_input_file("data.csv")
        >>> output_path = manager.get_path("output", "results.json")
    """

    def __init__(self, base_dir: Path, run_id: Optional[str] = None):
        """
        Initialize workspace manager.

        Args:
            base_dir: Base directory for all workspaces (e.g., ROOT_DIR/workspaces)
            run_id: Unique identifier for this run (auto-generated if None)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID: timestamp + random suffix
        if run_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = secrets.token_hex(3)
            run_id = f"run_{timestamp}_{random_suffix}"

        self.run_id = run_id
        self.workspace_root = self.base_dir / run_id

        # File metadata registry: maps relative paths to metadata
        self._file_metadata: Dict[str, Dict[str, Any]] = {}
        self._metadata_file = None  # Will be set after workspace creation

        logger.debug(f"Initialized WorkspaceManager with run_id={run_id}")

    def create_workspace(self) -> Path:
        """
        Create workspace directory structure.

        Creates the following subdirectories:
        - input/: For input data files
        - output/: For tool outputs and results
        - logs/: For tool execution logs
        - scratch/: For temporary files

        Returns:
            Path to the workspace root directory
        """
        logger.info(f"Creating workspace: {self.workspace_root}")

        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        subdirs = ["input", "output", "logs", "scratch"]
        for subdir in subdirs:
            subdir_path = self.workspace_root / subdir
            subdir_path.mkdir(exist_ok=True)
            logger.debug(f"Created subdirectory: {subdir_path}")

        # Set metadata file path
        self._metadata_file = self.workspace_root / ".file_metadata.json"

        # Load existing metadata if available (for resumed workspaces)
        self._load_metadata()

        logger.info(f"Workspace created successfully: {self.workspace_root}")
        return self.workspace_root

    def get_path(self, *parts: str) -> Path:
        """
        Get absolute path within workspace.

        Args:
            *parts: Path components relative to workspace root

        Returns:
            Absolute path within the workspace

        Example:
            >>> manager.get_path("output", "plot.png")
            Path("/path/to/workspace/output/plot.png")
        """
        return self.workspace_root.joinpath(*parts)

    def get_relative_path(self, absolute_path: Path) -> Optional[Path]:
        """
        Convert absolute path to workspace-relative path.

        Args:
            absolute_path: Absolute path to convert

        Returns:
            Path relative to workspace root, or None if path is outside workspace

        Example:
            >>> manager.get_relative_path("/workspace/output/file.txt")
            Path("output/file.txt")
        """
        try:
            abs_path = Path(absolute_path).resolve()
            return abs_path.relative_to(self.workspace_root.resolve())
        except ValueError:
            logger.warning(f"Path {absolute_path} is outside workspace")
            return None

    def copy_input_file(
        self,
        src: Path,
        dst_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Path:
        """
        Copy external file into workspace input directory.

        Args:
            src: Source file path
            dst_name: Destination filename (uses source filename if None)
            description: Optional description of the file

        Returns:
            Path to the copied file in workspace

        Raises:
            FileNotFoundError: If source file doesn't exist

        Example:
            >>> manager.copy_input_file(
            ...     "/data/dataset.csv",
            ...     "data.csv",
            ...     description="Original dataset with features and target"
            ... )
            Path("/workspace/input/data.csv")
        """
        src_path = Path(src)
        if not src_path.exists():
            logger.error(f"Source file not found: {src}")
            raise FileNotFoundError(f"Source file not found: {src}")

        if dst_name is None:
            dst_name = src_path.name

        dst_path = self.get_path("input", dst_name)

        logger.info(f"Copying input file: {src_path} -> {dst_path}")
        shutil.copy2(src_path, dst_path)

        # Register file with metadata
        if description is None:
            description = f"Input data file"

        self.register_file(
            dst_path,
            description=description,
            file_type="data",
            created_by="workflow",
            source_path=str(src_path)
        )

        return dst_path

    def list_files(
        self,
        subdir: Optional[str] = None,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in workspace or subdirectory.

        Args:
            subdir: Subdirectory to search (None for workspace root)
            pattern: Glob pattern for matching files (default: "*")
            recursive: If True, search recursively (default: False)

        Returns:
            Sorted list of matching file paths

        Example:
            >>> manager.list_files("output", "*.json")
            [Path("output/results.json"), Path("output/metrics.json")]
        """
        search_dir = self.workspace_root if subdir is None else self.get_path(subdir)

        if not search_dir.exists():
            logger.warning(f"Directory does not exist: {search_dir}")
            return []

        if recursive:
            matches = sorted(search_dir.rglob(pattern))
        else:
            matches = sorted(search_dir.glob(pattern))

        # Filter to only files (not directories)
        file_matches = [p for p in matches if p.is_file()]

        logger.debug(f"Found {len(file_matches)} files matching '{pattern}' in {search_dir}")
        return file_matches

    def read_file(self, *path_parts: str, encoding: str = 'utf-8') -> str:
        """
        Read text file from workspace.

        Args:
            *path_parts: Path components relative to workspace root
            encoding: Text encoding (default: utf-8)

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist

        Example:
            >>> content = manager.read_file("output", "results.json")
        """
        file_path = self.get_path(*path_parts)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.debug(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()

    def write_file(
        self,
        content: str,
        *path_parts: str,
        encoding: str = 'utf-8'
    ) -> Path:
        """
        Write text file to workspace.

        Args:
            content: Content to write
            *path_parts: Path components relative to workspace root
            encoding: Text encoding (default: utf-8)

        Returns:
            Path to the written file

        Example:
            >>> manager.write_file("results", "output", "summary.txt")
            Path("/workspace/output/summary.txt")
        """
        file_path = self.get_path(*path_parts)

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Writing file: {file_path}")
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)

        return file_path

    def get_info(self) -> dict:
        """
        Get workspace information.

        Returns:
            Dictionary with workspace metadata

        Example:
            >>> info = manager.get_info()
            >>> print(info['run_id'])
            'run_20231210_143022_abc123'
        """
        info = {
            "run_id": self.run_id,
            "workspace_root": str(self.workspace_root),
            "base_dir": str(self.base_dir),
            "exists": self.workspace_root.exists(),
        }

        if self.workspace_root.exists():
            # Count files in subdirectories
            for subdir in ["input", "output", "logs", "scratch"]:
                subdir_path = self.workspace_root / subdir
                if subdir_path.exists():
                    file_count = len(list(subdir_path.glob("*")))
                    info[f"{subdir}_files"] = file_count

        return info

    @classmethod
    def cleanup_old_workspaces(
        cls,
        base_dir: Path,
        keep_last_n: int = 10
    ) -> int:
        """
        Remove old workspace directories, keeping the N most recent.

        Args:
            base_dir: Base directory containing workspaces
            keep_last_n: Number of most recent workspaces to keep (default: 10)

        Returns:
            Number of workspaces removed

        Example:
            >>> removed = WorkspaceManager.cleanup_old_workspaces(
            ...     Path("/workspaces"),
            ...     keep_last_n=5
            ... )
            >>> print(f"Removed {removed} old workspaces")
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"Base directory does not exist: {base_path}")
            return 0

        # Get all workspace directories sorted by modification time (newest first)
        workspaces = sorted(
            [d for d in base_path.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        logger.info(f"Found {len(workspaces)} workspaces, keeping {keep_last_n} most recent")

        # Remove old workspaces beyond keep_last_n
        removed_count = 0
        for workspace in workspaces[keep_last_n:]:
            try:
                logger.info(f"Removing old workspace: {workspace}")
                shutil.rmtree(workspace)
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove workspace {workspace}: {e}")

        logger.info(f"Cleanup completed: removed {removed_count} workspace(s)")
        return removed_count

    # --- File Metadata Management ---

    def _load_metadata(self) -> None:
        """Load file metadata from the metadata file."""
        if self._metadata_file and self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    self._file_metadata = json.load(f)
                logger.debug(f"Loaded metadata for {len(self._file_metadata)} files")
            except Exception as e:
                logger.warning(f"Failed to load metadata file: {e}")
                self._file_metadata = {}
        else:
            self._file_metadata = {}

    def _save_metadata(self) -> None:
        """Save file metadata to the metadata file."""
        if self._metadata_file:
            try:
                with open(self._metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self._file_metadata, f, indent=2)
                logger.debug(f"Saved metadata for {len(self._file_metadata)} files")
            except Exception as e:
                logger.error(f"Failed to save metadata file: {e}")

    def register_file(
        self,
        file_path: Path,
        description: str,
        file_type: Optional[str] = None,
        created_by: Optional[str] = None,
        **extra_metadata
    ) -> None:
        """
        Register a file with metadata (description and other info).

        Args:
            file_path: Absolute or relative path to the file
            description: Human-readable description of the file's content/purpose
            file_type: Type of file (e.g., "data", "plot", "result", "log")
            created_by: Tool or node that created the file
            **extra_metadata: Additional metadata fields

        Example:
            >>> manager.register_file(
            ...     "output/correlation.png",
            ...     description="Correlation heatmap of all features",
            ...     file_type="plot",
            ...     created_by="python_interpreter"
            ... )
        """
        # Convert to relative path if absolute
        rel_path = self.get_relative_path(file_path)
        if rel_path is None:
            # If outside workspace, try using as-is
            rel_path = Path(file_path).relative_to(self.workspace_root) if Path(file_path).is_absolute() else Path(file_path)

        rel_path_str = str(rel_path)

        # Build metadata entry
        metadata = {
            "description": description,
            "file_type": file_type or "unknown",
            "created_by": created_by or "unknown",
            "registered_at": datetime.datetime.now().isoformat(),
        }

        # Add extra metadata
        metadata.update(extra_metadata)

        # Add file size if file exists
        full_path = self.get_path(rel_path_str)
        if full_path.exists():
            metadata["size_bytes"] = full_path.stat().st_size
            metadata["modified_at"] = datetime.datetime.fromtimestamp(
                full_path.stat().st_mtime
            ).isoformat()

        self._file_metadata[rel_path_str] = metadata
        self._save_metadata()

        logger.info(f"Registered file: {rel_path_str} - {description}")

    def get_file_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific file.

        Args:
            file_path: Absolute or relative path to the file

        Returns:
            Dictionary with file metadata, or None if not registered

        Example:
            >>> metadata = manager.get_file_metadata("output/plot.png")
            >>> print(metadata['description'])
            'Correlation heatmap'
        """
        rel_path = self.get_relative_path(file_path)
        if rel_path is None:
            rel_path = Path(file_path)

        return self._file_metadata.get(str(rel_path))

    def list_files_with_metadata(
        self,
        subdir: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all files with their metadata.

        Args:
            subdir: Filter by subdirectory (e.g., "output", "input")
            file_type: Filter by file type (e.g., "plot", "data")

        Returns:
            List of dictionaries with file info and metadata

        Example:
            >>> files = manager.list_files_with_metadata(subdir="output", file_type="plot")
            >>> for file in files:
            ...     print(f"{file['path']}: {file['description']}")
        """
        result = []

        for rel_path, metadata in self._file_metadata.items():
            # Filter by subdirectory
            if subdir and not rel_path.startswith(f"{subdir}/"):
                continue

            # Filter by file type
            if file_type and metadata.get("file_type") != file_type:
                continue

            # Build file info
            full_path = self.get_path(rel_path)
            file_info = {
                "path": rel_path,
                "full_path": str(full_path),
                "exists": full_path.exists(),
                **metadata
            }

            result.append(file_info)

        return result

    def get_files_summary(self) -> str:
        """
        Get a human-readable summary of all files in the workspace.

        This method is designed for LLM consumption - it provides a clear,
        structured summary of all files with their descriptions.

        Returns:
            Formatted string describing all workspace files

        Example:
            >>> summary = manager.get_files_summary()
            >>> print(summary)
            Workspace Files:

            input/
              - data.csv: Original dataset with features and target

            output/
              - correlation.png: Correlation heatmap of all features
              - analysis_results.json: Statistical analysis results
        """
        if not self._file_metadata:
            return "Workspace: No files registered yet."

        # Group files by subdirectory
        files_by_dir: Dict[str, List[tuple]] = {}

        for rel_path, metadata in self._file_metadata.items():
            parts = Path(rel_path).parts
            if len(parts) > 1:
                subdir = parts[0]
                filename = "/".join(parts[1:])
            else:
                subdir = "root"
                filename = rel_path

            if subdir not in files_by_dir:
                files_by_dir[subdir] = []

            files_by_dir[subdir].append((filename, metadata))

        # Build summary
        lines = ["Workspace Files:", ""]

        for subdir in sorted(files_by_dir.keys()):
            lines.append(f"{subdir}/")
            for filename, metadata in sorted(files_by_dir[subdir]):
                desc = metadata.get("description", "No description")
                file_type = metadata.get("file_type", "unknown")
                lines.append(f"  - {filename} ({file_type}): {desc}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"WorkspaceManager(run_id={self.run_id!r}, workspace_root={self.workspace_root})"

    def __str__(self) -> str:
        return f"Workspace[{self.run_id}]: {self.workspace_root}"
