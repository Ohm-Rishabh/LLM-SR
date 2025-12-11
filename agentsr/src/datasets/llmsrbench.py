#!/usr/bin/env python3
"""
LLM-SRBench Dataset Manager

This module provides utilities to extract datasets from HDF5 files and
load them as CSV files with metadata.

Dataset Structure:
- HDF5 file contains two main groups: 'lsr_synth' and 'lsr_transform'
- Each dataset (e.g., 'I.10.7_1_0') has train/test/val splits
- Metadata is stored in parquet files with dataset descriptions

Usage:
    # Extract all datasets from HDF5 to CSV (run once)
    python llmsrbench.py

    # Load a specific dataset in your code
    from datasets.llmsrbench import LLMSRBenchDataset
    dataset = LLMSRBenchDataset()
    csv_path, metadata = dataset.get_dataset("I.10.7_1_0")
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HDF5_PATH = Path("/data/agent-discovery/llmsrbench/lsr_bench_data.hdf5")
DEFAULT_METADATA_PATH = Path("/data/agent-discovery/llmsrbench")
DEFAULT_OUTPUT_DIR = Path("/data/agent-discovery/llmsrbench/csv")


class LLMSRBenchDataset:
    """
    Manager for LLM-SRBench datasets.

    This class handles:
    1. Extracting datasets from HDF5 to CSV files
    2. Loading metadata from parquet files
    3. Providing dataset access by name with descriptions
    """

    def __init__(
        self,
        hdf5_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the dataset manager.

        Args:
            hdf5_path: Path to the HDF5 file containing datasets
            metadata_path: Path to directory containing metadata parquet files
            output_dir: Directory where CSV files will be saved/loaded
        """
        self.hdf5_path = hdf5_path or DEFAULT_HDF5_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for metadata
        self._metadata_cache: Optional[pd.DataFrame] = None

    def _load_metadata(self) -> pd.DataFrame:
        """
        Load metadata from all parquet files.

        Returns:
            DataFrame with columns: name, symbols, symbol_descs, symbol_properties, expression
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        # Load metadata from parquet files
        metadata_dfs = []

        parquet_files = {
            'lsr_synth_bio_pop_growth': 'data/lsr_synth_bio_pop_growth-00000-of-00001.parquet',
            'lsr_synth_chem_react': 'data/lsr_synth_chem_react-00000-of-00001.parquet',
            'lsr_synth_matsci': 'data/lsr_synth_matsci-00000-of-00001.parquet',
            'lsr_synth_phys_osc': 'data/lsr_synth_phys_osc-00000-of-00001.parquet',
            'lsr_transform': 'data/lsr_transform-00000-of-00001.parquet'
        }

        for name, filepath in parquet_files.items():
            full_path = self.metadata_path / filepath
            if full_path.exists():
                df = pd.read_parquet(full_path)
                df['dataset_group'] = name  # Add group identifier
                metadata_dfs.append(df)
                logger.info(f"Loaded metadata from {name}: {len(df)} datasets")
            else:
                logger.warning(f"Metadata file not found: {full_path}")

        if not metadata_dfs:
            raise FileNotFoundError(f"No metadata files found in {self.metadata_path}")

        self._metadata_cache = pd.concat(metadata_dfs, ignore_index=True)
        logger.info(f"Total datasets in metadata: {len(self._metadata_cache)}")

        return self._metadata_cache

    def extract_all_datasets(self, overwrite: bool = False) -> Dict[str, int]:
        """
        Extract all datasets from HDF5 to CSV files.

        Structure:
        - lsr_transform/ -> lsr_transform/{dataset_name}_train.csv, {dataset_name}_test.csv
        - lsr_synth/bio_pop_growth/ -> {dataset_name}_train.csv, {dataset_name}_test.csv
        - lsr_synth/chem_react/ -> {dataset_name}_train.csv, {dataset_name}_test.csv
        - lsr_synth/matsci/ -> {dataset_name}_train.csv, {dataset_name}_test.csv
        - lsr_synth/phys_osc/ -> {dataset_name}_train.csv, {dataset_name}_test.csv

        Args:
            overwrite: If True, overwrite existing CSV files

        Returns:
            Dictionary with statistics: {full_path: num_rows}
        """
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Load metadata to get variable names
        metadata_df = self._load_metadata()

        stats = {}

        with h5py.File(self.hdf5_path, "r") as f:
            # Process lsr_transform group (flat structure)
            if 'lsr_transform' in f:
                transform_group = f['lsr_transform']
                output_subdir = self.output_dir / "lsr_transform"
                output_subdir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Processing lsr_transform ({len(transform_group)} datasets)")

                for dataset_name in transform_group.keys():
                    self._extract_dataset_splits(
                        f, 'lsr_transform', dataset_name, output_subdir, overwrite, stats, metadata_df
                    )

            # Process lsr_synth group (has subcategories)
            if 'lsr_synth' in f:
                synth_group = f['lsr_synth']
                subcategories = list(synth_group.keys())

                for subcat in subcategories:
                    output_subdir = self.output_dir / "lsr_synth" / subcat
                    output_subdir.mkdir(parents=True, exist_ok=True)

                    subcat_group = synth_group[subcat]
                    logger.info(f"Processing lsr_synth/{subcat} ({len(subcat_group)} datasets)")

                    for dataset_name in subcat_group.keys():
                        self._extract_dataset_splits(
                            f, f'lsr_synth/{subcat}', dataset_name, output_subdir, overwrite, stats, metadata_df
                        )

        logger.info(f"Extraction complete. Total files: {len(stats)}")
        return stats

    def _extract_dataset_splits(
        self,
        h5file,
        group_path: str,
        dataset_name: str,
        output_dir: Path,
        overwrite: bool,
        stats: Dict[str, int],
        metadata_df: pd.DataFrame
    ):
        """
        Extract train and test splits for a single dataset.

        Args:
            h5file: Open HDF5 file handle
            group_path: Path to the dataset group (e.g., 'lsr_transform' or 'lsr_synth/bio_pop_growth')
            dataset_name: Name of the dataset
            output_dir: Directory to save CSV files
            overwrite: Whether to overwrite existing files
            stats: Dictionary to update with statistics
            metadata_df: DataFrame containing metadata for all datasets
        """
        # Get variable names from metadata
        dataset_meta = metadata_df[metadata_df['name'] == dataset_name]

        if not dataset_meta.empty and 'symbols' in dataset_meta.columns:
            symbols = dataset_meta.iloc[0]['symbols']
            # Data columns are aligned with the symbols list
            columns = symbols
        else:
            columns = None

        for split in ['train', 'test']:
            csv_path = output_dir / f"{dataset_name}_{split}.csv"

            # Skip if file exists and overwrite is False
            if csv_path.exists() and not overwrite:
                logger.debug(f"Skipping {group_path}/{dataset_name}/{split} (already exists)")
                continue

            # Read split data
            data_key = f"{group_path}/{dataset_name}/{split}"
            if data_key not in h5file:
                logger.warning(f"Split not found: {data_key}")
                continue

            data = h5file[data_key][:]

            # Convert to DataFrame
            num_features = data.shape[1] - 1

            # Use actual variable names if available, otherwise use generic names
            if columns is not None and len(columns) == data.shape[1]:
                df = pd.DataFrame(data, columns=columns)
                logger.info(f"Extracted {group_path}/{dataset_name}/{split}: {len(df)} rows, columns: {list(columns)}")
            else:
                # Fallback to generic column names
                fallback_columns = [f"x{i}" for i in range(num_features)] + ["y"]
                df = pd.DataFrame(data, columns=fallback_columns)
                logger.info(f"Extracted {group_path}/{dataset_name}/{split}: {len(df)} rows, {num_features} features (no metadata)")

            # Save to CSV
            df.to_csv(csv_path, index=False)
            stats[str(csv_path)] = len(df)

    def get_dataset(self, name: str, split: str = "train") -> Tuple[Path, Dict[str, any]]:
        """
        Get dataset CSV path and metadata by name.

        Args:
            name: Dataset name (e.g., "I.10.7_1_0" or "BPG0")
            split: Dataset split ('train' or 'test'), default is 'train'

        Returns:
            Tuple of (csv_path, metadata_dict)

        Raises:
            FileNotFoundError: If dataset CSV or metadata not found
        """
        # Search for the dataset in all subdirectories
        csv_path = None
        search_paths = [
            self.output_dir / "lsr_transform" / f"{name}_{split}.csv",
            self.output_dir / "lsr_synth" / "bio_pop_growth" / f"{name}_{split}.csv",
            self.output_dir / "lsr_synth" / "chem_react" / f"{name}_{split}.csv",
            self.output_dir / "lsr_synth" / "matsci" / f"{name}_{split}.csv",
            self.output_dir / "lsr_synth" / "phys_osc" / f"{name}_{split}.csv",
        ]

        for path in search_paths:
            if path.exists():
                csv_path = path
                break

        if csv_path is None:
            raise FileNotFoundError(
                f"Dataset CSV not found for '{name}' (split: {split}). "
                f"Run 'python {__file__}' to extract datasets first."
            )

        # Load metadata
        metadata_df = self._load_metadata()

        # Find metadata for this dataset
        dataset_meta = metadata_df[metadata_df['name'] == name]

        if dataset_meta.empty:
            logger.warning(f"No metadata found for dataset {name}")
            metadata = {
                "name": name,
                "description": "No description available",
                "symbols": [],
                "symbol_descs": [],
                "expression": "Unknown"
            }
        else:
            row = dataset_meta.iloc[0]
            metadata = {
                "name": row['name'],
                "symbols": row['symbols'] if 'symbols' in row else [],
                "symbol_descs": row['symbol_descs'] if 'symbol_descs' in row else [],
                "symbol_properties": row['symbol_properties'] if 'symbol_properties' in row else [],
                "expression": row['expression'] if 'expression' in row else "Unknown",
                "dataset_group": row['dataset_group'] if 'dataset_group' in row else "Unknown",
            }

        return csv_path, metadata

    def list_datasets(self) -> List[str]:
        """
        List all available dataset names.

        Returns:
            List of dataset names
        """
        metadata_df = self._load_metadata()
        return metadata_df['name'].tolist()

    def get_dataset_info(self, name: str) -> Dict[str, any]:
        """
        Get detailed information about a dataset.

        Args:
            name: Dataset name

        Returns:
            Dictionary with dataset information
        """
        csv_path, metadata = self.get_dataset(name)

        # Load CSV to get shape
        df = pd.read_csv(csv_path)

        return {
            "name": name,
            "csv_path": str(csv_path),
            "num_rows": len(df),
            "num_features": len(df.columns) - 1,  # Exclude target column
            "columns": df.columns.tolist(),
            "metadata": metadata
        }


def main():
    """
    Main function to extract all datasets from HDF5 to CSV.
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("="*60)
    print("LLM-SRBench Dataset Extraction")
    print("="*60)

    dataset_manager = LLMSRBenchDataset()

    print(f"\nHDF5 file: {dataset_manager.hdf5_path}")
    print(f"Output directory: {dataset_manager.output_dir}")
    print(f"Metadata path: {dataset_manager.metadata_path}")

    # Check if HDF5 file exists
    if not dataset_manager.hdf5_path.exists():
        print(f"\nError: HDF5 file not found at {dataset_manager.hdf5_path}")
        sys.exit(1)

    # Extract all datasets
    print("\nExtracting datasets from HDF5...")
    stats = dataset_manager.extract_all_datasets(overwrite=True)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    print(f"Total datasets extracted: {len(stats)}")
    print(f"Output directory: {dataset_manager.output_dir}")

    # Show some examples
    print("\nExample datasets:")
    for i, (name, rows) in enumerate(list(stats.items())[:5]):
        print(f"  {name}: {rows} rows")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
