"""
RNA-seq Data Type Detector

Automatically detects whether input data is:
- Bulk RNA-seq (6-100 samples, requires 2-step pipeline)
- Single-cell RNA-seq (1,000+ cells, uses 1-step Scanpy pipeline)

Detection Criteria:
1. Sample/cell count: >= 500 cells → Single-cell
2. Matrix shape: genes × samples/cells
3. File format: .h5ad, 10X formats → Single-cell
4. Metadata hints: cell_type, cluster columns → Single-cell
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Literal
import json
import logging

logger = logging.getLogger(__name__)

# Type alias
DataType = Literal["bulk", "singlecell", "unknown"]


class DataTypeDetector:
    """Detect whether RNA-seq data is bulk or single-cell."""

    # Thresholds for detection
    SINGLECELL_MIN_SAMPLES = 500  # >= 500 columns likely single-cell
    BULK_MAX_SAMPLES = 200  # <= 200 columns likely bulk

    # Single-cell specific column names
    SINGLECELL_HINTS = [
        'cell_type', 'celltype', 'cell_id', 'barcode', 'cluster',
        'leiden', 'louvain', 'umap_1', 'umap_2', 'tsne_1', 'tsne_2',
        'n_genes_by_counts', 'total_counts', 'pct_counts_mt'
    ]

    # Bulk-specific column names
    BULK_HINTS = [
        'condition', 'treatment', 'control', 'tumor', 'normal',
        'patient_id', 'sample_id', 'replicate', 'batch', 'tissue'
    ]

    def __init__(self, input_dir: Path):
        """
        Initialize detector with input directory.

        Args:
            input_dir: Directory containing count_matrix.csv, metadata.csv, etc.
        """
        self.input_dir = Path(input_dir)
        self.detection_result: Dict[str, Any] = {}

    def detect(self) -> Dict[str, Any]:
        """
        Detect data type and return detailed result.

        Returns:
            {
                "data_type": "bulk" | "singlecell" | "unknown",
                "confidence": float (0-1),
                "n_genes": int,
                "n_samples": int,
                "evidence": {
                    "sample_count": str,
                    "file_format": str,
                    "metadata_hints": list,
                    "matrix_characteristics": str
                },
                "recommended_pipeline": str
            }
        """
        result = {
            "data_type": "unknown",
            "confidence": 0.0,
            "n_genes": 0,
            "n_samples": 0,
            "evidence": {},
            "recommended_pipeline": ""
        }

        scores = {"bulk": 0, "singlecell": 0}
        evidence = {}

        # 1. Check file formats
        file_score, file_evidence = self._check_file_formats()
        scores["singlecell"] += file_score.get("singlecell", 0)
        scores["bulk"] += file_score.get("bulk", 0)
        evidence["file_format"] = file_evidence

        # 2. Check matrix shape
        shape_score, shape_evidence, n_genes, n_samples = self._check_matrix_shape()
        scores["singlecell"] += shape_score.get("singlecell", 0)
        scores["bulk"] += shape_score.get("bulk", 0)
        evidence["sample_count"] = shape_evidence
        result["n_genes"] = n_genes
        result["n_samples"] = n_samples

        # 3. Check metadata
        meta_score, meta_evidence = self._check_metadata()
        scores["singlecell"] += meta_score.get("singlecell", 0)
        scores["bulk"] += meta_score.get("bulk", 0)
        evidence["metadata_hints"] = meta_evidence

        # 4. Check matrix characteristics (sparsity)
        char_score, char_evidence = self._check_matrix_characteristics()
        scores["singlecell"] += char_score.get("singlecell", 0)
        scores["bulk"] += char_score.get("bulk", 0)
        evidence["matrix_characteristics"] = char_evidence

        # Determine final type
        total_score = scores["bulk"] + scores["singlecell"]
        if total_score > 0:
            if scores["singlecell"] > scores["bulk"]:
                result["data_type"] = "singlecell"
                result["confidence"] = scores["singlecell"] / total_score
            else:
                result["data_type"] = "bulk"
                result["confidence"] = scores["bulk"] / total_score

        # Set recommended pipeline
        if result["data_type"] == "singlecell":
            result["recommended_pipeline"] = "SingleCellAgent (Scanpy 1-Step)"
        elif result["data_type"] == "bulk":
            if n_samples >= 6:
                result["recommended_pipeline"] = "Bulk 6-Agent Pipeline (DESeq2 2-Step)"
            else:
                result["recommended_pipeline"] = "Bulk Pre-computed (Fold Change only, samples < 6)"
        else:
            result["recommended_pipeline"] = "Unknown - Manual selection required"

        result["evidence"] = evidence
        self.detection_result = result

        logger.info(f"Data type detected: {result['data_type']} (confidence: {result['confidence']:.2f})")
        logger.info(f"Recommended pipeline: {result['recommended_pipeline']}")

        return result

    def _check_file_formats(self) -> Tuple[Dict[str, int], str]:
        """Check for single-cell specific file formats."""
        scores = {"bulk": 0, "singlecell": 0}
        evidence = []

        # Single-cell formats
        h5ad_files = list(self.input_dir.glob("*.h5ad"))
        if h5ad_files:
            scores["singlecell"] += 3
            evidence.append(f"Found .h5ad file: {h5ad_files[0].name}")

        # 10X Genomics formats
        if (self.input_dir / "matrix.mtx").exists() or \
           (self.input_dir / "matrix.mtx.gz").exists():
            scores["singlecell"] += 3
            evidence.append("Found 10X matrix.mtx format")

        if (self.input_dir / "barcodes.tsv").exists() or \
           (self.input_dir / "barcodes.tsv.gz").exists():
            scores["singlecell"] += 2
            evidence.append("Found 10X barcodes.tsv")

        # Standard bulk formats
        csv_files = list(self.input_dir.glob("count*.csv")) + \
                    list(self.input_dir.glob("*counts*.csv"))
        if csv_files and not h5ad_files:
            scores["bulk"] += 1
            evidence.append(f"Found bulk-style CSV: {csv_files[0].name}")

        return scores, "; ".join(evidence) if evidence else "No format hints"

    def _check_matrix_shape(self) -> Tuple[Dict[str, int], str, int, int]:
        """Check matrix dimensions."""
        scores = {"bulk": 0, "singlecell": 0}
        n_genes, n_samples = 0, 0

        # Try to load matrix
        count_file = self.input_dir / "count_matrix.csv"
        if not count_file.exists():
            # Try alternatives
            for alt in ["counts.csv", "expression.csv", "matrix.csv"]:
                alt_file = self.input_dir / alt
                if alt_file.exists():
                    count_file = alt_file
                    break

        if count_file.exists():
            try:
                # Read header only first
                df_head = pd.read_csv(count_file, nrows=5)
                n_samples = len(df_head.columns) - 1  # Exclude gene_id column

                # Count rows for genes
                n_genes = sum(1 for _ in open(count_file)) - 1  # Exclude header

                if n_samples >= self.SINGLECELL_MIN_SAMPLES:
                    scores["singlecell"] += 4
                    evidence = f"{n_samples} samples/cells (>={self.SINGLECELL_MIN_SAMPLES} → likely single-cell)"
                elif n_samples <= self.BULK_MAX_SAMPLES:
                    scores["bulk"] += 4
                    evidence = f"{n_samples} samples (<={self.BULK_MAX_SAMPLES} → likely bulk)"
                else:
                    # Ambiguous range
                    evidence = f"{n_samples} samples (ambiguous range)"

            except Exception as e:
                logger.warning(f"Error reading count matrix: {e}")
                evidence = "Could not read count matrix"
        else:
            # Try h5ad
            h5ad_files = list(self.input_dir.glob("*.h5ad"))
            if h5ad_files:
                try:
                    import scanpy as sc
                    adata = sc.read_h5ad(h5ad_files[0])
                    n_genes = adata.n_vars
                    n_samples = adata.n_obs

                    if n_samples >= self.SINGLECELL_MIN_SAMPLES:
                        scores["singlecell"] += 4
                        evidence = f"{n_samples} cells from h5ad (single-cell)"
                    else:
                        evidence = f"{n_samples} observations from h5ad"
                except:
                    evidence = "Could not read h5ad"
            else:
                evidence = "No count matrix found"

        return scores, evidence, n_genes, n_samples

    def _check_metadata(self) -> Tuple[Dict[str, int], list]:
        """Check metadata for hints."""
        scores = {"bulk": 0, "singlecell": 0}
        evidence = []

        meta_file = self.input_dir / "metadata.csv"
        if not meta_file.exists():
            return scores, ["No metadata.csv found"]

        try:
            meta_df = pd.read_csv(meta_file)
            columns = [c.lower() for c in meta_df.columns]

            # Check for single-cell hints
            sc_matches = [h for h in self.SINGLECELL_HINTS if h in columns]
            if sc_matches:
                scores["singlecell"] += len(sc_matches)
                evidence.append(f"Single-cell columns: {sc_matches}")

            # Check for bulk hints
            bulk_matches = [h for h in self.BULK_HINTS if h in columns]
            if bulk_matches:
                scores["bulk"] += len(bulk_matches)
                evidence.append(f"Bulk columns: {bulk_matches}")

        except Exception as e:
            evidence.append(f"Error reading metadata: {e}")

        return scores, evidence

    def _check_matrix_characteristics(self) -> Tuple[Dict[str, int], str]:
        """Check matrix sparsity and value distribution."""
        scores = {"bulk": 0, "singlecell": 0}

        count_file = self.input_dir / "count_matrix.csv"
        if not count_file.exists():
            return scores, "No matrix to analyze"

        try:
            # Sample a portion of the matrix
            df_sample = pd.read_csv(count_file, nrows=1000)

            # Get numeric columns only
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return scores, "No numeric data"

            values = df_sample[numeric_cols].values.flatten()

            # Calculate sparsity (% zeros)
            sparsity = (values == 0).sum() / len(values) * 100

            # Single-cell typically has >80% zeros
            if sparsity > 80:
                scores["singlecell"] += 2
                evidence = f"High sparsity ({sparsity:.1f}% zeros) → likely single-cell"
            elif sparsity < 30:
                scores["bulk"] += 2
                evidence = f"Low sparsity ({sparsity:.1f}% zeros) → likely bulk"
            else:
                evidence = f"Moderate sparsity ({sparsity:.1f}% zeros)"

            return scores, evidence

        except Exception as e:
            return scores, f"Error analyzing matrix: {e}"

    def get_pipeline_type(self) -> DataType:
        """Get the detected data type."""
        if not self.detection_result:
            self.detect()
        return self.detection_result.get("data_type", "unknown")

    def save_result(self, output_path: Optional[Path] = None) -> None:
        """Save detection result to JSON."""
        if not self.detection_result:
            self.detect()

        output_path = output_path or (self.input_dir / "data_type_detection.json")
        with open(output_path, 'w') as f:
            json.dump(self.detection_result, f, indent=2)
        logger.info(f"Detection result saved to {output_path}")


def detect_data_type(input_dir: Path) -> Dict[str, Any]:
    """
    Convenience function to detect data type.

    Args:
        input_dir: Directory containing RNA-seq data

    Returns:
        Detection result dictionary
    """
    detector = DataTypeDetector(input_dir)
    return detector.detect()


def is_singlecell(input_dir: Path) -> bool:
    """Quick check if data is single-cell."""
    result = detect_data_type(input_dir)
    return result["data_type"] == "singlecell"


def is_bulk(input_dir: Path) -> bool:
    """Quick check if data is bulk."""
    result = detect_data_type(input_dir)
    return result["data_type"] == "bulk"


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Detect RNA-seq data type")
    parser.add_argument("input_dir", type=Path, help="Input directory with RNA-seq data")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}")
        sys.exit(1)

    result = detect_data_type(args.input_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"RNA-seq Data Type Detection")
        print(f"{'='*50}")
        print(f"Data Type: {result['data_type'].upper()}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Genes: {result['n_genes']:,}")
        print(f"Samples/Cells: {result['n_samples']:,}")
        print(f"Recommended: {result['recommended_pipeline']}")
        print(f"\nEvidence:")
        for key, val in result['evidence'].items():
            print(f"  - {key}: {val}")
