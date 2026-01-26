#!/usr/bin/env python3
"""
External Validation Data Downloader for Pan-Cancer ML Model

Downloads RNA-seq data from GEO (independent of TCGA) for external validation.
These datasets are completely separate from the TCGA training data to avoid data leakage.

Independent External Datasets:
- Breast: GSE58135 (TNBC + ER+), GSE96058
- Lung: GSE81089 (LUAD), GSE30219 (multi-type)
- Colorectal: GSE39582, GSE87211
- Liver: GSE14520, GSE76427
- Kidney: GSE53757 (KIRC)
- Stomach: GSE62254 (STAD)
- Pancreas: GSE28735
- Prostate: GSE21034

Author: BioInsight AI
"""

import os
import sys
import json
import gzip
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class GEODataDownloader:
    """Download and process GEO RNA-seq data for external validation."""

    # Independent GEO datasets (NOT from TCGA)
    # Each entry: GSE_ID, cancer_type (TCGA code), description, expected_samples
    EXTERNAL_DATASETS = {
        # Breast Cancer - Independent cohorts
        'GSE58135': {
            'cancer_type': 'BRCA',
            'description': 'TNBC + ER+ breast cancer (Memorial Sloan Kettering)',
            'samples': 154,
            'has_normal': True,
        },
        'GSE96058': {
            'cancer_type': 'BRCA',
            'description': 'Sweden SCAN-B breast cancer cohort',
            'samples': 3409,
            'has_normal': False,
        },

        # Lung Cancer - Independent cohorts
        'GSE81089': {
            'cancer_type': 'LUAD',
            'description': 'Lung adenocarcinoma (Samsung Medical Center, Korea)',
            'samples': 199,
            'has_normal': True,
        },
        'GSE30219': {
            'cancer_type': 'LUAD',  # Mixed lung types
            'description': 'Non-small cell lung cancer (France)',
            'samples': 307,
            'has_normal': True,
        },

        # Colorectal Cancer - Independent cohorts
        'GSE39582': {
            'cancer_type': 'COAD',
            'description': 'Colorectal cancer (Cartes d\'Identite des Tumeurs, France)',
            'samples': 585,
            'has_normal': False,
        },
        'GSE87211': {
            'cancer_type': 'COAD',
            'description': 'Rectal cancer pre-treatment biopsies',
            'samples': 363,
            'has_normal': True,
        },

        # Liver Cancer - Independent cohorts
        'GSE14520': {
            'cancer_type': 'LIHC',
            'description': 'Hepatocellular carcinoma (HCC) LCI cohort',
            'samples': 445,
            'has_normal': True,
        },
        'GSE76427': {
            'cancer_type': 'LIHC',
            'description': 'HCC with survival data',
            'samples': 167,
            'has_normal': False,
        },

        # Kidney Cancer - Independent cohorts
        'GSE53757': {
            'cancer_type': 'KIRC',
            'description': 'Clear cell renal carcinoma (ccRCC)',
            'samples': 144,
            'has_normal': True,
        },

        # Stomach Cancer - Independent cohorts
        'GSE62254': {
            'cancer_type': 'STAD',
            'description': 'Gastric cancer (Samsung, ACRG cohort)',
            'samples': 300,
            'has_normal': False,
        },

        # Pancreatic Cancer - Independent cohorts
        'GSE28735': {
            'cancer_type': 'PAAD',
            'description': 'Pancreatic ductal adenocarcinoma',
            'samples': 90,
            'has_normal': True,
        },

        # Prostate Cancer - Independent cohorts
        'GSE21034': {
            'cancer_type': 'PRAD',
            'description': 'Prostate cancer (MSKCC)',
            'samples': 179,
            'has_normal': True,
        },

        # Head & Neck Cancer - Independent cohorts
        'GSE65858': {
            'cancer_type': 'HNSC',
            'description': 'Head and neck squamous cell carcinoma',
            'samples': 270,
            'has_normal': False,
        },

        # Bladder Cancer - Independent cohorts
        'GSE48075': {
            'cancer_type': 'BLCA',
            'description': 'Bladder cancer (Sweden)',
            'samples': 142,
            'has_normal': False,
        },

        # Ovarian Cancer - Independent cohorts
        'GSE26712': {
            'cancer_type': 'OV',
            'description': 'Ovarian cancer (Duke)',
            'samples': 195,
            'has_normal': True,
        },

        # Thyroid Cancer - Independent cohorts
        'GSE60542': {
            'cancer_type': 'THCA',
            'description': 'Papillary thyroid carcinoma',
            'samples': 122,
            'has_normal': True,
        },

        # Melanoma - Independent cohorts
        'GSE65904': {
            'cancer_type': 'SKCM',
            'description': 'Cutaneous melanoma (Leeds, UK)',
            'samples': 214,
            'has_normal': False,
        },
    }

    def __init__(self, output_dir: str = "data/external_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "https://www.ncbi.nlm.nih.gov/geo"
        self.ftp_base = "https://ftp.ncbi.nlm.nih.gov/geo/series"

    def download_geo_series(self, gse_id: str) -> Optional[Dict]:
        """Download a GEO series and extract expression data."""
        print(f"\n{'='*60}")
        print(f"Downloading {gse_id}...")
        print(f"{'='*60}")

        dataset_info = self.EXTERNAL_DATASETS.get(gse_id, {})
        cancer_type = dataset_info.get('cancer_type', 'UNKNOWN')

        # Create output directory
        gse_dir = self.output_dir / gse_id
        gse_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Get series matrix file (contains expression data)
            matrix_file = self._download_series_matrix(gse_id, gse_dir)

            if matrix_file is None:
                print(f"  Warning: Could not download matrix for {gse_id}")
                return None

            # Step 2: Parse the matrix file
            expression_df, sample_info = self._parse_series_matrix(matrix_file)

            if expression_df is None:
                print(f"  Warning: Could not parse matrix for {gse_id}")
                return None

            # Step 3: Save processed data
            output_path = gse_dir / "expression_matrix.csv"
            expression_df.to_csv(output_path)
            print(f"  Saved expression matrix: {expression_df.shape}")

            # Save sample info
            if sample_info is not None:
                sample_info.to_csv(gse_dir / "sample_info.csv", index=False)

            # Save metadata
            metadata = {
                'gse_id': gse_id,
                'cancer_type': cancer_type,
                'description': dataset_info.get('description', ''),
                'n_samples': expression_df.shape[1],
                'n_genes': expression_df.shape[0],
                'download_date': datetime.now().isoformat(),
            }
            with open(gse_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            return metadata

        except Exception as e:
            print(f"  Error downloading {gse_id}: {e}")
            return None

    def _download_series_matrix(self, gse_id: str, output_dir: Path) -> Optional[Path]:
        """Download the series matrix file from GEO FTP."""
        # Construct FTP URL
        gse_prefix = gse_id[:len(gse_id)-3] + "nnn"  # e.g., GSE58nnn

        # Try different matrix file patterns
        patterns = [
            f"{gse_id}_series_matrix.txt.gz",
            f"{gse_id}-GPL570_series_matrix.txt.gz",  # Affymetrix
            f"{gse_id}-GPL96_series_matrix.txt.gz",
            f"{gse_id}-GPL6244_series_matrix.txt.gz",  # HuGene
        ]

        ftp_url_base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_prefix}/{gse_id}/matrix/"

        for pattern in patterns:
            url = ftp_url_base + pattern
            local_file = output_dir / pattern

            try:
                print(f"  Trying: {url}")
                response = requests.get(url, timeout=60)

                if response.status_code == 200:
                    with open(local_file, 'wb') as f:
                        f.write(response.content)
                    print(f"  Downloaded: {pattern}")
                    return local_file
            except Exception as e:
                continue

        # Try soft file as fallback
        soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_prefix}/{gse_id}/soft/{gse_id}_family.soft.gz"
        local_soft = output_dir / f"{gse_id}_family.soft.gz"

        try:
            print(f"  Trying SOFT file: {soft_url}")
            response = requests.get(soft_url, timeout=120)
            if response.status_code == 200:
                with open(local_soft, 'wb') as f:
                    f.write(response.content)
                print(f"  Downloaded SOFT file")
                return local_soft
        except:
            pass

        return None

    def _parse_series_matrix(self, matrix_file: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Parse a GEO series matrix file."""
        try:
            # Read gzipped file
            if str(matrix_file).endswith('.gz'):
                import gzip
                with gzip.open(matrix_file, 'rt', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            else:
                with open(matrix_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

            # Parse header and data
            sample_info = {}
            data_start = 0
            data_end = len(lines)

            for i, line in enumerate(lines):
                if line.startswith('!Sample_geo_accession'):
                    sample_ids = line.strip().split('\t')[1:]
                    sample_ids = [s.strip('"') for s in sample_ids]
                    sample_info['sample_id'] = sample_ids
                elif line.startswith('!Sample_title'):
                    titles = line.strip().split('\t')[1:]
                    titles = [t.strip('"') for t in titles]
                    sample_info['title'] = titles
                elif line.startswith('!Sample_characteristics'):
                    chars = line.strip().split('\t')[1:]
                    chars = [c.strip('"') for c in chars]
                    # Parse characteristic name
                    if chars and ':' in chars[0]:
                        char_name = chars[0].split(':')[0].strip()
                        char_values = [c.split(':')[-1].strip() if ':' in c else c for c in chars]
                        sample_info[char_name] = char_values
                elif line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                    data_start = i
                elif line.startswith('!series_matrix_table_end'):
                    data_end = i
                    break

            # Parse expression data
            if data_start > 0:
                # Read expression data
                data_lines = lines[data_start:data_end]

                # Parse header
                header = data_lines[0].strip().split('\t')
                header = [h.strip('"') for h in header]

                # Parse data rows
                data_rows = []
                gene_ids = []
                for line in data_lines[1:]:
                    if line.strip() and not line.startswith('!'):
                        parts = line.strip().split('\t')
                        if len(parts) > 1:
                            gene_id = parts[0].strip('"')
                            values = []
                            for v in parts[1:]:
                                try:
                                    values.append(float(v.strip('"')))
                                except:
                                    values.append(np.nan)
                            if len(values) == len(header) - 1:
                                gene_ids.append(gene_id)
                                data_rows.append(values)

                if data_rows:
                    expression_df = pd.DataFrame(
                        data_rows,
                        index=gene_ids,
                        columns=header[1:]
                    )
                    expression_df.index.name = 'gene_id'

                    # Create sample info DataFrame
                    sample_df = None
                    if sample_info:
                        sample_df = pd.DataFrame(sample_info)

                    return expression_df, sample_df

            return None, None

        except Exception as e:
            print(f"  Error parsing matrix: {e}")
            return None, None

    def download_all_datasets(self, cancer_types: Optional[List[str]] = None) -> Dict:
        """Download all external validation datasets."""
        results = {}

        for gse_id, info in self.EXTERNAL_DATASETS.items():
            # Filter by cancer type if specified
            if cancer_types and info['cancer_type'] not in cancer_types:
                continue

            result = self.download_geo_series(gse_id)
            if result:
                results[gse_id] = result

            # Rate limiting
            time.sleep(2)

        # Save summary
        summary_path = self.output_dir / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Download Complete!")
        print(f"  Successful: {len(results)} datasets")
        print(f"  Summary saved: {summary_path}")
        print(f"{'='*60}")

        return results

    def prepare_for_pipeline(self, gse_id: str, output_name: str = None) -> Optional[Path]:
        """Prepare downloaded data for the RNA-seq pipeline."""
        gse_dir = self.output_dir / gse_id

        if not gse_dir.exists():
            print(f"Dataset {gse_id} not found. Downloading...")
            result = self.download_geo_series(gse_id)
            if result is None:
                return None

        # Load expression data
        expr_path = gse_dir / "expression_matrix.csv"
        if not expr_path.exists():
            print(f"Expression matrix not found for {gse_id}")
            return None

        expression_df = pd.read_csv(expr_path, index_col=0)

        # Load metadata
        with open(gse_dir / "metadata.json") as f:
            metadata = json.load(f)

        cancer_type = metadata.get('cancer_type', 'UNKNOWN')

        # Create pipeline input directory
        if output_name is None:
            output_name = f"{gse_id}_{cancer_type}"

        pipeline_input = Path(f"data/{output_name}_input")
        pipeline_input.mkdir(parents=True, exist_ok=True)

        # Save count matrix (genes x samples format)
        count_matrix = expression_df.copy()
        count_matrix.to_csv(pipeline_input / "count_matrix.csv")

        # Create metadata file
        sample_info_path = gse_dir / "sample_info.csv"
        if sample_info_path.exists():
            sample_info = pd.read_csv(sample_info_path)

            # Create pipeline metadata format
            pipeline_metadata = pd.DataFrame({
                'sample_id': sample_info['sample_id'] if 'sample_id' in sample_info.columns else expression_df.columns,
                'condition': ['tumor'] * len(expression_df.columns),  # Default to tumor
                'batch': ['batch1'] * len(expression_df.columns),
            })
        else:
            pipeline_metadata = pd.DataFrame({
                'sample_id': expression_df.columns,
                'condition': ['tumor'] * len(expression_df.columns),
                'batch': ['batch1'] * len(expression_df.columns),
            })

        pipeline_metadata.to_csv(pipeline_input / "metadata.csv", index=False)

        # Create config
        config = {
            'contrast': ['tumor', 'normal'],
            'cancer_type': cancer_type.lower() + '_cancer' if not cancer_type.lower().endswith('cancer') else cancer_type.lower(),
            'source': 'GEO',
            'gse_id': gse_id,
            'padj_cutoff': 0.05,
            'log2fc_cutoff': 1.0,
        }

        # Map TCGA code to cancer_type name
        tcga_to_name = {
            'BRCA': 'breast_cancer',
            'LUAD': 'lung_cancer',
            'LUSC': 'lung_cancer',
            'COAD': 'colorectal_cancer',
            'LIHC': 'liver_cancer',
            'KIRC': 'kidney_cancer',
            'STAD': 'stomach_cancer',
            'PAAD': 'pancreatic_cancer',
            'PRAD': 'prostate_cancer',
            'HNSC': 'head_neck_cancer',
            'BLCA': 'bladder_cancer',
            'OV': 'ovarian_cancer',
            'THCA': 'thyroid_cancer',
            'SKCM': 'melanoma',
        }
        config['cancer_type'] = tcga_to_name.get(cancer_type, 'unknown')

        with open(pipeline_input / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nPipeline input prepared: {pipeline_input}")
        print(f"  - count_matrix.csv: {expression_df.shape}")
        print(f"  - metadata.csv: {len(pipeline_metadata)} samples")
        print(f"  - config.json: cancer_type={config['cancer_type']}")

        return pipeline_input


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download external validation datasets from GEO')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--download', type=str, help='Download specific GSE ID')
    parser.add_argument('--download-all', action='store_true', help='Download all datasets')
    parser.add_argument('--cancer-types', type=str, nargs='+',
                        help='Filter by cancer types (e.g., BRCA LUAD)')
    parser.add_argument('--prepare', type=str, help='Prepare GSE for pipeline')
    parser.add_argument('--output', type=str, default='data/external_validation',
                        help='Output directory')

    args = parser.parse_args()

    downloader = GEODataDownloader(output_dir=args.output)

    if args.list:
        print("\n" + "="*70)
        print("Available External Validation Datasets (Independent of TCGA)")
        print("="*70)
        print(f"\n{'GSE ID':<12} {'Cancer':<8} {'Samples':<10} {'Description'}")
        print("-"*70)
        for gse_id, info in downloader.EXTERNAL_DATASETS.items():
            print(f"{gse_id:<12} {info['cancer_type']:<8} {info['samples']:<10} {info['description'][:40]}")
        print("-"*70)
        print(f"Total: {len(downloader.EXTERNAL_DATASETS)} datasets")
        return

    if args.download:
        downloader.download_geo_series(args.download)
        return

    if args.download_all:
        downloader.download_all_datasets(cancer_types=args.cancer_types)
        return

    if args.prepare:
        downloader.prepare_for_pipeline(args.prepare)
        return

    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()
