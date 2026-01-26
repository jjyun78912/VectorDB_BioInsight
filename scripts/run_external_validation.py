#!/usr/bin/env python3
"""
External Validation Pipeline Runner

Downloads independent GEO data, preprocesses it, runs the 6-Agent pipeline,
and validates ML predictions.

This provides TRUE external validation - completely independent of TCGA training data.

Usage:
    python scripts/run_external_validation.py --gse GSE81089 --cancer-type lung_cancer
    python scripts/run_external_validation.py --list
    python scripts/run_external_validation.py --download-all

Author: BioInsight AI
"""

import os
import sys
import json
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import requests

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# Ensembl to Gene Symbol mapping (top genes)
# Full mapping would require biomart or mygene
ENSEMBL_TO_SYMBOL = {}


class ExternalValidationRunner:
    """Run external validation pipeline on independent GEO datasets."""

    # GSE datasets with their cancer types
    DATASETS = {
        'GSE81089': {
            'cancer_type': 'LUAD',
            'cancer_name': 'lung_cancer',
            'description': 'NSCLC from Uppsala University Hospital (Sweden)',
            'url': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81089/suppl/GSE81089_readcounts_featurecounts.tsv.gz',
            'has_normal': True,
        },
        'GSE62254': {
            'cancer_type': 'STAD',
            'cancer_name': 'stomach_cancer',
            'description': 'Gastric cancer ACRG cohort (Samsung, Korea)',
            'url': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62254/suppl/GSE62254_RAW.tar',
            'has_normal': False,
        },
        'GSE53757': {
            'cancer_type': 'KIRC',
            'cancer_name': 'kidney_cancer',
            'description': 'Clear cell renal carcinoma',
            'url': None,  # Series matrix only
            'has_normal': True,
        },
    }

    def __init__(self, output_base: str = "data/external_validation"):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

    def download_and_prepare_gse81089(self) -> Path:
        """Download and prepare GSE81089 (NSCLC) for pipeline."""
        gse_id = 'GSE81089'
        gse_dir = self.output_base / gse_id
        gse_dir.mkdir(exist_ok=True)

        counts_file = gse_dir / "GSE81089_readcounts.tsv.gz"

        # Download if not exists
        if not counts_file.exists():
            print(f"Downloading {gse_id}...")
            url = self.DATASETS[gse_id]['url']
            response = requests.get(url, timeout=300)
            with open(counts_file, 'wb') as f:
                f.write(response.content)
            print(f"  Downloaded: {counts_file}")

        # Load and process
        print(f"Processing {gse_id}...")
        df = pd.read_csv(counts_file, sep='\t', compression='gzip', index_col=0)
        print(f"  Raw shape: {df.shape}")

        # Separate tumor and normal samples
        tumor_cols = [c for c in df.columns if c.endswith('T') or '_' in c]
        normal_cols = [c for c in df.columns if c.endswith('N')]

        print(f"  Tumor samples: {len(tumor_cols)}")
        print(f"  Normal samples: {len(normal_cols)}")

        # For DEG analysis, we need tumor vs normal
        # For ML prediction, we use all samples

        # Create pipeline input directory
        pipeline_dir = Path(f"data/GSE81089_lung_cancer_input")
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Map Ensembl IDs to Gene Symbols using biomart
        gene_mapping = self._get_gene_mapping(df.index.tolist())

        # Filter to mapped genes and rename
        mapped_genes = [g for g in df.index if g in gene_mapping]
        df_mapped = df.loc[mapped_genes].copy()
        df_mapped.index = [gene_mapping[g] for g in df_mapped.index]

        # Remove duplicates (keep max expression)
        df_mapped = df_mapped.groupby(df_mapped.index).max()
        print(f"  Mapped genes: {len(df_mapped)}")

        # Save count matrix (all samples for ML)
        df_mapped.to_csv(pipeline_dir / "count_matrix.csv")

        # Create metadata with tumor/normal labels
        metadata = []
        for col in df_mapped.columns:
            if col.endswith('N') or '_N' in col:
                condition = 'normal'
            else:
                condition = 'tumor'
            metadata.append({
                'sample_id': col,
                'condition': condition,
                'batch': 'GSE81089',
            })

        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(pipeline_dir / "metadata.csv", index=False)

        # Create config
        config = {
            'contrast': ['tumor', 'normal'],
            'cancer_type': 'lung_cancer',
            'source': 'GEO',
            'gse_id': gse_id,
            'padj_cutoff': 0.05,
            'log2fc_cutoff': 1.0,
            'expected_tcga_type': 'LUAD',  # For ML validation
        }
        with open(pipeline_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nPipeline input prepared: {pipeline_dir}")
        print(f"  - count_matrix.csv: {df_mapped.shape}")
        print(f"  - metadata.csv: {len(metadata_df)} samples")
        print(f"  - Tumor: {len([m for m in metadata if m['condition'] == 'tumor'])}")
        print(f"  - Normal: {len([m for m in metadata if m['condition'] == 'normal'])}")

        return pipeline_dir

    def _get_gene_mapping(self, ensembl_ids: list) -> Dict[str, str]:
        """Get Ensembl to Gene Symbol mapping using mygene."""
        try:
            import mygene
            mg = mygene.MyGeneInfo()

            # Query in batches
            print("  Fetching gene symbol mappings...")
            results = mg.querymany(
                ensembl_ids,
                scopes='ensembl.gene',
                fields='symbol',
                species='human',
                returnall=True
            )

            mapping = {}
            for hit in results['out']:
                if 'symbol' in hit and 'query' in hit:
                    mapping[hit['query']] = hit['symbol']

            print(f"  Mapped {len(mapping)}/{len(ensembl_ids)} genes")
            return mapping

        except ImportError:
            print("  mygene not installed. Using fallback mapping...")
            return self._get_fallback_mapping(ensembl_ids)

    def _get_fallback_mapping(self, ensembl_ids: list) -> Dict[str, str]:
        """Fallback: use Ensembl BioMart API."""
        # For now, just use Ensembl ID as-is (pipeline can handle both)
        # In production, would use BioMart REST API

        # Try to load cached mapping if exists
        cache_path = self.output_base / "ensembl_to_symbol_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        # Use BioMart REST API
        print("  Querying Ensembl BioMart (this may take a minute)...")

        # Split into chunks to avoid URL length limits
        chunk_size = 500
        all_mappings = {}

        for i in range(0, len(ensembl_ids), chunk_size):
            chunk = ensembl_ids[i:i+chunk_size]

            # BioMart XML query
            xml_query = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="0" count="" datasetConfigVersion="0.6">
    <Dataset name="hsapiens_gene_ensembl" interface="default">
        <Filter name="ensembl_gene_id" value="{','.join(chunk)}"/>
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="hgnc_symbol"/>
    </Dataset>
</Query>'''

            try:
                response = requests.post(
                    'http://www.ensembl.org/biomart/martservice',
                    data={'query': xml_query},
                    timeout=60
                )

                if response.status_code == 200:
                    for line in response.text.strip().split('\n'):
                        parts = line.split('\t')
                        if len(parts) >= 2 and parts[1]:
                            all_mappings[parts[0]] = parts[1]
            except Exception as e:
                print(f"    BioMart query failed: {e}")

            print(f"    Processed {min(i+chunk_size, len(ensembl_ids))}/{len(ensembl_ids)}")

        # Cache the results
        if all_mappings:
            with open(cache_path, 'w') as f:
                json.dump(all_mappings, f)

        print(f"  Mapped {len(all_mappings)}/{len(ensembl_ids)} genes")
        return all_mappings

    def run_pipeline(self, input_dir: Path, output_name: str = None) -> Dict:
        """Run the 6-Agent pipeline on prepared data."""
        from rnaseq_pipeline.orchestrator import RNAseqPipeline

        if output_name is None:
            output_name = input_dir.name.replace('_input', '')

        output_dir = Path(f"rnaseq_test_results/external_{output_name}")

        print(f"\n{'='*70}")
        print(f"Running 6-Agent Pipeline")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}\n")

        # Load config
        with open(input_dir / "config.json") as f:
            config = json.load(f)

        pipeline = RNAseqPipeline(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config=config,
            pipeline_type='bulk'
        )

        result = pipeline.run()

        # Validate ML prediction
        if 'expected_tcga_type' in config:
            self._validate_ml_prediction(result, config['expected_tcga_type'])

        return result

    def _validate_ml_prediction(self, result: Dict, expected_type: str):
        """Validate ML prediction against known cancer type."""
        print(f"\n{'='*70}")
        print("ML Prediction Validation (External Dataset)")
        print(f"{'='*70}")

        cancer_prediction = result.get('config', {}).get('cancer_prediction', {})

        if cancer_prediction:
            predicted = cancer_prediction.get('predicted_cancer', 'Unknown')
            confidence = cancer_prediction.get('confidence', 0)
            agreement = cancer_prediction.get('agreement_ratio', 0)

            print(f"  Expected (Ground Truth): {expected_type}")
            print(f"  ML Predicted: {predicted}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Sample Agreement: {agreement:.1%}")

            if predicted.upper() == expected_type.upper():
                print(f"\n  ✅ CORRECT - ML prediction matches ground truth!")
            elif predicted.upper() in ['LUAD', 'LUSC'] and expected_type.upper() in ['LUAD', 'LUSC']:
                print(f"\n  ⚠️ PARTIAL MATCH - Both are lung cancers (LUAD/LUSC confusion is expected)")
            else:
                print(f"\n  ❌ MISMATCH - ML prediction differs from ground truth")
        else:
            print("  No ML prediction available")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run external validation on independent GEO datasets')
    parser.add_argument('--gse', type=str, help='GSE ID to process (e.g., GSE81089)')
    parser.add_argument('--cancer-type', type=str, help='Override cancer type')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--download-only', action='store_true', help='Download and prepare without running pipeline')
    parser.add_argument('--run-only', type=str, help='Run pipeline on already prepared data (specify input dir)')

    args = parser.parse_args()

    runner = ExternalValidationRunner()

    if args.list:
        print("\n" + "="*70)
        print("Available External Validation Datasets")
        print("="*70)
        for gse_id, info in runner.DATASETS.items():
            print(f"\n{gse_id}:")
            print(f"  Cancer Type: {info['cancer_type']} ({info['cancer_name']})")
            print(f"  Description: {info['description']}")
            print(f"  Has Normal: {'Yes' if info['has_normal'] else 'No'}")
        return

    if args.run_only:
        input_dir = Path(args.run_only)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return
        runner.run_pipeline(input_dir)
        return

    # Default to GSE81089
    gse_id = args.gse or 'GSE81089'

    if gse_id == 'GSE81089':
        input_dir = runner.download_and_prepare_gse81089()

        if not args.download_only:
            runner.run_pipeline(input_dir)
    else:
        print(f"Dataset {gse_id} not yet implemented")


if __name__ == '__main__':
    main()
