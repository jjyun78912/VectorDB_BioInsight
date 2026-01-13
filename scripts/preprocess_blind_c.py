#!/usr/bin/env python3
"""
BLIND_C (GSE19804) 마이크로어레이 전처리 스크립트
================================================

1. Affymetrix 프로브 ID → 유전자 심볼 변환
2. Tumor/Normal 샘플 분리 (메타데이터 기반)
3. RNA-seq 파이프라인 입력 형식으로 변환

GSE19804: 대만 비흡연 여성 폐암 (60 tumor + 60 normal, paired)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_probe_annotation() -> Dict[str, str]:
    """Load Affymetrix GPL570 probe to gene symbol mapping"""
    annotation_path = Path("rnaseq_test_results/geo_lung_cancer/cache/GPL570_annotation.csv")
    if annotation_path.exists():
        df = pd.read_csv(annotation_path)
        mapping = {}
        for _, row in df.iterrows():
            probe = row['probe_id']
            gene = row['gene_symbol']
            if pd.notna(gene) and isinstance(gene, str) and gene.strip():
                mapping[probe] = gene.strip()
        return mapping
    return {}


def convert_probes_to_genes(expression_df: pd.DataFrame, probe_to_gene: Dict[str, str]) -> pd.DataFrame:
    """
    Convert probe IDs to gene symbols
    Input: rows=samples, columns=probes
    Output: rows=genes, columns=samples (Gene x Sample format)
    """
    # Get probe columns (exclude 'name')
    probe_cols = [c for c in expression_df.columns if c != 'name']
    sample_ids = expression_df['name'].tolist() if 'name' in expression_df.columns else expression_df.index.tolist()

    # Map probes to genes
    probe_to_symbol = {}
    for probe in probe_cols:
        gene = probe_to_gene.get(probe)
        if gene:
            probe_to_symbol[probe] = gene

    # Group probes by gene
    symbol_to_probes = {}
    for probe, symbol in probe_to_symbol.items():
        if symbol not in symbol_to_probes:
            symbol_to_probes[symbol] = []
        symbol_to_probes[symbol].append(probe)

    print(f"  Probes mapped: {len(probe_to_symbol)}")
    print(f"  Unique genes: {len(symbol_to_probes)}")

    # Average expression per gene (Gene x Sample)
    gene_data = {}
    for symbol, probes in symbol_to_probes.items():
        gene_data[symbol] = expression_df[probes].mean(axis=1).values

    # Create Gene x Sample dataframe
    gene_df = pd.DataFrame(gene_data, index=sample_ids).T
    gene_df.index.name = 'gene_id'

    return gene_df


def main():
    print("=" * 70)
    print("BLIND_C (GSE19804) Preprocessing for RNA-seq Pipeline")
    print("=" * 70)

    # Paths
    blind_c_path = Path("/Users/admin/blind_data/BLIND_C.csv")
    metadata_path = Path("data/blind_metadata/GSE19804_metadata.csv")
    output_dir = Path("data/blind_preprocessed/BLIND_C_GSE19804")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load expression data
    print("\n[1] Loading expression data...")
    expr_df = pd.read_csv(blind_c_path)
    print(f"  Raw shape: {expr_df.shape} (samples x probes)")

    # 2. Load metadata
    print("\n[2] Loading sample metadata...")
    meta_df = pd.read_csv(metadata_path)
    print(f"  Metadata: {len(meta_df)} samples")
    print(f"  Tissue distribution:")
    print(meta_df['tissue'].value_counts().to_string())

    # 3. Convert probes to genes
    print("\n[3] Converting probes to genes...")
    probe_to_gene = load_probe_annotation()
    print(f"  Loaded {len(probe_to_gene)} probe-gene mappings")

    gene_df = convert_probes_to_genes(expr_df, probe_to_gene)
    print(f"  Gene matrix: {gene_df.shape} (genes x samples)")

    # 4. Create sample metadata with condition
    print("\n[4] Creating sample metadata...")
    sample_meta = meta_df[['sample_id', 'title', 'tissue']].copy()
    sample_meta['condition'] = sample_meta['tissue'].apply(
        lambda x: 'tumor' if 'cancer' in x.lower() else 'normal'
    )

    # Extract patient ID for paired analysis (from title: "Lung Cancer 2T" -> "2")
    def extract_patient_id(title):
        import re
        match = re.search(r'(\d+)[TN]?$', title.replace(' ', ''))
        if match:
            return match.group(1)
        return title

    sample_meta['patient_id'] = sample_meta['title'].apply(extract_patient_id)

    print(f"  Condition distribution:")
    print(sample_meta['condition'].value_counts().to_string())

    # 5. Save processed data
    print("\n[5] Saving processed data...")

    # Count matrix (Gene x Sample)
    gene_df.to_csv(output_dir / 'count_matrix.csv')
    print(f"  Saved: count_matrix.csv")

    # Sample metadata
    sample_meta.to_csv(output_dir / 'sample_metadata.csv', index=False)
    print(f"  Saved: sample_metadata.csv")

    # Verify samples match
    expr_samples = set(gene_df.columns)
    meta_samples = set(sample_meta['sample_id'])
    common = expr_samples & meta_samples
    print(f"\n  Expression samples: {len(expr_samples)}")
    print(f"  Metadata samples: {len(meta_samples)}")
    print(f"  Common samples: {len(common)}")

    # 6. Create summary
    summary = {
        'dataset': 'GSE19804',
        'blind_file': 'BLIND_C',
        'cancer_type': 'lung_cancer',
        'description': 'Non-smoking female lung cancer in Taiwan',
        'n_samples': len(gene_df.columns),
        'n_genes': len(gene_df),
        'n_tumor': len(sample_meta[sample_meta['condition'] == 'tumor']),
        'n_normal': len(sample_meta[sample_meta['condition'] == 'normal']),
        'paired': True,
        'platform': 'GPL570 (Affymetrix Human Genome U133 Plus 2.0)',
    }

    import json
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: dataset_info.json")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Ready for RNA-seq pipeline analysis")

    return output_dir


if __name__ == "__main__":
    main()
