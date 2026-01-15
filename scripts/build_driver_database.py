#!/usr/bin/env python3
"""
Build Driver Gene Database from IntOGen data.

Creates a processed database of driver genes with mutation frequencies
mapped to our 17 ML cancer types.
"""

import pandas as pd
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INTOGEN_DIR = PROJECT_ROOT / "data" / "intogen" / "2024-06-18_IntOGen-Drivers"
OUTPUT_DIR = PROJECT_ROOT / "data" / "driver_db"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IntOGen cancer type to our ML cancer type mapping
CANCER_TYPE_MAP = {
    # Breast cancer
    'BRCA': 'breast_cancer',

    # Lung cancer
    'LUAD': 'lung_cancer',
    'LUSC': 'lung_cancer',
    'NSCLC': 'lung_cancer',
    'SCLC': 'lung_cancer',

    # Colorectal cancer
    'COAD': 'colorectal_cancer',
    'COADREAD': 'colorectal_cancer',
    'READ': 'colorectal_cancer',

    # Pancreatic cancer
    'PAAD': 'pancreatic_cancer',
    'PANET': 'pancreatic_cancer',

    # Liver cancer
    'HCC': 'liver_cancer',
    'LIHC': 'liver_cancer',

    # Glioblastoma
    'GBM': 'glioblastoma',

    # Low-grade glioma
    'LGG': 'low_grade_glioma',
    'HGGNOS': 'low_grade_glioma',  # High-grade glioma NOS

    # Blood cancer (AML, etc.)
    'AML': 'blood_cancer',
    'ALL': 'blood_cancer',
    'CLL': 'blood_cancer',
    'PCM': 'blood_cancer',  # Plasma cell myeloma
    'DLBCL': 'blood_cancer',
    'NHL': 'blood_cancer',

    # Kidney cancer
    'KIRC': 'kidney_cancer',
    'KIRP': 'kidney_cancer',
    'KICH': 'kidney_cancer',
    'RCC': 'kidney_cancer',

    # Prostate cancer
    'PRAD': 'prostate_cancer',

    # Ovarian cancer
    'OV': 'ovarian_cancer',
    'OVT': 'ovarian_cancer',

    # Stomach cancer
    'STAD': 'stomach_cancer',

    # Bladder cancer
    'BLCA': 'bladder_cancer',

    # Thyroid cancer
    'THCA': 'thyroid_cancer',
    'WDTC': 'thyroid_cancer',  # Well-differentiated thyroid carcinoma

    # Melanoma
    'MEL': 'melanoma',
    'SKCM': 'melanoma',

    # Head and neck cancer
    'HNSC': 'head_neck_cancer',

    # Uterine cancer
    'UCEC': 'uterine_cancer',
}

# Our 17 ML cancer types
ML_CANCER_TYPES = [
    'breast_cancer', 'lung_cancer', 'colorectal_cancer', 'pancreatic_cancer',
    'liver_cancer', 'glioblastoma', 'low_grade_glioma', 'blood_cancer',
    'kidney_cancer', 'prostate_cancer', 'ovarian_cancer', 'stomach_cancer',
    'bladder_cancer', 'thyroid_cancer', 'melanoma', 'head_neck_cancer',
    'uterine_cancer'
]


def load_intogen_data():
    """Load IntOGen Compendium Cancer Genes data."""
    filepath = INTOGEN_DIR / "Compendium_Cancer_Genes.tsv"
    if not filepath.exists():
        raise FileNotFoundError(f"IntOGen data not found: {filepath}")

    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded IntOGen data: {len(df)} rows, {df['SYMBOL'].nunique()} genes")
    return df


def process_intogen_data(df: pd.DataFrame) -> dict:
    """
    Process IntOGen data into our driver database format.

    Output format:
    {
        'cancer_type': {
            'GENE': {
                'mutation_freq': 0.34,
                'samples': 367,
                'total_samples': 1009,
                'role': 'Oncogene',  # or 'TSG', 'Unknown'
                'cgc_gene': True,
                'methods': ['dndscv', 'oncodriveclustl'],
                'qvalue': 0.001
            }
        }
    }
    """
    driver_db = {ct: {} for ct in ML_CANCER_TYPES}

    # Filter for driver genes only
    drivers = df[df['IS_DRIVER'] == True].copy()
    print(f"Driver genes: {len(drivers)} entries")

    # Map IntOGen cancer types to our types
    drivers['ml_cancer_type'] = drivers['CANCER_TYPE'].map(CANCER_TYPE_MAP)
    mapped = drivers[drivers['ml_cancer_type'].notna()]
    print(f"Mapped to ML types: {len(mapped)} entries")

    # Process each entry
    for _, row in mapped.iterrows():
        cancer_type = row['ml_cancer_type']
        gene = row['SYMBOL']

        # Parse role
        role = row['ROLE']
        if role == 'Act':
            role = 'Oncogene'
        elif role == 'LoF':
            role = 'TSG'
        else:
            role = 'Unknown'

        # Parse methods
        methods = row['METHODS'].split(',') if pd.notna(row['METHODS']) else []

        # If gene already exists for this cancer type, keep the one with more samples
        if gene in driver_db[cancer_type]:
            existing = driver_db[cancer_type][gene]
            if row['SAMPLES'] <= existing['samples']:
                continue

        driver_db[cancer_type][gene] = {
            'mutation_freq': float(row['%_SAMPLES_COHORT']) if pd.notna(row['%_SAMPLES_COHORT']) else 0,
            'samples': int(row['SAMPLES']) if pd.notna(row['SAMPLES']) else 0,
            'total_samples': int(row['TOTAL_SAMPLES']) if pd.notna(row['TOTAL_SAMPLES']) else 0,
            'role': role,
            'cgc_gene': bool(row['CGC_GENE']),
            'methods': methods,
            'qvalue': float(row['QVALUE_COMBINATION']) if pd.notna(row['QVALUE_COMBINATION']) else 1.0
        }

    return driver_db


def generate_summary(driver_db: dict) -> dict:
    """Generate summary statistics."""
    summary = {
        'total_genes': set(),
        'by_cancer_type': {}
    }

    for cancer_type, genes in driver_db.items():
        summary['total_genes'].update(genes.keys())
        summary['by_cancer_type'][cancer_type] = {
            'gene_count': len(genes),
            'top_genes': sorted(
                [(g, d['mutation_freq']) for g, d in genes.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    summary['total_genes'] = len(summary['total_genes'])
    return summary


def save_database(driver_db: dict, summary: dict):
    """Save processed database to files."""
    # Save full database as JSON
    with open(OUTPUT_DIR / "intogen_driver_db.json", 'w') as f:
        json.dump(driver_db, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'intogen_driver_db.json'}")

    # Save per-cancer-type CSVs
    for cancer_type, genes in driver_db.items():
        if genes:
            rows = []
            for gene, data in genes.items():
                rows.append({
                    'gene': gene,
                    'mutation_freq': data['mutation_freq'],
                    'samples': data['samples'],
                    'total_samples': data['total_samples'],
                    'role': data['role'],
                    'cgc_gene': data['cgc_gene'],
                    'qvalue': data['qvalue']
                })

            df = pd.DataFrame(rows)
            df = df.sort_values('mutation_freq', ascending=False)
            df.to_csv(OUTPUT_DIR / f"{cancer_type}_drivers.csv", index=False)

    print(f"Saved CSV files for {len([ct for ct, g in driver_db.items() if g])} cancer types")

    # Save summary
    with open(OUTPUT_DIR / "driver_db_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'driver_db_summary.json'}")


def main():
    print("=" * 60)
    print("Building Driver Gene Database from IntOGen")
    print("=" * 60)

    # Load data
    df = load_intogen_data()

    # Process
    driver_db = process_intogen_data(df)

    # Summary
    summary = generate_summary(driver_db)
    print(f"\nTotal unique driver genes: {summary['total_genes']}")
    print("\nGenes per cancer type:")
    for ct, info in summary['by_cancer_type'].items():
        if info['gene_count'] > 0:
            top3 = [f"{g[0]}({g[1]*100:.1f}%)" for g in info['top_genes'][:3]]
            print(f"  {ct}: {info['gene_count']} genes - Top: {', '.join(top3)}")

    # Save
    save_database(driver_db, summary)

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
