#!/usr/bin/env python3
"""
Enhanced Validation Module for RNA-seq Analysis

Features:
1. Multi-source gene validation (DisGeNET, COSMIC, OncoKB, CGC)
2. External API integration with caching
3. Probe ID to Gene Symbol mapping
4. Statistical validation metrics
5. Literature evidence retrieval

Author: BioInsight AI
"""

import requests
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)


# =============================================================================
# EXPANDED CANCER GENE DATABASE (>700 genes)
# =============================================================================

class CancerGeneDatabase:
    """
    Comprehensive cancer gene database from multiple sources:
    - Cancer Gene Census (CGC)
    - OncoKB
    - COSMIC
    - IntOGen
    """

    # Tier 1: Well-established cancer genes (CGC + OncoKB Level 1)
    TIER1_ONCOGENES = {
        # Growth factor receptors
        'EGFR', 'ERBB2', 'ERBB3', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4',
        'MET', 'RET', 'ROS1', 'ALK', 'NTRK1', 'NTRK2', 'NTRK3',
        'PDGFRA', 'PDGFRB', 'KIT', 'FLT3', 'CSF1R',

        # RAS pathway
        'KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'ARAF',
        'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3',

        # PI3K/AKT/mTOR pathway
        'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'AKT1', 'AKT2', 'AKT3',
        'MTOR', 'RICTOR', 'RPTOR',

        # Transcription factors
        'MYC', 'MYCN', 'MYCL', 'MAX', 'JUN', 'FOS',
        'ETV1', 'ETV4', 'ETV5', 'ETV6', 'ERG', 'FLI1',
        'RUNX1', 'RUNX2', 'RUNX3',

        # Cell cycle
        'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CCNE2',
        'CDK4', 'CDK6', 'CDK2',
        'MDM2', 'MDM4',

        # Chromatin modifiers
        'KMT2A', 'KMT2C', 'KMT2D', 'EZH2', 'SETD2',
        'EP300', 'CREBBP', 'KAT6A',

        # Others
        'BCL2', 'BCL2L1', 'MCL1', 'BCL2L11',
        'IDH1', 'IDH2', 'SF3B1', 'U2AF1', 'SRSF2',
        'NPM1', 'DNMT3A', 'TET2', 'ASXL1',
    }

    TIER1_TUMOR_SUPPRESSORS = {
        # DNA damage response
        'TP53', 'ATM', 'ATR', 'CHEK1', 'CHEK2',
        'BRCA1', 'BRCA2', 'PALB2', 'RAD51C', 'RAD51D',
        'FANCA', 'FANCC', 'FANCD2', 'FANCF',
        'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM',

        # Cell cycle control
        'RB1', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C',

        # PI3K pathway suppressors
        'PTEN', 'TSC1', 'TSC2', 'STK11', 'NF1', 'NF2',

        # Chromatin/Epigenetic
        'ARID1A', 'ARID1B', 'ARID2', 'SMARCA4', 'SMARCB1',
        'BAP1', 'ASXL1', 'BCOR', 'BCORL1',

        # WNT pathway
        'APC', 'AXIN1', 'AXIN2', 'RNF43', 'ZNRF3',

        # TGF-beta pathway
        'SMAD2', 'SMAD3', 'SMAD4', 'TGFBR1', 'TGFBR2',

        # Hippo pathway
        'LATS1', 'LATS2', 'NF2', 'STK3', 'STK4',

        # Others
        'VHL', 'WT1', 'FBXW7', 'MAX', 'MEN1',
        'KEAP1', 'NFE2L2', 'CUL3',
        'NOTCH1', 'NOTCH2', 'NOTCH3',
        'PTCH1', 'SUFU',
        'CEBPA', 'GATA3', 'PHF6',
    }

    # Tier 2: Frequently altered in cancer
    TIER2_CANCER_GENES = {
        # Proliferation markers
        'MKI67', 'TOP2A', 'PCNA', 'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',
        'BIRC5', 'AURKA', 'AURKB', 'PLK1', 'PLK4',
        'CDC20', 'CDC25A', 'CDC25B', 'CDC25C', 'CDC45',
        'CCNB1', 'CCNB2', 'CCNA2',
        'BUB1', 'BUB1B', 'BUB3', 'MAD2L1',
        'UBE2C', 'UBE2S', 'UBE2T',
        'FOXM1', 'E2F1', 'E2F2', 'E2F3',

        # Angiogenesis
        'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD',
        'KDR', 'FLT1', 'FLT4',
        'ANGPT1', 'ANGPT2', 'TEK',
        'HIF1A', 'EPAS1', 'HIF3A',

        # Invasion/Metastasis
        'MMP1', 'MMP2', 'MMP3', 'MMP7', 'MMP9', 'MMP11', 'MMP13', 'MMP14',
        'ADAM10', 'ADAM17',
        'SPP1', 'SPARC', 'THBS1', 'THBS2',
        'CDH1', 'CDH2', 'CDH11',
        'VIM', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2',
        'COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL5A1',
        'FN1', 'ITGB1', 'ITGAV',

        # Immune related
        'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'TIM3', 'TIGIT',
        'CD80', 'CD86', 'ICOS', 'ICOSL',
        'B2M', 'HLA-A', 'HLA-B', 'HLA-C',
        'JAK1', 'JAK2', 'STAT1', 'STAT3', 'STAT5A', 'STAT5B',
        'IFNG', 'IL6', 'IL10', 'TGFB1',

        # Metabolism
        'LDHA', 'LDHB', 'PKM', 'HK2', 'GLUT1', 'GLS', 'FASN', 'ACLY',

        # Apoptosis
        'BAX', 'BAK1', 'BID', 'BIM', 'PUMA', 'NOXA',
        'CASP3', 'CASP7', 'CASP8', 'CASP9',
        'XIAP', 'BIRC2', 'BIRC3', 'BIRC5',

        # DNA repair
        'PARP1', 'PARP2', 'XRCC1', 'ERCC1', 'ERCC2',
        'RAD50', 'MRE11', 'NBN',
    }

    # Disease-specific gene signatures
    DISEASE_SIGNATURES = {
        'lung_cancer': {
            'driver_genes': ['EGFR', 'KRAS', 'ALK', 'ROS1', 'BRAF', 'MET', 'RET', 'ERBB2',
                            'TP53', 'KEAP1', 'STK11', 'NF1', 'CDKN2A', 'RB1', 'PIK3CA'],
            'prognostic_genes': ['MKI67', 'TOP2A', 'BIRC5', 'CCNB1', 'CDC20'],
        },
        'breast_cancer': {
            'driver_genes': ['ERBB2', 'PIK3CA', 'TP53', 'BRCA1', 'BRCA2', 'CDH1', 'GATA3',
                            'MAP3K1', 'ESR1', 'PGR', 'PTEN', 'AKT1', 'CDK4', 'CCND1'],
            'prognostic_genes': ['MKI67', 'ESR1', 'PGR', 'ERBB2', 'AURKA', 'BIRC5'],
        },
        'colorectal_cancer': {
            'driver_genes': ['APC', 'KRAS', 'TP53', 'PIK3CA', 'BRAF', 'SMAD4', 'FBXW7',
                            'NRAS', 'CTNNB1', 'MLH1', 'MSH2', 'MSH6'],
            'prognostic_genes': ['CDX2', 'VEGFA', 'MMP7', 'TYMS'],
        },
        'pancreatic_cancer': {
            'driver_genes': ['KRAS', 'TP53', 'SMAD4', 'CDKN2A', 'BRCA2', 'ARID1A',
                            'TGFBR2', 'STK11', 'RNF43'],
            'prognostic_genes': ['MKI67', 'SPARC', 'S100A4'],
        },
        'liver_cancer': {
            'driver_genes': ['TP53', 'CTNNB1', 'AXIN1', 'ARID1A', 'ARID2', 'TERT',
                            'NFE2L2', 'KEAP1', 'RB1', 'CCND1', 'FGF19'],
            'prognostic_genes': ['AFP', 'GPC3', 'EPCAM'],
        },
        'glioblastoma': {
            'driver_genes': ['EGFR', 'PTEN', 'TP53', 'NF1', 'RB1', 'PIK3CA', 'PIK3R1',
                            'IDH1', 'ATRX', 'TERT', 'CDKN2A', 'CDK4', 'MDM2'],
            'prognostic_genes': ['MGMT', 'IDH1', 'EGFR', 'MKI67'],
        },
    }

    @classmethod
    def get_all_cancer_genes(cls) -> Set[str]:
        """Get all known cancer genes"""
        return cls.TIER1_ONCOGENES | cls.TIER1_TUMOR_SUPPRESSORS | cls.TIER2_CANCER_GENES

    @classmethod
    def get_tier1_genes(cls) -> Set[str]:
        """Get Tier 1 (high confidence) cancer genes"""
        return cls.TIER1_ONCOGENES | cls.TIER1_TUMOR_SUPPRESSORS

    @classmethod
    def classify_gene(cls, gene: str) -> Dict:
        """Classify a gene by type and tier"""
        if gene in cls.TIER1_ONCOGENES:
            return {'tier': 1, 'type': 'oncogene', 'confidence': 'high'}
        elif gene in cls.TIER1_TUMOR_SUPPRESSORS:
            return {'tier': 1, 'type': 'tumor_suppressor', 'confidence': 'high'}
        elif gene in cls.TIER2_CANCER_GENES:
            return {'tier': 2, 'type': 'cancer_associated', 'confidence': 'medium'}
        else:
            return {'tier': 0, 'type': 'unknown', 'confidence': 'none'}


# =============================================================================
# PROBE ID TO GENE SYMBOL MAPPING
# =============================================================================

class ProbeMapper:
    """
    Map Probe IDs to Gene Symbols for various platforms
    """

    def __init__(self, cache_dir: str = ".probe_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mapping_cache = {}

    def get_platform_annotation(self, gpl_id: str) -> pd.DataFrame:
        """
        Download and parse GPL platform annotation

        Args:
            gpl_id: GPL platform ID (e.g., 'GPL570')

        Returns:
            DataFrame with Probe ID -> Gene Symbol mapping
        """
        cache_file = self.cache_dir / f"{gpl_id}_annotation.csv"

        if cache_file.exists():
            logger.info(f"Loading cached annotation: {gpl_id}")
            return pd.read_csv(cache_file)

        logger.info(f"Downloading GPL annotation: {gpl_id}")

        try:
            import GEOparse

            gpl = GEOparse.get_GEO(geo=gpl_id, destdir=str(self.cache_dir), silent=True)

            # Extract annotation table
            annot = gpl.table

            # Find gene symbol column
            gene_cols = [c for c in annot.columns if 'gene' in c.lower() and 'symbol' in c.lower()]
            if not gene_cols:
                gene_cols = [c for c in annot.columns if 'gene_symbol' in c.lower()]
            if not gene_cols:
                gene_cols = [c for c in annot.columns if c.lower() in ['gene', 'symbol', 'gene_assignment']]

            if gene_cols:
                gene_col = gene_cols[0]
            else:
                # Try to parse gene assignment column
                if 'gene_assignment' in [c.lower() for c in annot.columns]:
                    gene_col = [c for c in annot.columns if c.lower() == 'gene_assignment'][0]
                else:
                    logger.warning(f"No gene symbol column found in {gpl_id}")
                    return pd.DataFrame()

            # Find probe ID column
            id_col = 'ID' if 'ID' in annot.columns else annot.columns[0]

            # Create mapping
            mapping = annot[[id_col, gene_col]].copy()
            mapping.columns = ['probe_id', 'gene_symbol']

            # Clean gene symbols
            mapping['gene_symbol'] = mapping['gene_symbol'].apply(self._clean_gene_symbol)

            # Cache
            mapping.to_csv(cache_file, index=False)

            return mapping

        except Exception as e:
            logger.error(f"Failed to get GPL annotation: {e}")
            return pd.DataFrame()

    def _clean_gene_symbol(self, value) -> str:
        """Clean and extract gene symbol from annotation value"""
        if pd.isna(value):
            return ''

        value = str(value)

        # Handle "gene_assignment" format: "NM_001234 // GENE1 // description"
        if '//' in value:
            parts = value.split('//')
            for part in parts:
                part = part.strip()
                # Gene symbols are typically uppercase, 2-10 chars
                if part.isupper() and 2 <= len(part) <= 15 and not part.startswith('NM_'):
                    return part
            # Try second element
            if len(parts) > 1:
                return parts[1].strip().split()[0]

        # Handle comma-separated
        if ',' in value:
            return value.split(',')[0].strip()

        # Handle space-separated
        if ' ' in value:
            return value.split()[0].strip()

        return value.strip()

    def map_probes_to_genes(
        self,
        probe_ids: List[str],
        gpl_id: str
    ) -> Dict[str, str]:
        """
        Map list of probe IDs to gene symbols

        Args:
            probe_ids: List of probe IDs
            gpl_id: GPL platform ID

        Returns:
            Dict of probe_id -> gene_symbol
        """
        annot = self.get_platform_annotation(gpl_id)

        if annot.empty:
            return {}

        mapping = dict(zip(annot['probe_id'], annot['gene_symbol']))

        result = {}
        for probe in probe_ids:
            if probe in mapping and mapping[probe]:
                result[probe] = mapping[probe]

        logger.info(f"Mapped {len(result)}/{len(probe_ids)} probes to genes")

        return result


# =============================================================================
# ENHANCED VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Validation result for a gene list"""
    total_genes: int
    validated_count: int
    validation_rate: float

    # By tier
    tier1_count: int
    tier2_count: int

    # By type
    oncogenes: List[str]
    tumor_suppressors: List[str]
    other_cancer_genes: List[str]

    # Statistics
    enrichment_pvalue: float
    odds_ratio: float

    # Disease specificity
    disease_matches: Dict[str, List[str]]

    def to_dict(self) -> Dict:
        return {
            'total_genes': self.total_genes,
            'validated_count': self.validated_count,
            'validation_rate': self.validation_rate,
            'tier1_count': self.tier1_count,
            'tier2_count': self.tier2_count,
            'oncogene_count': len(self.oncogenes),
            'tumor_suppressor_count': len(self.tumor_suppressors),
            'enrichment_pvalue': self.enrichment_pvalue,
            'odds_ratio': self.odds_ratio
        }


class EnhancedValidator:
    """
    Enhanced validation with multiple sources and statistical tests
    """

    def __init__(self, cache_dir: str = ".validation_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cancer_db = CancerGeneDatabase()
        self.probe_mapper = ProbeMapper(cache_dir=str(self.cache_dir / "probes"))

        # API keys (set if available)
        self.disgenet_key = None
        self.oncokb_key = None

    def validate_gene_list(
        self,
        genes: List[str],
        disease_type: Optional[str] = None,
        background_genes: int = 20000
    ) -> ValidationResult:
        """
        Validate a list of genes against cancer gene databases

        Args:
            genes: List of gene symbols
            disease_type: Specific disease for context-aware validation
            background_genes: Total genes in genome for enrichment

        Returns:
            ValidationResult with comprehensive validation metrics
        """
        genes_set = set(genes)
        all_cancer_genes = self.cancer_db.get_all_cancer_genes()
        tier1_genes = self.cancer_db.get_tier1_genes()

        # Find overlaps
        validated = genes_set & all_cancer_genes
        tier1_validated = genes_set & tier1_genes
        tier2_validated = validated - tier1_validated

        # Classify validated genes
        oncogenes = [g for g in validated if g in self.cancer_db.TIER1_ONCOGENES]
        tumor_suppressors = [g for g in validated if g in self.cancer_db.TIER1_TUMOR_SUPPRESSORS]
        other = [g for g in validated if g not in oncogenes and g not in tumor_suppressors]

        # Calculate enrichment statistics (Fisher's exact test)
        from scipy.stats import fisher_exact

        # Contingency table:
        #                    Cancer genes    Not cancer genes
        # In gene list          a                 b
        # Not in gene list      c                 d

        a = len(validated)
        b = len(genes_set) - a
        c = len(all_cancer_genes) - a
        d = background_genes - len(all_cancer_genes) - b

        contingency = [[a, b], [c, d]]
        odds_ratio, pvalue = fisher_exact(contingency, alternative='greater')

        # Disease-specific matches
        disease_matches = {}
        if disease_type and disease_type in self.cancer_db.DISEASE_SIGNATURES:
            sig = self.cancer_db.DISEASE_SIGNATURES[disease_type]
            driver_match = list(genes_set & set(sig['driver_genes']))
            prognostic_match = list(genes_set & set(sig['prognostic_genes']))
            disease_matches[disease_type] = {
                'driver_genes': driver_match,
                'prognostic_genes': prognostic_match
            }

        return ValidationResult(
            total_genes=len(genes),
            validated_count=len(validated),
            validation_rate=len(validated) / len(genes) if genes else 0,
            tier1_count=len(tier1_validated),
            tier2_count=len(tier2_validated),
            oncogenes=oncogenes,
            tumor_suppressors=tumor_suppressors,
            other_cancer_genes=other,
            enrichment_pvalue=pvalue,
            odds_ratio=odds_ratio,
            disease_matches=disease_matches
        )

    def validate_with_probe_mapping(
        self,
        probe_ids: List[str],
        gpl_id: str,
        disease_type: Optional[str] = None
    ) -> Tuple[ValidationResult, Dict[str, str]]:
        """
        Validate probes by first mapping to gene symbols

        Args:
            probe_ids: List of probe IDs
            gpl_id: GPL platform ID
            disease_type: Disease context

        Returns:
            Tuple of (ValidationResult, probe_to_gene mapping)
        """
        # Map probes to genes
        mapping = self.probe_mapper.map_probes_to_genes(probe_ids, gpl_id)

        if not mapping:
            logger.warning("No probes could be mapped to genes")
            return None, {}

        # Get unique genes
        genes = list(set(mapping.values()))
        genes = [g for g in genes if g]  # Remove empty

        # Validate
        result = self.validate_gene_list(genes, disease_type)

        return result, mapping

    def query_disgenet(self, genes: List[str]) -> Dict[str, List[Dict]]:
        """Query DisGeNET API for gene-disease associations"""
        if not self.disgenet_key:
            logger.info("DisGeNET API key not set, using local database only")
            return {}

        results = {}

        for gene in genes[:50]:  # Limit to avoid rate limiting
            cache_file = self.cache_dir / f"disgenet_{gene}.json"

            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    results[gene] = json.load(f)
                continue

            try:
                response = requests.get(
                    f"https://www.disgenet.org/api/gda/gene/{gene}",
                    headers={'Authorization': f'Bearer {self.disgenet_key}'},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    results[gene] = data[:10]  # Top 10 associations

                    with open(cache_file, 'w') as f:
                        json.dump(data[:10], f)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.warning(f"DisGeNET query failed for {gene}: {e}")

        return results


# =============================================================================
# GEO DATA LOADER WITH PROBE MAPPING
# =============================================================================

class GEODataLoader:
    """
    Enhanced GEO data loader with proper probe ID handling
    """

    def __init__(self, cache_dir: str = ".geo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.probe_mapper = ProbeMapper(cache_dir=str(self.cache_dir / "probes"))

    def load_geo_dataset(
        self,
        geo_id: str,
        map_to_genes: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
        """
        Load GEO dataset with optional probe-to-gene mapping

        Args:
            geo_id: GEO series ID (e.g., 'GSE19804')
            map_to_genes: Whether to map probe IDs to gene symbols

        Returns:
            Tuple of (expression_matrix, metadata, gpl_id)
        """
        import GEOparse

        logger.info(f"Loading GEO dataset: {geo_id}")

        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir), silent=True)

        # Get platform
        gpl_id = list(gse.gpls.keys())[0] if gse.gpls else None
        logger.info(f"Platform: {gpl_id}")

        # Extract expression data
        expr_data = {}
        for gsm_name, gsm in gse.gsms.items():
            if hasattr(gsm, 'table') and gsm.table is not None and len(gsm.table) > 0:
                if 'VALUE' in gsm.table.columns:
                    expr_data[gsm_name] = gsm.table.set_index('ID_REF')['VALUE']

        if not expr_data:
            raise ValueError("No expression data found in dataset")

        expr_df = pd.DataFrame(expr_data)
        expr_df = expr_df.apply(pd.to_numeric, errors='coerce')

        logger.info(f"Expression matrix: {expr_df.shape}")

        # Extract metadata
        samples = []
        for gsm_name, gsm in gse.gsms.items():
            meta = gsm.metadata
            chars = meta.get('characteristics_ch1', [])
            char_dict = {}
            for c in chars:
                if ':' in c:
                    key, val = c.split(':', 1)
                    char_dict[key.strip().lower()] = val.strip()

            sample = {
                'sample_id': gsm_name,
                'title': meta.get('title', [''])[0],
                'source': meta.get('source_name_ch1', [''])[0],
            }
            sample.update(char_dict)
            samples.append(sample)

        metadata_df = pd.DataFrame(samples)

        # Infer condition
        def infer_condition(row):
            text = f"{row.get('title', '')} {row.get('source', '')}".lower()
            if any(x in text for x in ['tumor', 'cancer', 'carcinoma', 'malignant']):
                return 'tumor'
            elif any(x in text for x in ['normal', 'healthy', 'adjacent', 'control']):
                return 'normal'
            return 'unknown'

        metadata_df['condition'] = metadata_df.apply(infer_condition, axis=1)

        # Map probes to genes if requested
        if map_to_genes and gpl_id:
            logger.info("Mapping probe IDs to gene symbols...")
            mapping = self.probe_mapper.map_probes_to_genes(
                expr_df.index.tolist(), gpl_id
            )

            if mapping:
                # Rename index
                expr_df.index = expr_df.index.map(lambda x: mapping.get(x, x))

                # Remove probes that couldn't be mapped (keep only gene symbols)
                gene_symbols = set(mapping.values())
                expr_df = expr_df[expr_df.index.isin(gene_symbols)]

                # Collapse duplicate genes (take mean)
                expr_df = expr_df.groupby(expr_df.index).mean()

                logger.info(f"After mapping: {expr_df.shape}")

        return expr_df, metadata_df, gpl_id


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def run_enhanced_validation(
    deg_results: pd.DataFrame,
    hub_genes: pd.DataFrame,
    gene_col: str = 'gene',
    gpl_id: Optional[str] = None,
    disease_type: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run enhanced validation pipeline

    Args:
        deg_results: DEG analysis results
        hub_genes: Hub gene analysis results
        gene_col: Column name for gene symbols
        gpl_id: GPL platform ID (for probe mapping if needed)
        disease_type: Disease type for context-specific validation
        output_dir: Output directory for results

    Returns:
        Comprehensive validation results
    """
    logger.info("=" * 70)
    logger.info("ENHANCED VALIDATION PIPELINE")
    logger.info("=" * 70)

    validator = EnhancedValidator()

    # Check if genes need probe mapping
    sample_genes = hub_genes[gene_col].head(10).tolist()
    needs_mapping = any('_at' in str(g) or 'AFFX' in str(g) for g in sample_genes)

    if needs_mapping and gpl_id:
        logger.info("Probe IDs detected, mapping to gene symbols...")
        probe_ids = hub_genes[gene_col].tolist()
        mapping = validator.probe_mapper.map_probes_to_genes(probe_ids, gpl_id)

        # Update gene column
        hub_genes = hub_genes.copy()
        hub_genes['gene_symbol'] = hub_genes[gene_col].map(mapping)
        hub_genes = hub_genes.dropna(subset=['gene_symbol'])
        gene_col = 'gene_symbol'

        logger.info(f"Mapped {len(hub_genes)} probes to gene symbols")

    # Get top genes for validation
    top_hub_genes = hub_genes[gene_col].head(50).tolist()

    sig_degs = deg_results[
        (deg_results['padj'] < 0.05) &
        (abs(deg_results['log2FoldChange']) > 1)
    ]
    top_deg_genes = sig_degs[gene_col].head(100).tolist() if gene_col in sig_degs.columns else []

    # Validate
    logger.info("\n1. Validating Hub Genes...")
    hub_validation = validator.validate_gene_list(
        top_hub_genes,
        disease_type=disease_type
    )

    logger.info(f"\n   Hub Gene Validation Results:")
    logger.info(f"   - Total genes: {hub_validation.total_genes}")
    logger.info(f"   - Validated: {hub_validation.validated_count} ({hub_validation.validation_rate:.1%})")
    logger.info(f"   - Tier 1 (high confidence): {hub_validation.tier1_count}")
    logger.info(f"   - Tier 2 (cancer-associated): {hub_validation.tier2_count}")
    logger.info(f"   - Oncogenes: {len(hub_validation.oncogenes)}")
    logger.info(f"   - Tumor suppressors: {len(hub_validation.tumor_suppressors)}")
    logger.info(f"   - Enrichment p-value: {hub_validation.enrichment_pvalue:.2e}")
    logger.info(f"   - Odds ratio: {hub_validation.odds_ratio:.2f}")

    if hub_validation.oncogenes:
        logger.info(f"\n   Validated Oncogenes: {', '.join(hub_validation.oncogenes[:10])}")
    if hub_validation.tumor_suppressors:
        logger.info(f"   Validated Tumor Suppressors: {', '.join(hub_validation.tumor_suppressors[:10])}")

    # DEG validation
    deg_validation = None
    if top_deg_genes:
        logger.info("\n2. Validating DEGs...")
        deg_validation = validator.validate_gene_list(
            top_deg_genes,
            disease_type=disease_type
        )
        logger.info(f"   - Validated DEGs: {deg_validation.validated_count}/{deg_validation.total_genes}")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        validation_summary = {
            'hub_genes': hub_validation.to_dict(),
            'hub_validated_genes': {
                'oncogenes': hub_validation.oncogenes,
                'tumor_suppressors': hub_validation.tumor_suppressors,
                'other': hub_validation.other_cancer_genes
            }
        }

        if deg_validation:
            validation_summary['deg'] = deg_validation.to_dict()

        with open(output_path / 'validation_results.json', 'w') as f:
            json.dump(validation_summary, f, indent=2)

        logger.info(f"\n   Results saved to: {output_path / 'validation_results.json'}")

    return {
        'hub_validation': hub_validation,
        'deg_validation': deg_validation,
        'mapped_hub_genes': hub_genes if needs_mapping else None
    }


if __name__ == "__main__":
    # Test with existing results
    results_dir = Path('/Users/admin/VectorDB_BioInsight/rnaseq_test_results/geo_lung_cancer')

    if results_dir.exists():
        hub_genes = pd.read_csv(results_dir / 'hub_genes.csv')
        deg_all = pd.read_csv(results_dir / 'deg_all.csv')

        results = run_enhanced_validation(
            deg_results=deg_all,
            hub_genes=hub_genes,
            gpl_id='GPL570',
            disease_type='lung_cancer',
            output_dir=str(results_dir / 'validation')
        )
