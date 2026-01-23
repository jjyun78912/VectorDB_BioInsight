"""
Integrated Driver Gene Analysis Agent

Combines RNA-seq expression data with WGS/WES mutation data
to identify TRUE driver genes with multi-omic evidence.

The key insight:
- RNA-seq alone: Can only PREDICT drivers (expression-based)
- WGS/WES alone: Can find mutations, but may miss expression context
- RNA-seq + WGS/WES: Can IDENTIFY drivers with both mutation AND expression evidence

Input:
- From RNA-seq pipeline:
  - deg_significant.csv: Differentially expressed genes
  - hub_genes.csv: Network hub genes
  - integrated_gene_table.csv: Combined RNA-seq annotations

- From Variant pipeline:
  - driver_mutations.csv: Identified driver mutations
  - annotated_variants.csv: All annotated variants
  - mutation_summary.csv: Per-gene mutation summary

Output:
- integrated_drivers.csv: TRUE driver genes with multi-omic evidence
- driver_classification.csv: Classification of all candidates
- expression_mutation_correlation.csv: Expression-mutation relationship
- meta_integrated_driver.json: Execution metadata
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..utils.base_agent import BaseAgent

# DGIdb client for drug validation
try:
    from ..rag.dgidb_client import DGIdbClient
    HAS_DGIDB = True
except ImportError:
    HAS_DGIDB = False
    logging.getLogger(__name__).warning("DGIdb client not available - drug validation disabled")


@dataclass
class IntegratedDriver:
    """Driver gene with integrated multi-omic evidence."""
    gene_symbol: str

    # Classification
    classification: str = ""  # "confirmed_driver", "high_confidence", "candidate", "expression_only", "mutation_only"
    confidence_score: float = 0.0  # 0-100

    # Mutation evidence
    has_mutation: bool = False
    mutation_count: int = 0
    is_hotspot: bool = False
    hotspot_variant: str = ""
    mutation_driver_score: float = 0.0
    mutation_vaf: float = 0.0

    # Expression evidence
    has_expression_change: bool = False
    log2fc: float = 0.0
    padj: float = 1.0
    direction: str = ""  # up, down
    is_hub_gene: bool = False
    hub_score: float = 0.0

    # Gene role
    gene_role: str = ""  # Oncogene, TSG, Unknown
    role_consistent: bool = False  # Expression consistent with role?

    # Database evidence & Validation Status
    cosmic_tier: str = ""
    oncokb_level: str = ""
    tcga_mutation_freq: float = 0.0

    # Validation status (검증 상태)
    db_validated: bool = False  # 외부 DB에서 검증됨
    hotspot_validated: bool = False  # Hotspot이 COSMIC/OncoKB에서 검증됨
    drug_validated: bool = False  # 약물 정보가 DGIdb에서 검증됨
    validation_sources: List[str] = field(default_factory=list)  # 검증 출처 목록
    validation_notes: str = ""  # 검증 관련 추가 정보

    # Pathway context
    pathway_count: int = 0
    top_pathways: List[str] = field(default_factory=list)

    # Evidence summary
    evidence_list: List[str] = field(default_factory=list)

    # Actionability
    is_actionable: bool = False
    actionable_drugs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class IntegratedDriverAgent(BaseAgent):
    """
    Agent for integrated driver gene analysis.

    Combines RNA-seq expression with WGS/WES mutations to identify
    true driver genes with multi-omic evidence.
    """

    # Classification thresholds
    CONFIRMED_DRIVER_THRESHOLD = 80
    HIGH_CONFIDENCE_THRESHOLD = 60
    CANDIDATE_THRESHOLD = 40

    # Gene role definitions
    TSG_GENES = {
        'TP53', 'RB1', 'PTEN', 'APC', 'BRCA1', 'BRCA2', 'CDKN2A', 'NF1', 'NF2',
        'VHL', 'STK11', 'SMAD4', 'ATM', 'CHEK2', 'CDH1', 'ARID1A', 'BAP1',
        'FBXW7', 'MLH1', 'MSH2', 'MSH6', 'PALB2', 'SETD2', 'SMARCA4', 'WT1',
    }

    ONCOGENES = {
        'KRAS', 'NRAS', 'HRAS', 'BRAF', 'PIK3CA', 'EGFR', 'ERBB2', 'MET', 'ALK',
        'ROS1', 'RET', 'FGFR1', 'FGFR2', 'FGFR3', 'KIT', 'PDGFRA', 'ABL1', 'JAK2',
        'MYC', 'MYCN', 'CCND1', 'CDK4', 'CDK6', 'MDM2', 'BCL2', 'CTNNB1', 'IDH1',
        'IDH2', 'FLT3', 'NPM1', 'DNMT3A', 'SF3B1',
    }

    # =========================================================================
    # VALIDATED HOTSPOTS from COSMIC & OncoKB (curated)
    # Source: COSMIC v98, OncoKB (2024-01), ClinVar pathogenic
    # Only include hotspots with strong clinical evidence
    # =========================================================================
    VALIDATED_HOTSPOTS = {
        'KRAS': {
            'G12C': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': ['Sotorasib', 'Adagrasib']},
            'G12D': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': []},
            'G12V': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': []},
            'G12A': {'source': 'COSMIC', 'level': '2', 'drugs': []},
            'G12R': {'source': 'COSMIC', 'level': '2', 'drugs': []},
            'G13D': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': []},
            'Q61H': {'source': 'COSMIC', 'level': '2', 'drugs': []},
            'Q61K': {'source': 'COSMIC', 'level': '2', 'drugs': []},
            'Q61L': {'source': 'COSMIC', 'level': '2', 'drugs': []},
        },
        'BRAF': {
            'V600E': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Vemurafenib', 'Dabrafenib', 'Encorafenib']},
            'V600K': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': ['Dabrafenib', 'Trametinib']},
            'V600D': {'source': 'COSMIC', 'level': '2', 'drugs': []},
        },
        'EGFR': {
            'L858R': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Erlotinib', 'Gefitinib', 'Afatinib', 'Osimertinib']},
            'T790M': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Osimertinib']},
            'C797S': {'source': 'COSMIC/OncoKB', 'level': '2', 'drugs': []},
            'exon19del': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Erlotinib', 'Gefitinib', 'Osimertinib']},
            'exon20ins': {'source': 'COSMIC/OncoKB', 'level': '2', 'drugs': ['Mobocertinib']},
        },
        'PIK3CA': {
            'E542K': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': ['Alpelisib']},
            'E545K': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Alpelisib']},
            'E545Q': {'source': 'COSMIC', 'level': '2', 'drugs': []},
            'H1047R': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Alpelisib']},
            'H1047L': {'source': 'COSMIC', 'level': '2', 'drugs': []},
        },
        'ERBB2': {
            'S310F': {'source': 'COSMIC/OncoKB', 'level': '2', 'drugs': ['Trastuzumab', 'Neratinib']},
            'S310Y': {'source': 'COSMIC', 'level': '3', 'drugs': []},
            'L755S': {'source': 'COSMIC/OncoKB', 'level': '2', 'drugs': ['Neratinib']},
            'V777L': {'source': 'COSMIC/OncoKB', 'level': '2', 'drugs': ['Neratinib']},
        },
        'TP53': {
            'R175H': {'source': 'COSMIC/ClinVar', 'level': '1', 'drugs': []},
            'R248Q': {'source': 'COSMIC/ClinVar', 'level': '1', 'drugs': []},
            'R248W': {'source': 'COSMIC/ClinVar', 'level': '1', 'drugs': []},
            'R273C': {'source': 'COSMIC/ClinVar', 'level': '1', 'drugs': []},
            'R273H': {'source': 'COSMIC/ClinVar', 'level': '1', 'drugs': []},
            'R282W': {'source': 'COSMIC/ClinVar', 'level': '1', 'drugs': []},
        },
        'IDH1': {
            'R132H': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Ivosidenib']},
            'R132C': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': ['Ivosidenib']},
        },
        'IDH2': {
            'R140Q': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Enasidenib']},
            'R172K': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': ['Enasidenib']},
        },
        'NRAS': {
            'G12D': {'source': 'COSMIC', 'level': '2', 'drugs': []},
            'G13R': {'source': 'COSMIC', 'level': '2', 'drugs': []},
            'Q61K': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': []},
            'Q61R': {'source': 'COSMIC/OncoKB', 'level': '1', 'drugs': []},
        },
        'AKT1': {
            'E17K': {'source': 'COSMIC/OncoKB', 'level': '2', 'drugs': ['Capivasertib']},
        },
        'MET': {
            'exon14skip': {'source': 'COSMIC/OncoKB/FDA', 'level': '1', 'drugs': ['Capmatinib', 'Tepotinib']},
        },
    }

    # Known drug-gene interactions (Fallback when DGIdb unavailable)
    ACTIONABLE_TARGETS = {
        'EGFR': ['Erlotinib', 'Gefitinib', 'Osimertinib', 'Afatinib'],
        'BRAF': ['Vemurafenib', 'Dabrafenib', 'Encorafenib'],
        'ALK': ['Crizotinib', 'Alectinib', 'Brigatinib', 'Lorlatinib'],
        'ROS1': ['Crizotinib', 'Entrectinib'],
        'KRAS': ['Sotorasib (G12C)', 'Adagrasib (G12C)'],
        'ERBB2': ['Trastuzumab', 'Pertuzumab', 'T-DM1', 'Tucatinib'],
        'PIK3CA': ['Alpelisib'],
        'BRCA1': ['Olaparib', 'Rucaparib', 'Niraparib'],
        'BRCA2': ['Olaparib', 'Rucaparib', 'Niraparib'],
        'MET': ['Capmatinib', 'Tepotinib', 'Crizotinib'],
        'RET': ['Selpercatinib', 'Pralsetinib'],
        'NTRK1': ['Larotrectinib', 'Entrectinib'],
        'NTRK2': ['Larotrectinib', 'Entrectinib'],
        'NTRK3': ['Larotrectinib', 'Entrectinib'],
        'FGFR2': ['Pemigatinib', 'Infigratinib'],
        'IDH1': ['Ivosidenib'],
        'IDH2': ['Enasidenib'],
        'BCR-ABL1': ['Imatinib', 'Dasatinib', 'Nilotinib', 'Ponatinib'],
        'FLT3': ['Midostaurin', 'Gilteritinib'],
        'KIT': ['Imatinib', 'Sunitinib', 'Regorafenib'],
        'PDGFRA': ['Imatinib', 'Avapritinib'],
    }

    # COSMIC Tier 1 genes (validated cancer genes)
    COSMIC_TIER1 = {
        'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'APC', 'RB1',
        'BRCA1', 'BRCA2', 'MYC', 'ERBB2', 'CDK4', 'MDM2', 'CCND1', 'CDKN2A',
        'ATM', 'AKT1', 'NRAS', 'HRAS', 'FGFR1', 'FGFR2', 'FGFR3', 'MET',
        'ALK', 'ROS1', 'RET', 'KIT', 'PDGFRA', 'ABL1', 'JAK2', 'BCL2',
        'VHL', 'NF1', 'NF2', 'WT1', 'SMAD4', 'CTNNB1', 'IDH1', 'IDH2'
    }

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "cancer_type": "unknown",

            # Integration weights
            "mutation_weight": 0.5,
            "expression_weight": 0.3,
            "network_weight": 0.2,

            # Thresholds
            "min_expression_fc": 1.0,  # |log2FC| > 1
            "max_padj": 0.05,
            "min_mutation_score": 40,

            # Output
            "top_n_drivers": 30,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent_integrated_driver", input_dir, output_dir, merged_config)

        # Data storage
        self.deg_df: Optional[pd.DataFrame] = None
        self.hub_df: Optional[pd.DataFrame] = None
        self.integrated_df: Optional[pd.DataFrame] = None
        self.mutation_df: Optional[pd.DataFrame] = None
        self.driver_mutation_df: Optional[pd.DataFrame] = None
        self.mutation_summary_df: Optional[pd.DataFrame] = None

        self.integrated_drivers: List[IntegratedDriver] = []

    def validate_inputs(self) -> bool:
        """Validate that we have both RNA-seq and mutation data."""
        has_rnaseq = False
        has_mutation = False

        # Check RNA-seq outputs
        if (self.input_dir / "deg_significant.csv").exists():
            has_rnaseq = True
            self.logger.info("Found RNA-seq DEG data")

        # Check mutation outputs
        if (self.input_dir / "driver_mutations.csv").exists() or \
           (self.input_dir / "annotated_variants.csv").exists():
            has_mutation = True
            self.logger.info("Found mutation data")

        if not has_rnaseq:
            self.logger.warning("No RNA-seq data found - expression evidence unavailable")

        if not has_mutation:
            self.logger.warning("No mutation data found - mutation evidence unavailable")

        return has_rnaseq or has_mutation  # Allow partial analysis

    def _load_data(self):
        """Load all input data."""
        # RNA-seq data
        if (self.input_dir / "deg_significant.csv").exists():
            self.deg_df = pd.read_csv(self.input_dir / "deg_significant.csv")
            self.logger.info(f"Loaded DEG data: {len(self.deg_df)} genes")

        if (self.input_dir / "hub_genes.csv").exists():
            self.hub_df = pd.read_csv(self.input_dir / "hub_genes.csv")
            self.logger.info(f"Loaded hub genes: {len(self.hub_df)} genes")

        if (self.input_dir / "integrated_gene_table.csv").exists():
            self.integrated_df = pd.read_csv(self.input_dir / "integrated_gene_table.csv")
            self.logger.info(f"Loaded integrated table: {len(self.integrated_df)} genes")

        # Mutation data
        if (self.input_dir / "driver_mutations.csv").exists():
            self.driver_mutation_df = pd.read_csv(self.input_dir / "driver_mutations.csv")
            self.logger.info(f"Loaded driver mutations: {len(self.driver_mutation_df)} mutations")

        if (self.input_dir / "annotated_variants.csv").exists():
            self.mutation_df = pd.read_csv(self.input_dir / "annotated_variants.csv")
            self.logger.info(f"Loaded all variants: {len(self.mutation_df)} variants")

        if (self.input_dir / "mutation_summary.csv").exists():
            self.mutation_summary_df = pd.read_csv(self.input_dir / "mutation_summary.csv")
            self.logger.info(f"Loaded mutation summary: {len(self.mutation_summary_df)} genes")

        # Initialize DGIdb client for drug validation
        self.dgidb_client = None
        self.dgidb_cache: Dict[str, List[str]] = {}
        if HAS_DGIDB:
            try:
                self.dgidb_client = DGIdbClient(timeout=30)
                self.logger.info("DGIdb client initialized for drug validation")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DGIdb client: {e}")

    def _validate_hotspot(self, gene: str, variant: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate hotspot mutation against COSMIC/OncoKB database.

        Args:
            gene: Gene symbol (e.g., 'KRAS')
            variant: Amino acid change (e.g., 'p.G12C' or 'G12C')

        Returns:
            (is_validated, validation_info) - validation_info contains source, level, drugs
        """
        if not gene or not variant:
            return False, None

        gene = gene.upper()

        # Normalize variant format (remove 'p.' prefix)
        variant_clean = variant.replace('p.', '').strip()

        # Check if gene has validated hotspots
        if gene not in self.VALIDATED_HOTSPOTS:
            return False, None

        gene_hotspots = self.VALIDATED_HOTSPOTS[gene]

        # Try exact match first
        if variant_clean in gene_hotspots:
            return True, gene_hotspots[variant_clean]

        # Try position-based match (e.g., 'G12C' matches 'G12')
        for hotspot_key, hotspot_info in gene_hotspots.items():
            # Match by position (first letters + numbers)
            import re
            position_match = re.match(r'^([A-Z])(\d+)', variant_clean)
            hotspot_match = re.match(r'^([A-Z])(\d+)', hotspot_key)

            if position_match and hotspot_match:
                if (position_match.group(1) == hotspot_match.group(1) and
                    position_match.group(2) == hotspot_match.group(2)):
                    return True, hotspot_info

        return False, None

    def _validate_drugs_dgidb(self, genes: List[str]) -> Dict[str, List[str]]:
        """
        Validate drug-gene interactions via DGIdb API.

        Args:
            genes: List of gene symbols to query

        Returns:
            Dict mapping gene -> list of validated drugs
        """
        result = {}

        # Use cached results if available
        genes_to_query = [g for g in genes if g not in self.dgidb_cache]

        if genes_to_query and self.dgidb_client:
            try:
                self.logger.info(f"Querying DGIdb for {len(genes_to_query)} genes...")
                interactions = self.dgidb_client.get_drug_interactions(genes_to_query)

                for interaction in interactions:
                    gene = interaction.gene_name.upper()
                    if gene not in self.dgidb_cache:
                        self.dgidb_cache[gene] = []
                    if interaction.drug_name not in self.dgidb_cache[gene]:
                        self.dgidb_cache[gene].append(interaction.drug_name)

                # Mark queried genes with no results
                for gene in genes_to_query:
                    if gene.upper() not in self.dgidb_cache:
                        self.dgidb_cache[gene.upper()] = []

                self.logger.info(f"DGIdb validation complete: {len(self.dgidb_cache)} genes cached")

            except Exception as e:
                self.logger.warning(f"DGIdb query failed: {e}")

        # Return results from cache
        for gene in genes:
            gene_upper = gene.upper()
            if gene_upper in self.dgidb_cache:
                result[gene_upper] = self.dgidb_cache[gene_upper]

        return result

    def _validate_gene_cosmic(self, gene: str) -> Tuple[bool, str]:
        """
        Check if gene is in COSMIC Tier 1 (validated cancer gene).

        Returns:
            (is_validated, cosmic_tier)
        """
        gene = gene.upper()
        if gene in self.COSMIC_TIER1:
            return True, "Tier1"
        return False, ""

    def _build_gene_profiles(self) -> Dict[str, Dict]:
        """Build unified gene profiles from all data sources."""
        gene_profiles = {}

        # 1. Add expression data
        if self.deg_df is not None:
            gene_col = 'gene_symbol' if 'gene_symbol' in self.deg_df.columns else 'gene_id'
            for _, row in self.deg_df.iterrows():
                gene = str(row.get(gene_col, '')).upper()
                if not gene or gene.startswith('ENSG'):
                    continue

                gene_profiles[gene] = {
                    'gene_symbol': gene,
                    'has_expression_change': True,
                    'log2fc': float(row.get('log2FC', row.get('log2FoldChange', 0))),
                    'padj': float(row.get('padj', 1)),
                    'direction': str(row.get('direction', 'up' if row.get('log2FC', 0) > 0 else 'down')),
                }

        # 2. Add hub gene info
        if self.hub_df is not None:
            gene_col = 'gene_symbol' if 'gene_symbol' in self.hub_df.columns else 'gene_id'
            for _, row in self.hub_df.iterrows():
                gene = str(row.get(gene_col, '')).upper()
                if not gene or gene.startswith('ENSG'):
                    continue

                if gene not in gene_profiles:
                    gene_profiles[gene] = {'gene_symbol': gene}

                gene_profiles[gene]['is_hub_gene'] = True
                gene_profiles[gene]['hub_score'] = float(row.get('hub_score', row.get('enhanced_hub_score', 0)))

        # 3. Add mutation data
        if self.driver_mutation_df is not None:
            for _, row in self.driver_mutation_df.iterrows():
                gene = str(row.get('gene', '')).upper()
                if not gene:
                    continue

                if gene not in gene_profiles:
                    gene_profiles[gene] = {'gene_symbol': gene}

                profile = gene_profiles[gene]
                profile['has_mutation'] = True
                profile['mutation_count'] = profile.get('mutation_count', 0) + 1
                profile['is_hotspot'] = profile.get('is_hotspot', False) or bool(row.get('is_hotspot', False))
                profile['mutation_driver_score'] = max(
                    profile.get('mutation_driver_score', 0),
                    float(row.get('driver_score', 0))
                )
                profile['mutation_vaf'] = max(
                    profile.get('mutation_vaf', 0),
                    float(row.get('vaf', 0))
                )

                if row.get('is_hotspot'):
                    profile['hotspot_variant'] = str(row.get('amino_acid_change', ''))

        # 4. Add mutation summary for genes without driver mutations
        if self.mutation_summary_df is not None:
            for _, row in self.mutation_summary_df.iterrows():
                gene = str(row.get('gene', '')).upper()
                if not gene:
                    continue

                if gene not in gene_profiles:
                    gene_profiles[gene] = {'gene_symbol': gene}

                profile = gene_profiles[gene]
                if not profile.get('has_mutation'):
                    profile['has_mutation'] = int(row.get('total_mutations', 0)) > 0
                    profile['mutation_count'] = int(row.get('total_mutations', 0))

        self.logger.info(f"Built profiles for {len(gene_profiles)} genes")
        return gene_profiles

    def _get_gene_role(self, gene: str) -> str:
        """Get gene role."""
        if gene in self.TSG_GENES:
            return 'TSG'
        elif gene in self.ONCOGENES:
            return 'Oncogene'
        return 'Unknown'

    def _check_role_consistency(self, gene: str, direction: str) -> bool:
        """Check if expression direction is consistent with gene role."""
        role = self._get_gene_role(gene)
        if role == 'Oncogene':
            return direction == 'up'
        elif role == 'TSG':
            return direction == 'down'
        return True  # Unknown role is always "consistent"

    def _calculate_integrated_score(self, profile: Dict) -> Tuple[float, str, List[str]]:
        """
        Calculate integrated driver score and classification.

        Returns: (score, classification, evidence_list)
        """
        score = 0.0
        evidence = []
        weights = self.config

        gene = profile.get('gene_symbol', '')

        # === MUTATION EVIDENCE (max 50 points) ===
        has_mutation = profile.get('has_mutation', False)

        if has_mutation:
            # Base mutation score
            mut_score = profile.get('mutation_driver_score', 0)
            mutation_points = min(30, mut_score * 0.5)  # Max 30 from driver score
            score += mutation_points

            evidence.append(f"Driver mutation detected (score: {mut_score:.0f})")

            # Hotspot bonus
            if profile.get('is_hotspot'):
                score += 15
                hotspot_var = profile.get('hotspot_variant', '')
                evidence.append(f"Hotspot mutation: {hotspot_var}")

            # High VAF bonus (clonal)
            vaf = profile.get('mutation_vaf', 0)
            if vaf >= 0.3:
                score += 5
                evidence.append(f"High VAF ({vaf:.1%}) - likely clonal")

        # === EXPRESSION EVIDENCE (max 30 points) ===
        has_expr = profile.get('has_expression_change', False)

        if has_expr:
            log2fc = abs(profile.get('log2fc', 0))
            padj = profile.get('padj', 1)
            direction = profile.get('direction', '')

            # Expression magnitude
            if log2fc > 2:
                score += 15
            elif log2fc > 1.5:
                score += 12
            elif log2fc > 1:
                score += 8

            # Statistical significance
            if padj < 0.001:
                score += 10
            elif padj < 0.01:
                score += 7
            elif padj < 0.05:
                score += 4

            evidence.append(f"Differential expression: log2FC={profile.get('log2fc', 0):.2f}, padj={padj:.2e}")

            # Role consistency bonus
            if self._check_role_consistency(gene, direction):
                role = self._get_gene_role(gene)
                if role != 'Unknown':
                    score += 5
                    evidence.append(f"Expression consistent with {role} role ({direction})")

        # === NETWORK EVIDENCE (max 20 points) ===
        if profile.get('is_hub_gene'):
            hub_score = profile.get('hub_score', 0)
            hub_points = min(15, hub_score * 25)
            score += hub_points
            evidence.append(f"Network hub gene (score: {hub_score:.2f})")

        # === MULTI-OMIC BONUS ===
        if has_mutation and has_expr:
            score += 10
            evidence.append("Multi-omic evidence (mutation + expression)")

        # === CLASSIFICATION ===
        if score >= self.CONFIRMED_DRIVER_THRESHOLD:
            if has_mutation and has_expr:
                classification = "confirmed_driver"
            elif has_mutation:
                classification = "high_confidence"
            else:
                classification = "high_confidence"
        elif score >= self.HIGH_CONFIDENCE_THRESHOLD:
            classification = "high_confidence"
        elif score >= self.CANDIDATE_THRESHOLD:
            classification = "candidate"
        elif has_mutation:
            classification = "mutation_only"
        elif has_expr:
            classification = "expression_only"
        else:
            classification = "insufficient_evidence"

        return min(100, score), classification, evidence

    def _analyze_drivers(self):
        """Perform integrated driver analysis with external DB validation."""
        self.logger.info("Performing integrated driver analysis...")

        # Build gene profiles
        gene_profiles = self._build_gene_profiles()

        # Pre-fetch DGIdb data for all candidate genes
        candidate_genes = list(gene_profiles.keys())
        dgidb_drugs = self._validate_drugs_dgidb(candidate_genes)

        # Analyze each gene
        for gene, profile in gene_profiles.items():
            score, classification, evidence = self._calculate_integrated_score(profile)

            # Skip low-scoring genes
            if score < 20:
                continue

            # Get gene role and actionability
            gene_role = self._get_gene_role(gene)
            direction = profile.get('direction', '')
            role_consistent = self._check_role_consistency(gene, direction)

            # ========================================================
            # VALIDATION STEP 1: Validate hotspot against COSMIC/OncoKB
            # ========================================================
            hotspot_validated = False
            validation_sources = []
            validation_notes = ""
            oncokb_level = ""
            cosmic_tier = ""

            hotspot_var = profile.get('hotspot_variant', '')
            if profile.get('is_hotspot') and hotspot_var:
                validated, hotspot_info = self._validate_hotspot(gene, hotspot_var)
                if validated and hotspot_info:
                    hotspot_validated = True
                    validation_sources.append(hotspot_info.get('source', 'COSMIC'))
                    oncokb_level = hotspot_info.get('level', '')
                    evidence.append(f"✓ Hotspot validated: {hotspot_info.get('source', 'DB')} (Level {oncokb_level})")
                else:
                    # Hotspot claimed but NOT in validated DB
                    validation_notes = f"⚠️ Hotspot {hotspot_var} not found in COSMIC/OncoKB validated list"
                    evidence.append(validation_notes)

            # ========================================================
            # VALIDATION STEP 2: Check COSMIC Tier 1 gene
            # ========================================================
            is_cosmic_validated, tier = self._validate_gene_cosmic(gene)
            if is_cosmic_validated:
                cosmic_tier = tier
                if 'COSMIC' not in ' '.join(validation_sources):
                    validation_sources.append(f"COSMIC_{tier}")
                evidence.append(f"✓ COSMIC {tier} cancer gene")

            # ========================================================
            # VALIDATION STEP 3: Validate drugs via DGIdb
            # ========================================================
            drug_validated = False
            actionable_drugs = []

            # First try DGIdb validated drugs
            if gene in dgidb_drugs and dgidb_drugs[gene]:
                drug_validated = True
                actionable_drugs = dgidb_drugs[gene][:5]  # Top 5 drugs
                validation_sources.append("DGIdb")
                evidence.append(f"✓ DGIdb validated drugs: {', '.join(actionable_drugs[:3])}")
            else:
                # Fallback to curated list (but mark as not externally validated)
                if gene in self.ACTIONABLE_TARGETS:
                    actionable_drugs = self.ACTIONABLE_TARGETS[gene]
                    evidence.append(f"△ Curated drugs (not DGIdb validated): {', '.join(actionable_drugs[:3])}")

            # Overall validation status
            db_validated = hotspot_validated or is_cosmic_validated or drug_validated

            # ========================================================
            # ADJUST CLASSIFICATION based on validation
            # ========================================================
            # "confirmed_driver" REQUIRES external DB validation
            if classification == "confirmed_driver" and not db_validated:
                classification = "high_confidence"
                validation_notes += " Classification downgraded: no external DB validation"
                evidence.append("⚠️ Downgraded to high_confidence: external validation required for confirmed_driver")

            is_actionable = len(actionable_drugs) > 0

            driver = IntegratedDriver(
                gene_symbol=gene,
                classification=classification,
                confidence_score=score,

                # Mutation
                has_mutation=profile.get('has_mutation', False),
                mutation_count=profile.get('mutation_count', 0),
                is_hotspot=profile.get('is_hotspot', False),
                hotspot_variant=hotspot_var,
                mutation_driver_score=profile.get('mutation_driver_score', 0),
                mutation_vaf=profile.get('mutation_vaf', 0),

                # Expression
                has_expression_change=profile.get('has_expression_change', False),
                log2fc=profile.get('log2fc', 0),
                padj=profile.get('padj', 1),
                direction=direction,
                is_hub_gene=profile.get('is_hub_gene', False),
                hub_score=profile.get('hub_score', 0),

                # Role
                gene_role=gene_role,
                role_consistent=role_consistent,

                # Database validation
                cosmic_tier=cosmic_tier,
                oncokb_level=oncokb_level,

                # Validation status
                db_validated=db_validated,
                hotspot_validated=hotspot_validated,
                drug_validated=drug_validated,
                validation_sources=validation_sources,
                validation_notes=validation_notes.strip(),

                # Evidence
                evidence_list=evidence,

                # Actionability
                is_actionable=is_actionable,
                actionable_drugs=actionable_drugs,
            )

            self.integrated_drivers.append(driver)

        # Sort by score
        self.integrated_drivers.sort(key=lambda x: x.confidence_score, reverse=True)

        # Log summary with validation stats
        confirmed = len([d for d in self.integrated_drivers if d.classification == 'confirmed_driver'])
        high_conf = len([d for d in self.integrated_drivers if d.classification == 'high_confidence'])
        candidates = len([d for d in self.integrated_drivers if d.classification == 'candidate'])
        actionable = len([d for d in self.integrated_drivers if d.is_actionable])
        validated = len([d for d in self.integrated_drivers if d.db_validated])
        hotspot_val = len([d for d in self.integrated_drivers if d.hotspot_validated])

        self.logger.info(f"  Confirmed drivers: {confirmed}")
        self.logger.info(f"  High confidence: {high_conf}")
        self.logger.info(f"  Candidates: {candidates}")
        self.logger.info(f"  Actionable targets: {actionable}")
        self.logger.info(f"  ✓ DB validated: {validated}")
        self.logger.info(f"  ✓ Hotspot validated: {hotspot_val}")

    def _generate_visualizations(self):
        """Generate integrated driver visualizations."""
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')

            # 1. Classification distribution
            classifications = [d.classification for d in self.integrated_drivers]
            class_counts = {}
            for c in set(classifications):
                class_counts[c] = classifications.count(c)

            if class_counts:
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = {
                    'confirmed_driver': '#e74c3c',
                    'high_confidence': '#f39c12',
                    'candidate': '#3498db',
                    'mutation_only': '#9b59b6',
                    'expression_only': '#1abc9c',
                }
                bars = ax.bar(
                    list(class_counts.keys()),
                    list(class_counts.values()),
                    color=[colors.get(c, '#95a5a6') for c in class_counts.keys()]
                )
                ax.set_xlabel('Classification')
                ax.set_ylabel('Count')
                ax.set_title('Integrated Driver Classification')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(figures_dir / 'driver_classification.png', dpi=150)
                plt.close()

            # 2. Evidence type comparison
            both_evidence = len([d for d in self.integrated_drivers if d.has_mutation and d.has_expression_change])
            mutation_only = len([d for d in self.integrated_drivers if d.has_mutation and not d.has_expression_change])
            expression_only = len([d for d in self.integrated_drivers if not d.has_mutation and d.has_expression_change])

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(
                [both_evidence, mutation_only, expression_only],
                labels=['Mutation + Expression', 'Mutation Only', 'Expression Only'],
                colors=['#e74c3c', '#9b59b6', '#1abc9c'],
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title('Evidence Type Distribution')
            plt.tight_layout()
            plt.savefig(figures_dir / 'evidence_distribution.png', dpi=150)
            plt.close()

            # 3. Top drivers barplot
            top_drivers = self.integrated_drivers[:20]
            if top_drivers:
                fig, ax = plt.subplots(figsize=(12, 8))
                genes = [d.gene_symbol for d in top_drivers]
                scores = [d.confidence_score for d in top_drivers]
                colors = ['#e74c3c' if d.classification == 'confirmed_driver'
                         else '#f39c12' if d.classification == 'high_confidence'
                         else '#3498db' for d in top_drivers]

                bars = ax.barh(genes[::-1], scores[::-1], color=colors[::-1])

                # Add mutation/expression indicators
                for i, d in enumerate(reversed(top_drivers)):
                    indicator = ''
                    if d.has_mutation:
                        indicator += 'M'
                    if d.has_expression_change:
                        indicator += 'E'
                    if d.is_hotspot:
                        indicator += '*'
                    ax.text(scores[-(i+1)] + 1, i, indicator, va='center', fontsize=9)

                ax.set_xlabel('Integrated Driver Score')
                ax.set_title('Top 20 Integrated Drivers\n(M=Mutation, E=Expression, *=Hotspot)')
                ax.set_xlim(0, 110)
                plt.tight_layout()
                plt.savefig(figures_dir / 'top_integrated_drivers.png', dpi=150)
                plt.close()

            self.logger.info("Generated visualizations")

        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualizations")

    def run(self) -> Dict[str, Any]:
        """Run integrated driver analysis."""
        self.logger.info("="*60)
        self.logger.info("Starting Integrated Driver Analysis")
        self.logger.info("(RNA-seq + WGS/WES Integration)")
        self.logger.info("="*60)

        # 1. Load data
        self._load_data()

        # 2. Analyze drivers
        self._analyze_drivers()

        # 3. Generate visualizations
        self._generate_visualizations()

        # 4. Save outputs
        self._save_outputs()

        # Compile results
        confirmed = [d for d in self.integrated_drivers if d.classification == 'confirmed_driver']
        high_conf = [d for d in self.integrated_drivers if d.classification == 'high_confidence']
        actionable = [d for d in self.integrated_drivers if d.is_actionable]

        results = {
            "status": "success",
            "total_analyzed": len(self.integrated_drivers),
            "confirmed_drivers": len(confirmed),
            "high_confidence": len(high_conf),
            "actionable_targets": len(actionable),
            "top_drivers": [d.gene_symbol for d in self.integrated_drivers[:10]],
            "top_confirmed": [d.gene_symbol for d in confirmed[:5]],
            "top_actionable": [d.gene_symbol for d in actionable[:5]],
        }

        self.logger.info("="*60)
        self.logger.info("Integrated Driver Analysis Complete")
        self.logger.info(f"  Total analyzed: {results['total_analyzed']}")
        self.logger.info(f"  Confirmed drivers: {results['confirmed_drivers']}")
        self.logger.info(f"  High confidence: {results['high_confidence']}")
        self.logger.info(f"  Actionable targets: {results['actionable_targets']}")
        self.logger.info("="*60)

        return results

    def _save_outputs(self):
        """Save analysis outputs."""
        self.logger.info("Saving outputs...")

        # All integrated drivers
        all_df = pd.DataFrame([d.to_dict() for d in self.integrated_drivers])
        if len(all_df) > 0:
            # Convert list columns to strings for CSV
            all_df['evidence_list'] = all_df['evidence_list'].apply(lambda x: '; '.join(x) if x else '')
            all_df['top_pathways'] = all_df['top_pathways'].apply(lambda x: '; '.join(x) if x else '')
            all_df['actionable_drugs'] = all_df['actionable_drugs'].apply(lambda x: '; '.join(x) if x else '')
            self.save_csv(all_df, "integrated_drivers.csv")

        # Confirmed drivers only
        confirmed = [d for d in self.integrated_drivers if d.classification == 'confirmed_driver']
        if confirmed:
            confirmed_df = pd.DataFrame([d.to_dict() for d in confirmed])
            confirmed_df['evidence_list'] = confirmed_df['evidence_list'].apply(lambda x: '; '.join(x) if x else '')
            confirmed_df['actionable_drugs'] = confirmed_df['actionable_drugs'].apply(lambda x: '; '.join(x) if x else '')
            self.save_csv(confirmed_df, "confirmed_drivers.csv")

        # Actionable targets
        actionable = [d for d in self.integrated_drivers if d.is_actionable]
        if actionable:
            actionable_df = pd.DataFrame([d.to_dict() for d in actionable])
            actionable_df['actionable_drugs'] = actionable_df['actionable_drugs'].apply(lambda x: '; '.join(x) if x else '')
            self.save_csv(actionable_df, "actionable_targets.csv")

        # Summary JSON
        summary = {
            "cancer_type": self.config.get('cancer_type', 'unknown'),
            "total_genes_analyzed": len(self.integrated_drivers),
            "confirmed_drivers": len([d for d in self.integrated_drivers if d.classification == 'confirmed_driver']),
            "high_confidence": len([d for d in self.integrated_drivers if d.classification == 'high_confidence']),
            "candidates": len([d for d in self.integrated_drivers if d.classification == 'candidate']),
            "with_mutation": len([d for d in self.integrated_drivers if d.has_mutation]),
            "with_expression": len([d for d in self.integrated_drivers if d.has_expression_change]),
            "with_both": len([d for d in self.integrated_drivers if d.has_mutation and d.has_expression_change]),
            "actionable": len([d for d in self.integrated_drivers if d.is_actionable]),
            "top_10_drivers": [
                {
                    "gene": d.gene_symbol,
                    "score": d.confidence_score,
                    "classification": d.classification,
                    "has_mutation": d.has_mutation,
                    "has_expression": d.has_expression_change,
                }
                for d in self.integrated_drivers[:10]
            ],
            "generated_at": datetime.now().isoformat(),
        }

        with open(self.output_dir / "integrated_driver_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"  Saved integrated_drivers.csv ({len(self.integrated_drivers)} genes)")
        self.logger.info(f"  Saved confirmed_drivers.csv ({len(confirmed)} genes)")
        self.logger.info(f"  Saved actionable_targets.csv ({len(actionable)} genes)")

    def validate_outputs(self) -> bool:
        """Validate that required output files were generated."""
        required_files = ["integrated_drivers.csv", "integrated_driver_summary.json"]

        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing required output: {filename}")
                return False

        return True
