"""
Driver Gene Prediction Module.

Predicts potential driver genes from RNA-seq data using:
1. Known Driver Track: COSMIC/OncoKB + TCGA mutation frequency
2. Novel Driver Track: Hub genes + expression patterns + cancer-specificity

Output:
- Known Driver candidates with validation confidence
- Novel Driver candidates with discovery potential
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from functools import lru_cache

# Gene function lookup
try:
    import mygene
    HAS_MYGENE = True
except ImportError:
    HAS_MYGENE = False

# Translation service for Korean output
try:
    from backend.app.core.translator import TranslationService
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "driver_db"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DriverCandidate:
    """Driver gene candidate with evidence."""
    gene_symbol: str
    track: str  # "known" or "candidate_regulator" (renamed from "novel")
    score: float  # 0-100

    # Expression evidence
    log2fc: float = 0.0
    padj: float = 1.0
    direction: str = "unchanged"

    # Network evidence
    hub_score: float = 0.0
    is_hub: bool = False

    # Database evidence (Known track)
    cosmic_tier: Optional[str] = None
    cosmic_role: Optional[str] = None  # oncogene, TSG
    oncokb_level: Optional[str] = None
    tcga_mutation_freq: float = 0.0
    tcga_sample_count: int = 0
    hotspots: List[str] = None

    # Candidate Regulator track evidence (formerly Novel)
    cancer_specific: bool = False
    pathway_impact: float = 0.0
    literature_count: int = 0

    # Literature support level
    # - "well_established": Multiple studies in this cancer type
    # - "emerging": Some reports in literature
    # - "uncharacterized": No significant literature support
    literature_support: str = "uncharacterized"

    # Pan-cancer driver status
    is_pancancer_driver: bool = False
    pancancer_cancers: List[str] = None  # Cancer types where this is a known driver

    # Role-expression consistency
    # True if Oncogene+up or TSG+down, False if opposite
    role_expression_consistent: Optional[bool] = None

    # Validation suggestion
    validation_method: str = ""
    validation_detail: str = ""

    # Gene function description (from mygene/NCBI)
    gene_function: str = ""

    def __post_init__(self):
        if self.hotspots is None:
            self.hotspots = []
        if self.pancancer_cancers is None:
            self.pancancer_cancers = []

    def to_dict(self) -> Dict:
        return asdict(self)


class DriverDatabase:
    """
    Database manager for driver gene information.
    Sources: COSMIC Cancer Gene Census, OncoKB, TCGA MC3 mutations
    """

    # COSMIC Cancer Gene Census Tier 1 genes (curated list)
    COSMIC_TIER1_ONCOGENES = {
        'KRAS', 'NRAS', 'HRAS', 'BRAF', 'PIK3CA', 'EGFR', 'ERBB2', 'MET',
        'ALK', 'ROS1', 'RET', 'FGFR1', 'FGFR2', 'FGFR3', 'KIT', 'PDGFRA',
        'ABL1', 'JAK2', 'MPL', 'CALR', 'NPM1', 'FLT3', 'IDH1', 'IDH2',
        'MYC', 'MYCN', 'CCND1', 'CDK4', 'CDK6', 'MDM2', 'BCL2', 'MCL1',
        'CTNNB1', 'NOTCH1', 'NOTCH2', 'SMO', 'GLI1', 'GLI2'
    }

    COSMIC_TIER1_TSG = {
        'TP53', 'RB1', 'CDKN2A', 'CDKN2B', 'PTEN', 'APC', 'VHL', 'NF1',
        'NF2', 'BRCA1', 'BRCA2', 'ATM', 'CHEK2', 'PALB2', 'MLH1', 'MSH2',
        'MSH6', 'PMS2', 'STK11', 'SMAD4', 'FBXW7', 'ARID1A', 'ARID2',
        'SMARCA4', 'SMARCB1', 'BAP1', 'WT1', 'SETD2', 'KDM6A', 'KMT2D',
        'CREBBP', 'EP300', 'CYLD', 'PTCH1', 'SUFU'
    }

    COSMIC_TIER2 = {
        'GNAS', 'SF3B1', 'U2AF1', 'SRSF2', 'ZRSR2', 'EZH2', 'DNMT3A',
        'TET2', 'ASXL1', 'BCOR', 'STAG2', 'RAD21', 'SMC1A', 'SMC3',
        'PHF6', 'CEBPA', 'RUNX1', 'GATA2', 'ETV6', 'IKZF1', 'PAX5',
        'BTK', 'CARD11', 'MYD88', 'CD79A', 'CD79B', 'TNFAIP3'
    }

    # Emerging regulators reported in literature (TSG-like or regulatory roles)
    # These are NOT confirmed drivers but have literature support
    EMERGING_REGULATORS = {
        # lncRNAs with tumor suppressor-like activity
        'ADAMTS9-AS2': {'role': 'TSG-like lncRNA', 'cancers': ['breast', 'TNBC', 'lung']},
        'MAGI2-AS3': {'role': 'TSG-like lncRNA', 'cancers': ['breast', 'gastric']},
        'LINC00052': {'role': 'Oncogenic lncRNA', 'cancers': ['breast', 'hepatocellular']},

        # EMT/metastasis regulators
        'RBMS3': {'role': 'TSG', 'cancers': ['breast', 'gastric', 'lung']},
        'SPARCL1': {'role': 'TSG', 'cancers': ['breast', 'colorectal', 'prostate']},

        # Stemness/pluripotency factors
        'SOX2': {'role': 'Oncogene/Stemness', 'cancers': ['pan-cancer', 'breast CSC']},

        # Signaling adaptors/kinases
        'NTRK2': {'role': 'Oncogene', 'cancers': ['neuroblastoma', 'breast']},

        # Other emerging candidates with literature support
        'ANK2': {'role': 'Unknown', 'cancers': ['cardiac', 'emerging cancer']},
        'RAB26': {'role': 'Unknown', 'cancers': ['autophagy', 'emerging cancer']},
        'SHE': {'role': 'Adaptor', 'cancers': ['signaling', 'breast']},
        'SCN4B': {'role': 'Ion channel', 'cancers': ['breast', 'prostate']},
        'TSHZ2': {'role': 'TF', 'cancers': ['breast', 'lung']},
        'LDB2': {'role': 'TF cofactor', 'cancers': ['endothelial', 'breast']},
        'PROS1': {'role': 'Anticoagulant', 'cancers': ['breast', 'ovarian']},
        'JAM2': {'role': 'Cell adhesion', 'cancers': ['breast', 'endothelial']},
        'FLRT2': {'role': 'Cell adhesion', 'cancers': ['breast', 'neuronal']},
        'CYRIA': {'role': 'Actin regulator', 'cancers': ['breast', 'migration']},
        'RHOJ': {'role': 'Rho GTPase', 'cancers': ['breast', 'angiogenesis']},
        'STARD9': {'role': 'Lipid transport', 'cancers': ['breast', 'mitotic']},
    }

    # TCGA cancer type mapping
    TCGA_CANCER_TYPES = {
        'lung_cancer': ['LUAD', 'LUSC'],
        'breast_cancer': ['BRCA'],
        'colorectal_cancer': ['COAD', 'READ'],
        'pancreatic_cancer': ['PAAD'],
        'liver_cancer': ['LIHC'],
        'glioblastoma': ['GBM'],
        'blood_cancer': ['LAML', 'DLBC'],
        'kidney_cancer': ['KIRC', 'KIRP', 'KICH'],
        'prostate_cancer': ['PRAD'],
        'ovarian_cancer': ['OV'],
        'stomach_cancer': ['STAD'],
        'bladder_cancer': ['BLCA'],
        'thyroid_cancer': ['THCA'],
        'melanoma': ['SKCM'],
        'head_neck_cancer': ['HNSC'],
        'uterine_cancer': ['UCEC'],
        'low_grade_glioma': ['LGG']
    }

    # Pre-computed TCGA mutation frequencies (top drivers per cancer)
    # Format: {cancer_type: {gene: (frequency, sample_count, hotspots)}}
    TCGA_MUTATION_FREQ = {
        'lung_cancer': {
            'TP53': (0.46, 566, ['R175H', 'R248Q', 'R273C']),
            'KRAS': (0.32, 394, ['G12C', 'G12V', 'G12D']),
            'EGFR': (0.14, 172, ['L858R', 'T790M', 'exon19del']),
            'STK11': (0.17, 209, ['frameshift', 'nonsense']),
            'KEAP1': (0.19, 234, ['missense']),
            'NF1': (0.11, 135, ['frameshift', 'nonsense']),
            'BRAF': (0.07, 86, ['V600E']),
            'PIK3CA': (0.07, 86, ['E545K', 'H1047R']),
            'RB1': (0.07, 86, ['frameshift', 'nonsense']),
            'CDKN2A': (0.12, 148, ['deletion', 'nonsense']),
        },
        'breast_cancer': {
            'PIK3CA': (0.34, 367, ['H1047R', 'E545K', 'E542K']),
            'TP53': (0.33, 357, ['R175H', 'R248Q', 'Y220C']),
            'CDH1': (0.11, 119, ['frameshift', 'nonsense']),
            'GATA3': (0.10, 108, ['frameshift']),
            'MAP3K1': (0.08, 86, ['frameshift', 'nonsense']),
            'KMT2C': (0.08, 86, ['frameshift']),
            'PTEN': (0.04, 43, ['frameshift', 'nonsense']),
            'AKT1': (0.04, 43, ['E17K']),
            'ERBB2': (0.02, 22, ['amplification']),
            'BRCA1': (0.03, 32, ['frameshift', 'nonsense']),
        },
        'colorectal_cancer': {
            'APC': (0.76, 435, ['frameshift', 'nonsense']),
            'TP53': (0.54, 309, ['R175H', 'R248W', 'R273H']),
            'KRAS': (0.43, 246, ['G12D', 'G12V', 'G13D']),
            'PIK3CA': (0.18, 103, ['E545K', 'H1047R']),
            'SMAD4': (0.14, 80, ['frameshift', 'missense']),
            'FBXW7': (0.11, 63, ['R465C', 'R505C']),
            'TCF7L2': (0.10, 57, ['frameshift']),
            'NRAS': (0.09, 51, ['Q61K', 'Q61R']),
            'BRAF': (0.10, 57, ['V600E']),
            'SOX9': (0.09, 51, ['frameshift']),
        },
        'pancreatic_cancer': {
            'KRAS': (0.93, 168, ['G12D', 'G12V', 'G12R']),
            'TP53': (0.72, 130, ['R175H', 'R248W', 'R273H']),
            'CDKN2A': (0.30, 54, ['deletion', 'frameshift']),
            'SMAD4': (0.26, 47, ['frameshift', 'missense']),
            'BRCA2': (0.05, 9, ['frameshift']),
            'ARID1A': (0.06, 11, ['frameshift', 'nonsense']),
            'RNF43': (0.05, 9, ['frameshift']),
            'TGFBR2': (0.04, 7, ['frameshift']),
            'GNAS': (0.06, 11, ['R201H', 'R201C']),
            'STK11': (0.04, 7, ['frameshift', 'nonsense']),
        },
        'liver_cancer': {
            'TP53': (0.31, 116, ['R249S', 'R175H']),
            'CTNNB1': (0.26, 97, ['S45P', 'D32G', 'S33C']),
            'AXIN1': (0.08, 30, ['frameshift', 'nonsense']),
            'ARID1A': (0.07, 26, ['frameshift']),
            'ARID2': (0.05, 19, ['frameshift']),
            'ALB': (0.05, 19, ['missense']),
            'APOB': (0.10, 37, ['missense']),
            'RB1': (0.04, 15, ['frameshift', 'nonsense']),
            'CDKN2A': (0.03, 11, ['deletion']),
            'NFE2L2': (0.03, 11, ['D29H', 'R34Q']),
        },
        'glioblastoma': {
            'PTEN': (0.31, 96, ['frameshift', 'nonsense']),
            'TP53': (0.28, 87, ['R175H', 'R248Q', 'R273C']),
            'EGFR': (0.27, 84, ['A289V', 'R108K', 'EGFRvIII']),
            'NF1': (0.10, 31, ['frameshift', 'nonsense']),
            'PIK3CA': (0.09, 28, ['E545K', 'H1047R']),
            'PIK3R1': (0.09, 28, ['frameshift']),
            'RB1': (0.08, 25, ['frameshift', 'nonsense']),
            'IDH1': (0.06, 19, ['R132H']),
            'ATRX': (0.07, 22, ['frameshift', 'nonsense']),
            'PDGFRA': (0.05, 16, ['amplification']),
        },
    }

    def __init__(self, cancer_type: str = "unknown"):
        self.cancer_type = cancer_type
        self.intogen_db = {}  # IntOGen driver database
        self._load_databases()

    def _load_databases(self):
        """Load or create driver gene databases."""
        # Combined COSMIC genes
        self.cosmic_genes = (
            self.COSMIC_TIER1_ONCOGENES |
            self.COSMIC_TIER1_TSG |
            self.COSMIC_TIER2
        )

        # Gene roles from COSMIC
        self.gene_roles = {}
        for gene in self.COSMIC_TIER1_ONCOGENES:
            self.gene_roles[gene] = {'tier': 'Tier1', 'role': 'Oncogene'}
        for gene in self.COSMIC_TIER1_TSG:
            self.gene_roles[gene] = {'tier': 'Tier1', 'role': 'TSG'}
        for gene in self.COSMIC_TIER2:
            self.gene_roles[gene] = {'tier': 'Tier2', 'role': 'Unknown'}

        # Load IntOGen database (pan-cancer driver data)
        self._load_intogen_database()

    def _load_intogen_database(self):
        """Load IntOGen driver database from JSON file."""
        intogen_path = DATA_DIR.parent / "intogen_driver_db.json"

        if not intogen_path.exists():
            # Try alternative path
            intogen_path = PROJECT_ROOT / "data" / "driver_db" / "intogen_driver_db.json"

        if intogen_path.exists():
            try:
                with open(intogen_path, 'r') as f:
                    self.intogen_db = json.load(f)
                print(f"✅ Loaded IntOGen database: {sum(len(g) for g in self.intogen_db.values())} driver entries across {len(self.intogen_db)} cancer types")

                # Also update gene roles from IntOGen
                for cancer_type, genes in self.intogen_db.items():
                    for gene, info in genes.items():
                        gene_upper = gene.upper()
                        if gene_upper not in self.gene_roles:
                            role = info.get('role', 'Unknown')
                            tier = 'IntOGen' if info.get('cgc_gene') else 'IntOGen-Novel'
                            self.gene_roles[gene_upper] = {'tier': tier, 'role': role}

                        # Add IntOGen genes to cosmic_genes set for is_known_driver check
                        if info.get('cgc_gene'):
                            self.cosmic_genes.add(gene_upper)

            except Exception as e:
                print(f"⚠️ Error loading IntOGen database: {e}")
        else:
            print(f"⚠️ IntOGen database not found at {intogen_path}")
            print("   Run: python scripts/build_driver_database.py")

    def get_cosmic_info(self, gene: str) -> Optional[Dict]:
        """Get COSMIC Cancer Gene Census info."""
        gene_upper = gene.upper()
        if gene_upper in self.gene_roles:
            return self.gene_roles[gene_upper]
        return None

    def get_tcga_mutation_freq(self, gene: str, cancer_type: str = None) -> Tuple[float, int, List[str]]:
        """
        Get mutation frequency for a gene in specific cancer type.
        Uses IntOGen database first, then falls back to hardcoded TCGA data.
        Returns: (frequency, sample_count, hotspots)
        """
        cancer = cancer_type or self.cancer_type
        gene_upper = gene.upper()

        # 1. First check IntOGen database (comprehensive pan-cancer data)
        if cancer in self.intogen_db:
            if gene_upper in self.intogen_db[cancer]:
                info = self.intogen_db[cancer][gene_upper]
                return (info['mutation_freq'], info['samples'], [])

        # 2. Fallback to hardcoded TCGA data (with hotspot info)
        if cancer in self.TCGA_MUTATION_FREQ:
            if gene_upper in self.TCGA_MUTATION_FREQ[cancer]:
                return self.TCGA_MUTATION_FREQ[cancer][gene_upper]

        # 3. Check if gene is in COSMIC but not in this cancer's data
        if gene_upper in self.cosmic_genes:
            return (0.01, 0, [])  # Low frequency placeholder

        return (0.0, 0, [])

    def is_known_driver(self, gene: str) -> bool:
        """Check if gene is a known driver (COSMIC or IntOGen CGC gene)."""
        gene_upper = gene.upper()

        # Check COSMIC genes
        if gene_upper in self.cosmic_genes:
            return True

        # Check IntOGen CGC genes for current cancer type
        cancer = self.cancer_type
        if cancer in self.intogen_db:
            if gene_upper in self.intogen_db[cancer]:
                return self.intogen_db[cancer][gene_upper].get('cgc_gene', False)

        return False

    def is_intogen_driver(self, gene: str, cancer_type: str = None) -> bool:
        """Check if gene is a driver in IntOGen for specific cancer type."""
        cancer = cancer_type or self.cancer_type
        gene_upper = gene.upper()

        if cancer in self.intogen_db:
            return gene_upper in self.intogen_db[cancer]
        return False

    def get_pancancer_driver_status(self, gene: str) -> Tuple[bool, List[str]]:
        """
        Check if gene is a pan-cancer driver (driver in multiple cancer types).

        Returns: (is_pancancer_driver, list of cancer types where it's a driver)
        """
        gene_upper = gene.upper()
        driver_cancers = []

        # Check IntOGen database for all cancer types
        for cancer_type, genes in self.intogen_db.items():
            if gene_upper in genes:
                driver_cancers.append(cancer_type)

        # Check COSMIC Tier1 (always pan-cancer relevant)
        if gene_upper in self.COSMIC_TIER1_ONCOGENES | self.COSMIC_TIER1_TSG:
            if 'pan-cancer' not in driver_cancers:
                driver_cancers.append('pan-cancer (COSMIC Tier1)')

        is_pancancer = len(driver_cancers) >= 2  # Driver in 2+ cancer types
        return is_pancancer, driver_cancers

    def get_literature_support(self, gene: str, cancer_type: str = None) -> str:
        """
        Assess literature support level for a gene in the given cancer context.

        Returns:
        - "well_established": Known driver in COSMIC/OncoKB + IntOGen for this cancer
        - "emerging": In IntOGen, pan-cancer driver, or curated emerging regulator list
        - "uncharacterized": No significant database support

        Note: This is a heuristic based on database presence.
        For true literature search, integrate PubMed API.
        """
        gene_upper = gene.upper()
        cancer = cancer_type or self.cancer_type

        # Well-established: COSMIC Tier1 + IntOGen driver in this cancer
        is_cosmic_tier1 = gene_upper in (self.COSMIC_TIER1_ONCOGENES | self.COSMIC_TIER1_TSG)
        is_intogen_this_cancer = self.is_intogen_driver(gene_upper, cancer)

        if is_cosmic_tier1 and is_intogen_this_cancer:
            return "well_established"

        # COSMIC Tier1 alone is also well-established
        if is_cosmic_tier1:
            return "well_established"

        # Check pan-cancer status
        is_pancancer, driver_cancers = self.get_pancancer_driver_status(gene_upper)

        # Emerging: IntOGen driver (this cancer) OR pan-cancer driver OR COSMIC Tier2
        if is_intogen_this_cancer:
            return "emerging"
        if is_pancancer:
            return "emerging"
        if gene_upper in self.COSMIC_TIER2:
            return "emerging"

        # Check if it's a known gene in any database
        if gene_upper in self.gene_roles:
            return "emerging"

        # Check curated EMERGING_REGULATORS list (literature-based)
        if gene_upper in self.EMERGING_REGULATORS:
            return "emerging"

        return "uncharacterized"

    def get_emerging_regulator_info(self, gene: str) -> Optional[Dict]:
        """Get info about an emerging regulator from curated list."""
        gene_upper = gene.upper()
        if gene_upper in self.EMERGING_REGULATORS:
            return self.EMERGING_REGULATORS[gene_upper]
        return None

    def check_role_expression_consistency(self, gene: str, direction: str) -> Optional[bool]:
        """
        Check if expression direction is consistent with known role.

        - Oncogene: expect upregulated in cancer (direction='up' is consistent)
        - TSG: expect downregulated in cancer (direction='down' is consistent)

        Returns:
        - True: Consistent (Oncogene+up or TSG+down)
        - False: Inconsistent (Oncogene+down or TSG+up) - may indicate complex biology
        - None: Role unknown
        """
        gene_upper = gene.upper()

        role_info = self.gene_roles.get(gene_upper)
        if not role_info:
            return None

        role = role_info.get('role', 'Unknown')

        if role == 'Oncogene':
            return direction == 'up'
        elif role == 'TSG':
            return direction == 'down'
        else:
            return None

    def get_validation_suggestion(self, gene: str, cancer_type: str = None) -> Tuple[str, str]:
        """Get validation method suggestion for a gene."""
        gene_upper = gene.upper()
        freq, count, hotspots = self.get_tcga_mutation_freq(gene_upper, cancer_type)

        if hotspots:
            hotspot_str = ', '.join(hotspots[:3])
            if 'G12' in hotspot_str or 'G13' in hotspot_str:
                return ('PCR/Sanger', f'{gene_upper} codon 12/13 hotspot sequencing')
            elif 'V600' in hotspot_str:
                return ('PCR/Sanger', f'{gene_upper} V600 mutation detection')
            elif 'exon' in hotspot_str.lower():
                return ('Targeted NGS', f'{gene_upper} exon sequencing')
            else:
                return ('Targeted NGS', f'{gene_upper} hotspot panel ({hotspot_str})')

        # Default suggestions by gene role
        cosmic_info = self.get_cosmic_info(gene_upper)
        if cosmic_info:
            if cosmic_info['role'] == 'TSG':
                return ('Targeted NGS', f'{gene_upper} full gene sequencing (TSG)')
            else:
                return ('Targeted NGS', f'{gene_upper} kinase domain / hotspot panel')

        return ('WES/RNA-seq validation', f'{gene_upper} expression-mutation correlation')


class DriverPredictor:
    """
    Main driver gene prediction engine.

    Analyzes DEG results and network analysis to predict:
    1. Known Drivers: High confidence, validated targets
    2. Novel Drivers: Discovery candidates for further research
    """

    def __init__(self, cancer_type: str = "unknown"):
        self.cancer_type = cancer_type
        self.db = DriverDatabase(cancer_type)
        self.results = {
            'known_drivers': [],
            'novel_drivers': [],
            'summary': {}
        }

    def predict(
        self,
        deg_df: pd.DataFrame,
        hub_genes_df: Optional[pd.DataFrame] = None,
        integrated_df: Optional[pd.DataFrame] = None,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Predict driver genes from analysis results.

        Args:
            deg_df: DEG results (gene_id, log2FC, padj, direction)
            hub_genes_df: Hub gene results (gene_id, hub_score, etc.)
            integrated_df: Integrated gene table with all annotations
            top_n: Number of top candidates per track

        Returns:
            Dict with known_drivers, novel_drivers, and summary
        """
        print(f"\n{'='*60}")
        print("🎯 Driver Gene Prediction")
        print(f"{'='*60}")
        print(f"Cancer type: {self.cancer_type}")
        print(f"DEG genes: {len(deg_df)}")

        # Prepare gene data
        gene_data = self._prepare_gene_data(deg_df, hub_genes_df, integrated_df)
        print(f"Genes for analysis: {len(gene_data)}")

        # Score Known Drivers
        known_candidates = self._score_known_drivers(gene_data)
        print(f"Known driver candidates: {len(known_candidates)}")

        # Score Novel Drivers
        novel_candidates = self._score_novel_drivers(gene_data)
        print(f"Novel driver candidates: {len(novel_candidates)}")

        # Select top candidates
        self.results['known_drivers'] = sorted(
            known_candidates,
            key=lambda x: x.score,
            reverse=True
        )[:top_n]

        self.results['novel_drivers'] = sorted(
            novel_candidates,
            key=lambda x: x.score,
            reverse=True
        )[:top_n]

        # Generate summary
        self.results['summary'] = self._generate_summary()

        # Add gene function descriptions
        self._add_gene_functions()

        print(f"\n✅ Top Known Drivers: {len(self.results['known_drivers'])}")
        print(f"✅ Top Novel Drivers: {len(self.results['novel_drivers'])}")

        return self.results

    def _add_gene_functions(self) -> None:
        """Add gene function descriptions from mygene/NCBI with Korean translation."""
        if not HAS_MYGENE:
            print("⚠️ mygene not installed. Gene functions will not be added.")
            return

        # Collect all gene symbols
        all_genes = []
        for driver in self.results['known_drivers']:
            all_genes.append(driver.gene_symbol)
        for driver in self.results['novel_drivers']:
            all_genes.append(driver.gene_symbol)

        if not all_genes:
            return

        try:
            mg = mygene.MyGeneInfo()
            # Query gene summary/description
            results = mg.querymany(
                all_genes,
                scopes='symbol',
                fields='summary,name',
                species='human',
                returnall=True
            )

            # Build gene -> function mapping (English)
            gene_functions_en = {}
            for hit in results.get('out', []):
                if isinstance(hit, dict) and 'query' in hit:
                    gene = hit['query'].upper()
                    # Prefer summary, fallback to name
                    summary = hit.get('summary', '')
                    name = hit.get('name', '')

                    if summary:
                        # Truncate long summaries
                        if len(summary) > 300:
                            summary = summary[:297] + '...'
                        gene_functions_en[gene] = summary
                    elif name:
                        gene_functions_en[gene] = name

            # Translate to Korean using LLM
            gene_functions_kr = self._translate_gene_functions(gene_functions_en)

            # Apply to driver candidates
            for driver in self.results['known_drivers']:
                driver.gene_function = gene_functions_kr.get(driver.gene_symbol.upper(), '')
            for driver in self.results['novel_drivers']:
                driver.gene_function = gene_functions_kr.get(driver.gene_symbol.upper(), '')

            func_count = len([d for d in self.results['known_drivers'] + self.results['novel_drivers'] if d.gene_function])
            print(f"✅ Added gene functions for {func_count} genes (Korean)")

        except Exception as e:
            print(f"⚠️ Failed to fetch gene functions: {e}")

    def _translate_gene_functions(self, gene_functions_en: Dict[str, str]) -> Dict[str, str]:
        """Translate gene functions to Korean using LLM."""
        if not gene_functions_en:
            return {}

        # Try using OpenAI for batch translation (faster)
        try:
            from openai import OpenAI
            import os

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Batch translate all functions at once
            functions_text = "\n".join([f"{gene}: {desc}" for gene, desc in gene_functions_en.items()])

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """유전자 기능 설명을 한국어로 번역하세요.
규칙:
1. 유전자 기호(KRAS, TP53 등)는 영어 그대로 유지
2. 단백질명, 경로명 등 전문 용어는 영어로 유지하고 괄호 안에 한글 설명 추가 가능
3. 각 줄은 "유전자명: 설명" 형식으로 출력
4. 간결하게 핵심만 번역 (150자 이내)
5. 번역만 출력, 다른 설명 없이"""
                    },
                    {"role": "user", "content": functions_text}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            # Parse response
            translated_text = response.choices[0].message.content.strip()
            gene_functions_kr = {}

            for line in translated_text.split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    gene = parts[0].strip().upper()
                    desc = parts[1].strip() if len(parts) > 1 else ""
                    if gene in gene_functions_en:
                        gene_functions_kr[gene] = desc

            print(f"✅ Translated {len(gene_functions_kr)} gene functions to Korean")
            return gene_functions_kr

        except Exception as e:
            print(f"⚠️ Translation failed, using English: {e}")
            return gene_functions_en

    def _prepare_gene_data(
        self,
        deg_df: pd.DataFrame,
        hub_genes_df: Optional[pd.DataFrame],
        integrated_df: Optional[pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Combine all gene data into unified structure."""
        gene_data = {}

        # Prefer integrated_df if it has gene_symbol column
        if integrated_df is not None and len(integrated_df) > 0 and 'gene_symbol' in integrated_df.columns:
            print("Using integrated_gene_table as primary source (has gene_symbol)")
            for _, row in integrated_df.iterrows():
                gene = str(row.get('gene_symbol', '')).strip().upper()

                # Skip empty or invalid gene symbols
                if not gene or gene == 'NAN' or gene.startswith('ENSG'):
                    continue

                gene_data[gene] = {
                    'gene_symbol': gene,
                    'log2fc': float(row.get('log2FC', row.get('log2FoldChange', 0))),
                    'padj': float(row.get('padj', row.get('adj.P.Val', 1))),
                    'direction': str(row.get('direction', 'up' if row.get('log2FC', 0) > 0 else 'down')),
                    'hub_score': float(row.get('hub_score', 0)),
                    'is_hub': bool(row.get('is_hub', False)),
                    'pathway_count': int(row.get('pathway_count', 0)),
                    'db_matched': bool(row.get('db_matched', False))
                }
        else:
            # Fallback to DEG data
            print("Using DEG as primary source")
            gene_col = 'gene_id' if 'gene_id' in deg_df.columns else deg_df.columns[0]
            for _, row in deg_df.iterrows():
                gene = str(row[gene_col]).split('.')[0].upper()  # Remove version, uppercase

                # Skip if looks like Ensembl ID without symbol
                if gene.startswith('ENSG'):
                    continue

                gene_data[gene] = {
                    'gene_symbol': gene,
                    'log2fc': float(row.get('log2FC', row.get('log2FoldChange', 0))),
                    'padj': float(row.get('padj', row.get('adj.P.Val', 1))),
                    'direction': str(row.get('direction', 'up' if row.get('log2FC', 0) > 0 else 'down')),
                    'hub_score': 0.0,
                    'is_hub': False,
                    'pathway_count': 0,
                    'db_matched': False
                }

            # Add hub gene info
            if hub_genes_df is not None and len(hub_genes_df) > 0:
                hub_col = 'gene_id' if 'gene_id' in hub_genes_df.columns else hub_genes_df.columns[0]
                for _, row in hub_genes_df.iterrows():
                    gene = str(row[hub_col]).split('.')[0].upper()
                    if gene in gene_data:
                        gene_data[gene]['hub_score'] = float(row.get('hub_score', 0))
                        gene_data[gene]['is_hub'] = True

        return gene_data

    def _score_known_drivers(self, gene_data: Dict[str, Dict]) -> List[DriverCandidate]:
        """Score genes for Known Driver track."""
        candidates = []

        for gene, data in gene_data.items():
            # Only consider genes in driver databases
            if not self.db.is_known_driver(gene):
                continue

            # Get database info
            cosmic_info = self.db.get_cosmic_info(gene)
            tcga_freq, tcga_count, hotspots = self.db.get_tcga_mutation_freq(gene)

            # Get new evidence fields
            literature_support = self.db.get_literature_support(gene, self.cancer_type)
            is_pancancer, pancancer_cancers = self.db.get_pancancer_driver_status(gene)
            role_consistent = self.db.check_role_expression_consistency(gene, data['direction'])

            # Calculate score components
            # 1. COSMIC tier (25%)
            cosmic_score = 0
            if cosmic_info:
                cosmic_score = 25 if cosmic_info['tier'] == 'Tier1' else 15

            # 2. TCGA mutation frequency (25%)
            tcga_score = min(25, tcga_freq * 50)  # Max 25 for 50%+ frequency

            # 3. Expression change (25%)
            expr_score = 0
            abs_fc = abs(data['log2fc'])
            if abs_fc > 2:
                expr_score = 25
            elif abs_fc > 1:
                expr_score = 20
            elif abs_fc > 0.5:
                expr_score = 10

            # 4. Statistical significance (15%)
            sig_score = 0
            if data['padj'] < 0.001:
                sig_score = 15
            elif data['padj'] < 0.01:
                sig_score = 12
            elif data['padj'] < 0.05:
                sig_score = 8

            # 5. Hub gene bonus (10%)
            hub_bonus = 10 if data['is_hub'] else 0

            # Total score
            total_score = cosmic_score + tcga_score + expr_score + sig_score + hub_bonus

            # Get validation suggestion
            val_method, val_detail = self.db.get_validation_suggestion(gene, self.cancer_type)

            candidate = DriverCandidate(
                gene_symbol=gene,
                track='known',
                score=total_score,
                log2fc=data['log2fc'],
                padj=data['padj'],
                direction=data['direction'],
                hub_score=data['hub_score'],
                is_hub=data['is_hub'],
                cosmic_tier=cosmic_info['tier'] if cosmic_info else None,
                cosmic_role=cosmic_info['role'] if cosmic_info else None,
                tcga_mutation_freq=tcga_freq,
                tcga_sample_count=tcga_count,
                hotspots=hotspots,
                literature_support=literature_support,
                is_pancancer_driver=is_pancancer,
                pancancer_cancers=pancancer_cancers,
                role_expression_consistent=role_consistent,
                validation_method=val_method,
                validation_detail=val_detail
            )
            candidates.append(candidate)

        return candidates

    def _score_novel_drivers(self, gene_data: Dict[str, Dict]) -> List[DriverCandidate]:
        """
        Score genes for Candidate Regulator track (formerly Novel Driver).

        NOTE: These are NOT confirmed drivers but network-level key regulators
        that warrant further investigation. The term "candidate_regulator" is used
        instead of "novel_driver" to avoid overclaiming.
        """
        candidates = []

        for gene, data in gene_data.items():
            # Skip known drivers (they go to Known track)
            if self.db.is_known_driver(gene):
                continue

            # Minimum criteria for candidate regulator consideration
            if abs(data['log2fc']) < 1.0 or data['padj'] > 0.05:
                continue

            # Get evidence fields
            literature_support = self.db.get_literature_support(gene, self.cancer_type)
            is_pancancer, pancancer_cancers = self.db.get_pancancer_driver_status(gene)

            # Calculate score components (different weighting)
            # 1. Expression change magnitude (30%)
            expr_score = 0
            abs_fc = abs(data['log2fc'])
            if abs_fc > 3:
                expr_score = 30
            elif abs_fc > 2:
                expr_score = 25
            elif abs_fc > 1.5:
                expr_score = 20
            elif abs_fc > 1:
                expr_score = 15

            # 2. Hub gene / network centrality (30%)
            hub_score = 0
            if data['is_hub']:
                hub_score = 30
            elif data['hub_score'] > 0.5:
                hub_score = 20
            elif data['hub_score'] > 0.3:
                hub_score = 10

            # 3. Statistical significance (20%)
            sig_score = 0
            if data['padj'] < 0.0001:
                sig_score = 20
            elif data['padj'] < 0.001:
                sig_score = 15
            elif data['padj'] < 0.01:
                sig_score = 10
            elif data['padj'] < 0.05:
                sig_score = 5

            # 4. Pathway involvement (10%)
            pathway_score = min(10, data['pathway_count'] * 2)

            # 5. DB validation bonus (10%)
            db_score = 10 if data['db_matched'] else 0

            # Total score
            total_score = expr_score + hub_score + sig_score + pathway_score + db_score

            # Minimum threshold for candidates
            if total_score < 40:
                continue

            # Determine appropriate validation method based on evidence
            if literature_support == "emerging":
                val_method = 'Literature validation + Functional'
                val_detail = f'{gene}: Review existing literature, then knockdown/overexpression assay'
            elif is_pancancer:
                val_method = 'Cross-cancer validation'
                val_detail = f'{gene}: Compare with {", ".join(pancancer_cancers[:2])} datasets'
            else:
                val_method = 'Functional validation'
                val_detail = f'{gene} knockdown/overexpression + phenotype assay'

            candidate = DriverCandidate(
                gene_symbol=gene,
                track='candidate_regulator',  # Renamed from 'novel'
                score=total_score,
                log2fc=data['log2fc'],
                padj=data['padj'],
                direction=data['direction'],
                hub_score=data['hub_score'],
                is_hub=data['is_hub'],
                pathway_impact=pathway_score / 10,
                literature_support=literature_support,
                is_pancancer_driver=is_pancancer,
                pancancer_cancers=pancancer_cancers,
                validation_method=val_method,
                validation_detail=val_detail
            )
            candidates.append(candidate)

        return candidates

    def _generate_summary(self) -> Dict:
        """Generate analysis summary."""
        known = self.results['known_drivers']
        regulators = self.results['novel_drivers']  # Now called candidate_regulators

        # High confidence counts
        high_conf_known = len([d for d in known if d.score >= 70])
        high_conf_regulators = len([d for d in regulators if d.score >= 70])

        # Top actionable targets (known drivers with high confidence)
        actionable = [d.gene_symbol for d in known if d.score >= 70][:3]

        # Top research targets (candidate regulators for further study)
        research = [d.gene_symbol for d in regulators if d.score >= 70][:3]

        # Literature support breakdown for candidate regulators
        lit_counts = {
            'well_established': len([d for d in regulators if d.literature_support == 'well_established']),
            'emerging': len([d for d in regulators if d.literature_support == 'emerging']),
            'uncharacterized': len([d for d in regulators if d.literature_support == 'uncharacterized'])
        }

        # Role-expression consistency for known drivers
        consistent_count = len([d for d in known if d.role_expression_consistent is True])
        inconsistent_count = len([d for d in known if d.role_expression_consistent is False])

        return {
            'total_known_candidates': len(known),
            'total_candidate_regulators': len(regulators),  # Renamed from novel
            'high_confidence_known': high_conf_known,
            'high_confidence_regulators': high_conf_regulators,
            'actionable_targets': actionable,
            'research_targets': research,
            'cancer_type': self.cancer_type,
            # New fields
            'literature_support_breakdown': lit_counts,
            'role_expression_consistent': consistent_count,
            'role_expression_inconsistent': inconsistent_count
        }

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert results to DataFrames."""
        known_df = pd.DataFrame([d.to_dict() for d in self.results['known_drivers']])
        regulator_df = pd.DataFrame([d.to_dict() for d in self.results['novel_drivers']])
        return known_df, regulator_df

    def save_results(self, output_dir: Path):
        """Save results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        known_df, regulator_df = self.to_dataframe()

        if len(known_df) > 0:
            known_df.to_csv(output_dir / 'driver_known.csv', index=False)

        if len(regulator_df) > 0:
            # Save as candidate_regulators (renamed from novel)
            regulator_df.to_csv(output_dir / 'driver_candidate_regulators.csv', index=False)
            # Also keep old filename for backwards compatibility
            regulator_df.to_csv(output_dir / 'driver_novel.csv', index=False)

        # Save summary
        with open(output_dir / 'driver_summary.json', 'w') as f:
            json.dump(self.results['summary'], f, indent=2)

        print(f"\n💾 Driver results saved to {output_dir}")


def predict_drivers(
    deg_csv: str,
    hub_genes_csv: Optional[str] = None,
    integrated_csv: Optional[str] = None,
    cancer_type: str = "unknown",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run driver prediction.

    Args:
        deg_csv: Path to DEG results CSV
        hub_genes_csv: Path to hub genes CSV (optional)
        integrated_csv: Path to integrated gene table CSV (optional)
        cancer_type: Cancer type for context
        output_dir: Output directory (optional)

    Returns:
        Prediction results dict
    """
    # Load data
    deg_df = pd.read_csv(deg_csv)
    hub_df = pd.read_csv(hub_genes_csv) if hub_genes_csv else None
    int_df = pd.read_csv(integrated_csv) if integrated_csv else None

    # Run prediction
    predictor = DriverPredictor(cancer_type)
    results = predictor.predict(deg_df, hub_df, int_df)

    # Save if output directory specified
    if output_dir:
        predictor.save_results(Path(output_dir))

    return results


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict driver genes from RNA-seq data")
    parser.add_argument("--deg", required=True, help="DEG results CSV")
    parser.add_argument("--hub", help="Hub genes CSV")
    parser.add_argument("--integrated", help="Integrated gene table CSV")
    parser.add_argument("--cancer", default="unknown", help="Cancer type")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()

    results = predict_drivers(
        deg_csv=args.deg,
        hub_genes_csv=args.hub,
        integrated_csv=args.integrated,
        cancer_type=args.cancer,
        output_dir=args.output
    )

    print("\n" + "="*60)
    print("Summary:")
    print(json.dumps(results['summary'], indent=2))
