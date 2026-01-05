#!/usr/bin/env python3
"""
Disease Database Module

Provides gene-disease association lookups from:
- DisGeNET (online API + offline cache)
- COSMIC (cancer mutations)
- Local curated database

Author: BioInsight AI
"""

import requests
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from functools import lru_cache
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DiseaseAssociation:
    """Gene-Disease association data"""
    disease_id: str
    disease_name: str
    score: float  # 0-1 normalized score
    source: str   # DisGeNET, COSMIC, OMIM, etc.
    pmid_count: int = 0
    evidence_type: str = ""  # curated, literature, etc.


@dataclass
class GeneCard:
    """Complete gene status card with disease associations"""
    gene_symbol: str
    regulation: str  # Upregulated / Downregulated
    log2_fold_change: float
    p_value: float
    adjusted_p_value: float
    fold_change: float  # 2^|log2FC|

    # Disease associations
    diseases: List[DiseaseAssociation] = field(default_factory=list)
    top_disease: Optional[str] = None
    top_disease_score: float = 0.0

    # Therapeutic info
    therapeutics: List[str] = field(default_factory=list)

    # Literature evidence
    supporting_papers: List[Dict] = field(default_factory=list)

    # Sources used
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'gene_symbol': self.gene_symbol,
            'regulation': self.regulation,
            'log2_fold_change': self.log2_fold_change,
            'p_value': self.p_value,
            'adjusted_p_value': self.adjusted_p_value,
            'fold_change': self.fold_change,
            'top_disease': self.top_disease,
            'top_disease_score': self.top_disease_score,
            'disease_count': len(self.diseases),
            'therapeutic_count': len(self.therapeutics),
            'sources': self.sources
        }


class DiseaseDatabase:
    """
    Disease database with multi-source gene-disease associations
    """

    # Curated cancer gene database (expanded)
    CANCER_GENES = {
        # Oncogenes
        'EGFR': {'type': 'oncogene', 'diseases': ['Lung Cancer', 'Glioblastoma', 'Colorectal Cancer'],
                 'therapeutics': ['Erlotinib', 'Gefitinib', 'Osimertinib', 'Cetuximab']},
        'KRAS': {'type': 'oncogene', 'diseases': ['Pancreatic Cancer', 'Lung Cancer', 'Colorectal Cancer'],
                 'therapeutics': ['Sotorasib (G12C)', 'Adagrasib (G12C)']},
        'BRAF': {'type': 'oncogene', 'diseases': ['Melanoma', 'Colorectal Cancer', 'Thyroid Cancer'],
                 'therapeutics': ['Vemurafenib', 'Dabrafenib', 'Encorafenib']},
        'MYC': {'type': 'oncogene', 'diseases': ['Burkitt Lymphoma', 'Breast Cancer', 'Lung Cancer'],
                'therapeutics': ['Omomyc (investigational)']},
        'PIK3CA': {'type': 'oncogene', 'diseases': ['Breast Cancer', 'Colorectal Cancer', 'Ovarian Cancer'],
                   'therapeutics': ['Alpelisib']},
        'ERBB2': {'type': 'oncogene', 'diseases': ['Breast Cancer', 'Gastric Cancer'],
                  'therapeutics': ['Trastuzumab', 'Pertuzumab', 'T-DM1', 'Lapatinib']},
        'ALK': {'type': 'oncogene', 'diseases': ['Lung Cancer', 'Neuroblastoma', 'Lymphoma'],
                'therapeutics': ['Crizotinib', 'Alectinib', 'Lorlatinib']},
        'MET': {'type': 'oncogene', 'diseases': ['Lung Cancer', 'Renal Cancer', 'Gastric Cancer'],
                'therapeutics': ['Capmatinib', 'Tepotinib']},
        'RET': {'type': 'oncogene', 'diseases': ['Thyroid Cancer', 'Lung Cancer'],
                'therapeutics': ['Selpercatinib', 'Pralsetinib']},
        'ROS1': {'type': 'oncogene', 'diseases': ['Lung Cancer'],
                 'therapeutics': ['Crizotinib', 'Entrectinib']},

        # Tumor suppressors
        'TP53': {'type': 'tumor_suppressor', 'diseases': ['Pan-Cancer (>50%)', 'Li-Fraumeni Syndrome'],
                 'therapeutics': ['APR-246 (investigational)', 'p53 reactivators']},
        'RB1': {'type': 'tumor_suppressor', 'diseases': ['Retinoblastoma', 'Small Cell Lung Cancer'],
                'therapeutics': ['CDK4/6 inhibitors (context-dependent)']},
        'PTEN': {'type': 'tumor_suppressor', 'diseases': ['Prostate Cancer', 'Breast Cancer', 'Glioblastoma'],
                 'therapeutics': ['mTOR inhibitors', 'PI3K inhibitors']},
        'BRCA1': {'type': 'tumor_suppressor', 'diseases': ['Breast Cancer', 'Ovarian Cancer'],
                  'therapeutics': ['Olaparib', 'Talazoparib', 'Niraparib']},
        'BRCA2': {'type': 'tumor_suppressor', 'diseases': ['Breast Cancer', 'Ovarian Cancer', 'Pancreatic Cancer'],
                  'therapeutics': ['PARP inhibitors']},
        'APC': {'type': 'tumor_suppressor', 'diseases': ['Colorectal Cancer', 'FAP'],
                'therapeutics': ['Celecoxib (prevention)']},
        'CDKN2A': {'type': 'tumor_suppressor', 'diseases': ['Melanoma', 'Pancreatic Cancer', 'Lung Cancer'],
                   'therapeutics': ['CDK4/6 inhibitors']},
        'NF1': {'type': 'tumor_suppressor', 'diseases': ['Neurofibromatosis', 'Glioma'],
                'therapeutics': ['MEK inhibitors']},
        'VHL': {'type': 'tumor_suppressor', 'diseases': ['Renal Cell Carcinoma', 'VHL Syndrome'],
                'therapeutics': ['HIF-2α inhibitors', 'Sunitinib']},

        # Cell cycle / Proliferation
        'CDK4': {'type': 'oncogene', 'diseases': ['Breast Cancer', 'Melanoma', 'Sarcoma'],
                 'therapeutics': ['Palbociclib', 'Ribociclib', 'Abemaciclib']},
        'CCND1': {'type': 'oncogene', 'diseases': ['Breast Cancer', 'Head and Neck Cancer', 'Mantle Cell Lymphoma'],
                  'therapeutics': ['CDK4/6 inhibitors']},
        'CCNB1': {'type': 'proliferation', 'diseases': ['Multiple Cancers (proliferation marker)'],
                  'therapeutics': []},
        'TOP2A': {'type': 'proliferation', 'diseases': ['Breast Cancer', 'Leukemia'],
                  'therapeutics': ['Doxorubicin', 'Etoposide']},
        'BIRC5': {'type': 'anti-apoptotic', 'diseases': ['Multiple Cancers'],
                  'therapeutics': ['YM155 (investigational)']},
        'AURKA': {'type': 'proliferation', 'diseases': ['Breast Cancer', 'Ovarian Cancer'],
                  'therapeutics': ['Alisertib (investigational)']},

        # Angiogenesis / TME
        'VEGFA': {'type': 'angiogenesis', 'diseases': ['Multiple Cancers (angiogenesis)'],
                  'therapeutics': ['Bevacizumab', 'Ramucirumab']},
        'HIF1A': {'type': 'hypoxia', 'diseases': ['Multiple Cancers (hypoxia response)'],
                  'therapeutics': ['HIF inhibitors (investigational)']},

        # Invasion / Metastasis
        'MMP1': {'type': 'invasion', 'diseases': ['Multiple Cancers (metastasis)'],
                 'therapeutics': []},
        'MMP9': {'type': 'invasion', 'diseases': ['Multiple Cancers (metastasis)'],
                 'therapeutics': []},
        'SPP1': {'type': 'invasion', 'diseases': ['Multiple Cancers (metastasis marker)'],
                 'therapeutics': []},
        'COL1A1': {'type': 'stroma', 'diseases': ['Fibrosis', 'Multiple Cancers (TME)'],
                   'therapeutics': []},

        # Apoptosis
        'BCL2': {'type': 'anti-apoptotic', 'diseases': ['CLL', 'Lymphoma'],
                 'therapeutics': ['Venetoclax']},
        'BAX': {'type': 'pro-apoptotic', 'diseases': ['Multiple Cancers'],
                'therapeutics': []},

        # Signaling
        'AKT1': {'type': 'oncogene', 'diseases': ['Breast Cancer', 'Colorectal Cancer'],
                 'therapeutics': ['AKT inhibitors (Capivasertib)']},
        'MTOR': {'type': 'signaling', 'diseases': ['Renal Cancer', 'Breast Cancer'],
                 'therapeutics': ['Everolimus', 'Temsirolimus']},
        'STK11': {'type': 'tumor_suppressor', 'diseases': ['Lung Cancer', 'Peutz-Jeghers Syndrome'],
                  'therapeutics': []},

        # Transcription factors
        'FOXM1': {'type': 'transcription_factor', 'diseases': ['Multiple Cancers (proliferation)'],
                  'therapeutics': []},
        'E2F1': {'type': 'transcription_factor', 'diseases': ['Multiple Cancers (cell cycle)'],
                 'therapeutics': ['CDK4/6 inhibitors (indirect)']},

        # DNA repair / Replication
        'MCM2': {'type': 'replication', 'diseases': ['Multiple Cancers (proliferation marker)'],
                 'therapeutics': []},
        'TYMS': {'type': 'metabolism', 'diseases': ['Colorectal Cancer'],
                 'therapeutics': ['5-Fluorouracil', 'Capecitabine']},

        # Cell cycle checkpoints
        'CDC20': {'type': 'cell_cycle', 'diseases': ['Multiple Cancers'],
                  'therapeutics': []},
        'BUB1': {'type': 'cell_cycle', 'diseases': ['Multiple Cancers (CIN)'],
                 'therapeutics': []},
        'UBE2C': {'type': 'cell_cycle', 'diseases': ['Multiple Cancers'],
                  'therapeutics': []},
    }

    # Disease name normalization
    DISEASE_ALIASES = {
        'nsclc': 'Lung Cancer',
        'non-small cell lung cancer': 'Lung Cancer',
        'sclc': 'Small Cell Lung Cancer',
        'crc': 'Colorectal Cancer',
        'hcc': 'Liver Cancer',
        'rcc': 'Renal Cell Carcinoma',
        'aml': 'Acute Myeloid Leukemia',
        'cll': 'Chronic Lymphocytic Leukemia',
        'dlbcl': 'Diffuse Large B-Cell Lymphoma',
        'gbm': 'Glioblastoma',
        'pdac': 'Pancreatic Cancer',
    }

    def __init__(self, cache_dir: str = ".disease_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # DisGeNET API (requires API key for production)
        self.disgenet_api = "https://www.disgenet.org/api"
        self.disgenet_key = None  # Set if you have API key

        # Cache for API responses
        self._cache = {}

    def get_gene_diseases(
        self,
        gene_symbol: str,
        use_api: bool = False
    ) -> List[DiseaseAssociation]:
        """
        Get disease associations for a gene

        Args:
            gene_symbol: Gene symbol (e.g., 'TP53')
            use_api: Whether to query DisGeNET API

        Returns:
            List of DiseaseAssociation objects
        """
        associations = []

        # 1. Check local curated database first
        if gene_symbol in self.CANCER_GENES:
            info = self.CANCER_GENES[gene_symbol]
            for disease in info.get('diseases', []):
                associations.append(DiseaseAssociation(
                    disease_id=f"LOCAL:{disease.replace(' ', '_')}",
                    disease_name=disease,
                    score=0.9,  # High score for curated
                    source='BioInsight_Curated',
                    evidence_type='curated'
                ))

        # 2. Query DisGeNET API if enabled
        if use_api and self.disgenet_key:
            api_results = self._query_disgenet(gene_symbol)
            associations.extend(api_results)

        # 3. Sort by score
        associations.sort(key=lambda x: x.score, reverse=True)

        return associations

    def get_gene_therapeutics(self, gene_symbol: str) -> List[str]:
        """Get known therapeutics targeting a gene"""
        if gene_symbol in self.CANCER_GENES:
            return self.CANCER_GENES[gene_symbol].get('therapeutics', [])
        return []

    def get_gene_type(self, gene_symbol: str) -> str:
        """Get gene functional type"""
        if gene_symbol in self.CANCER_GENES:
            return self.CANCER_GENES[gene_symbol].get('type', 'unknown')
        return 'unknown'

    def create_gene_card(
        self,
        gene_symbol: str,
        log2fc: float,
        pvalue: float,
        padj: float,
        use_api: bool = False
    ) -> GeneCard:
        """
        Create a complete GeneCard for a gene

        Args:
            gene_symbol: Gene symbol
            log2fc: log2 fold change
            pvalue: Raw p-value
            padj: Adjusted p-value
            use_api: Query external APIs

        Returns:
            GeneCard object with disease associations
        """
        # Determine regulation
        if padj < 0.05 and log2fc > 1:
            regulation = "Upregulated"
        elif padj < 0.05 and log2fc < -1:
            regulation = "Downregulated"
        else:
            regulation = "Not Significant"

        # Get disease associations
        diseases = self.get_gene_diseases(gene_symbol, use_api)

        # Get therapeutics
        therapeutics = self.get_gene_therapeutics(gene_symbol)

        # Create card
        card = GeneCard(
            gene_symbol=gene_symbol,
            regulation=regulation,
            log2_fold_change=log2fc,
            p_value=pvalue,
            adjusted_p_value=padj,
            fold_change=2 ** abs(log2fc),
            diseases=diseases,
            top_disease=diseases[0].disease_name if diseases else None,
            top_disease_score=diseases[0].score if diseases else 0.0,
            therapeutics=therapeutics,
            sources=['BioInsight_Curated']
        )

        return card

    def create_gene_cards_from_deg(
        self,
        deg_results: pd.DataFrame,
        gene_col: str = 'gene',
        log2fc_col: str = 'log2FoldChange',
        pvalue_col: str = 'pvalue',
        padj_col: str = 'padj',
        top_n: int = 50,
        use_api: bool = False
    ) -> List[GeneCard]:
        """
        Create GeneCards from DEG results DataFrame

        Args:
            deg_results: DEG results DataFrame
            top_n: Number of top genes to process

        Returns:
            List of GeneCard objects
        """
        # Filter significant
        sig = deg_results[
            (deg_results[padj_col] < 0.05) &
            (abs(deg_results[log2fc_col]) > 1)
        ].nlargest(top_n, log2fc_col, keep='all')

        cards = []
        for _, row in sig.iterrows():
            card = self.create_gene_card(
                gene_symbol=row[gene_col],
                log2fc=row[log2fc_col],
                pvalue=row.get(pvalue_col, 0),
                padj=row[padj_col],
                use_api=use_api
            )
            cards.append(card)

        logger.info(f"Created {len(cards)} gene cards")

        return cards

    def get_disease_enrichment(
        self,
        gene_list: List[str],
        min_genes: int = 2
    ) -> pd.DataFrame:
        """
        Perform disease enrichment analysis

        Args:
            gene_list: List of gene symbols
            min_genes: Minimum genes per disease

        Returns:
            DataFrame with disease enrichment results
        """
        disease_genes = {}  # disease -> [genes]

        for gene in gene_list:
            if gene in self.CANCER_GENES:
                for disease in self.CANCER_GENES[gene].get('diseases', []):
                    if disease not in disease_genes:
                        disease_genes[disease] = []
                    disease_genes[disease].append(gene)

        # Filter and sort
        results = []
        for disease, genes in disease_genes.items():
            if len(genes) >= min_genes:
                results.append({
                    'disease': disease,
                    'gene_count': len(genes),
                    'genes': ', '.join(sorted(genes)),
                    'score': len(genes) / len(gene_list)
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('gene_count', ascending=False)

        return df

    def validate_hub_genes(
        self,
        hub_genes: pd.DataFrame,
        gene_col: str = 'gene',
        top_n: int = 50
    ) -> Dict:
        """
        Validate hub genes against known cancer genes

        Returns:
            Validation summary dictionary
        """
        top_hubs = set(hub_genes.head(top_n)[gene_col].tolist())
        known = set(self.CANCER_GENES.keys())

        validated = top_hubs & known

        # Categorize validated genes
        oncogenes = [g for g in validated if self.CANCER_GENES.get(g, {}).get('type') == 'oncogene']
        tumor_suppressors = [g for g in validated if self.CANCER_GENES.get(g, {}).get('type') == 'tumor_suppressor']
        other = [g for g in validated if g not in oncogenes and g not in tumor_suppressors]

        # Get therapeutics coverage
        genes_with_therapeutics = [
            g for g in validated
            if len(self.CANCER_GENES.get(g, {}).get('therapeutics', [])) > 0
        ]

        return {
            'total_analyzed': top_n,
            'validated_count': len(validated),
            'validation_rate': len(validated) / top_n,
            'validated_genes': sorted(validated),
            'oncogenes': oncogenes,
            'tumor_suppressors': tumor_suppressors,
            'other_types': other,
            'with_therapeutics': genes_with_therapeutics,
            'therapeutic_coverage': len(genes_with_therapeutics) / max(len(validated), 1)
        }

    def _query_disgenet(self, gene_symbol: str) -> List[DiseaseAssociation]:
        """Query DisGeNET API for gene-disease associations"""
        if not self.disgenet_key:
            return []

        # Check cache first
        cache_file = self.cache_dir / f"{gene_symbol}_disgenet.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return self._parse_disgenet_response(data)

        try:
            headers = {'Authorization': f'Bearer {self.disgenet_key}'}
            response = requests.get(
                f"{self.disgenet_api}/gda/gene/{gene_symbol}",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                # Cache response
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                return self._parse_disgenet_response(data)

        except Exception as e:
            logger.warning(f"DisGeNET query failed for {gene_symbol}: {e}")

        return []

    def _parse_disgenet_response(self, data: List[Dict]) -> List[DiseaseAssociation]:
        """Parse DisGeNET API response"""
        associations = []

        for item in data[:20]:  # Limit to top 20
            associations.append(DiseaseAssociation(
                disease_id=item.get('diseaseid', ''),
                disease_name=item.get('disease_name', ''),
                score=float(item.get('score', 0)),
                source='DisGeNET',
                pmid_count=int(item.get('pmid_count', 0)),
                evidence_type=item.get('source', '')
            ))

        return associations


def get_gene_cards_summary(cards: List[GeneCard]) -> str:
    """Generate text summary of gene cards"""
    lines = [
        "=" * 70,
        "GENE STATUS CARDS SUMMARY",
        "=" * 70,
        f"\nTotal genes analyzed: {len(cards)}",
        f"With disease associations: {len([c for c in cards if c.diseases])}",
        f"With therapeutics: {len([c for c in cards if c.therapeutics])}",
        "\n" + "-" * 70,
        "TOP GENE CARDS:",
        "-" * 70
    ]

    for i, card in enumerate(cards[:15], 1):
        lines.append(f"\n{i}. {card.gene_symbol} ({card.regulation})")
        lines.append(f"   Fold Change: {card.fold_change:.2f}x | padj: {card.adjusted_p_value:.2e}")

        if card.top_disease:
            lines.append(f"   Top Disease: {card.top_disease} (score: {card.top_disease_score:.2f})")

        if card.therapeutics:
            lines.append(f"   Therapeutics: {', '.join(card.therapeutics[:3])}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test
    db = DiseaseDatabase()

    # Test gene card creation
    card = db.create_gene_card('EGFR', log2fc=2.5, pvalue=1e-10, padj=1e-8)
    print(f"\nGene Card for {card.gene_symbol}:")
    print(f"  Regulation: {card.regulation}")
    print(f"  Fold Change: {card.fold_change:.2f}x")
    print(f"  Top Disease: {card.top_disease}")
    print(f"  Therapeutics: {card.therapeutics}")

    # Test disease enrichment
    test_genes = ['EGFR', 'KRAS', 'TP53', 'BRAF', 'PIK3CA', 'MYC', 'BRCA1']
    enrichment = db.get_disease_enrichment(test_genes)
    print(f"\nDisease Enrichment:")
    print(enrichment.to_string())
