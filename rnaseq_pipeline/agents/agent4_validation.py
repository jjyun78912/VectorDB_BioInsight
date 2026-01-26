"""
Agent 4: DB Validation & Interpretation ⭐

This is the CORE interpretive agent of the pipeline.
Validates DEGs against cancer gene databases and applies systematic interpretation.

Key Principle:
- DB match is not "proof" - it's context for interpretation
- DB mismatch is not "wrong" - it's "unknown" or "novel candidate"
- Always cross-reference with Network (hub) and Pathway results

Input:
- deg_significant.csv: From Agent 1
- hub_genes.csv: From Agent 2
- gene_to_pathway.csv: From Agent 3
- config.json: Analysis parameters

Output:
- db_matched_genes.csv: Genes found in cancer databases
- integrated_gene_table.csv: All DEGs with all annotations
- interpretation_report.json: Detailed interpretation with checklist
- meta_agent4.json: Execution metadata
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from ..utils.base_agent import BaseAgent

# Gene ID conversion
try:
    import mygene
    HAS_MYGENE = True
except ImportError:
    HAS_MYGENE = False

# RAG interpretation
try:
    from ..rag.gene_interpreter import GeneRAGInterpreter, create_interpreter
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

# Multi-source interpreter (RAG + External APIs)
try:
    from ..rag.enhanced_interpreter import MultiSourceGeneInterpreter, create_multisource_interpreter
    HAS_MULTISOURCE = True
except ImportError:
    HAS_MULTISOURCE = False


class ConfidenceLevel(str, Enum):
    """Interpretation confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOVEL = "novel_candidate"
    REQUIRES_VALIDATION = "requires_validation"


@dataclass
class InterpretationChecklist:
    """Checklist for systematic gene interpretation."""
    # For DB-matched genes
    cancer_type_match: Optional[bool] = None
    is_hub: bool = False
    hub_score: float = 0.0
    pathway_position: Optional[str] = None
    pathway_count: int = 0
    expression_direction_consistent: Optional[bool] = None
    stage_specific: Optional[bool] = None

    # For DB-unmatched genes
    in_known_pathway: bool = False
    tme_related: bool = False
    recent_literature: Optional[str] = None

    # Computed
    interpretation_score: float = 0.0
    confidence: str = "requires_validation"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ValidationAgent(BaseAgent):
    """Agent for cancer DB validation and interpretation."""

    # Known cancer gene databases (simplified local versions)
    # In production, these would be loaded from actual DB files
    COSMIC_TIER1_GENES = {
        'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'APC', 'RB1',
        'BRCA1', 'BRCA2', 'MYC', 'ERBB2', 'CDK4', 'MDM2', 'CCND1', 'CDKN2A',
        'ATM', 'AKT1', 'NRAS', 'HRAS', 'FGFR1', 'FGFR2', 'FGFR3', 'MET',
        'ALK', 'ROS1', 'RET', 'KIT', 'PDGFRA', 'ABL1', 'JAK2', 'BCL2',
        'VHL', 'NF1', 'NF2', 'WT1', 'SMAD4', 'CTNNB1', 'IDH1', 'IDH2'
    }

    ONCOKB_GENES = {
        'TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN', 'ERBB2', 'MYC',
        'BRCA1', 'BRCA2', 'ALK', 'ROS1', 'RET', 'MET', 'NTRK1', 'NTRK2',
        'FGFR1', 'FGFR2', 'FGFR3', 'CDKN2A', 'AKT1', 'MTOR', 'STK11'
    }

    # Tumor microenvironment related genes
    TME_GENES = {
        'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'TIM3', 'TIGIT',  # Immune checkpoints
        'CD8A', 'CD4', 'FOXP3', 'CD68', 'CD163',  # Immune cell markers
        'VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR',  # Angiogenesis
        'TGFB1', 'TGFB2', 'IL6', 'IL10', 'CXCL8',  # Cytokines
        'COL1A1', 'COL3A1', 'FN1', 'FAP', 'ACTA2'  # Stromal markers
    }

    # Cancer type specific genes (simplified)
    CANCER_TYPE_GENES = {
        'lung_cancer': {'EGFR', 'KRAS', 'ALK', 'ROS1', 'MET', 'BRAF', 'RET', 'ERBB2', 'STK11'},
        'breast_cancer': {'BRCA1', 'BRCA2', 'ERBB2', 'ESR1', 'PGR', 'PIK3CA', 'CDH1', 'GATA3'},
        'colorectal_cancer': {'APC', 'KRAS', 'TP53', 'SMAD4', 'PIK3CA', 'BRAF', 'MLH1', 'MSH2'},
        'pancreatic_cancer': {'KRAS', 'TP53', 'CDKN2A', 'SMAD4', 'BRCA2'},
        'liver_cancer': {'TP53', 'CTNNB1', 'AXIN1', 'ARID1A', 'TERT'},
        'glioblastoma': {'EGFR', 'PTEN', 'TP53', 'IDH1', 'MGMT', 'TERT', 'CDKN2A'}
    }

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            "cancer_type": "lung_cancer",
            "databases": ["COSMIC", "OncoKB"],
            "score_weights": {
                "is_hub": 2.0,
                "db_matched": 2.0,
                "cancer_type_match": 1.5,
                "pathway_key_position": 1.0,
                "expression_consistent": 0.5,
                "high_pathway_count": 0.5
            },
            "confidence_thresholds": {
                "high": 5.0,
                "medium": 3.0,
                "low": 1.5
            },
            # Interpretation settings
            "enable_rag": True,  # Enable RAG-based literature interpretation
            "rag_max_genes": 20,  # Max genes to interpret (top by score)
            "rag_top_k": 5,  # Number of papers to retrieve per gene
            # Multi-source mode: RAG (VectorDB) + External API Context
            # Note: RAG = literature search, API = structured data (not RAG)
            "enable_multisource": True,  # Use multi-source interpreter
            "use_external_apis": True,  # Fetch structured data from external APIs
            # 2-stage validation settings
            "validation_stage": 1,  # 1 = DEG/Network/Pathway, 2 = ML Prediction
            "validate_ml_prediction": False  # Set True for stage 2
        }

        merged_config = {**default_config, **(config or {})}

        # Use different output directory for stage 2
        agent_name = "agent4_validation"
        if merged_config.get("validation_stage") == 2:
            agent_name = "agent4_validation_ml"

        super().__init__(agent_name, input_dir, output_dir, merged_config)

        self.deg_significant: Optional[pd.DataFrame] = None
        self.hub_genes: Optional[pd.DataFrame] = None
        self.gene_to_pathway: Optional[pd.DataFrame] = None
        self.ensembl_to_symbol: Dict[str, str] = {}  # Ensembl ID -> Gene Symbol mapping
        self.rag_interpreter: Optional[Any] = None  # RAG interpreter instance
        self.rag_interpretations: Dict[str, Any] = {}  # gene_symbol -> interpretation
        self.cancer_prediction: Optional[Dict[str, Any]] = None  # ML prediction results

    def validate_inputs(self) -> bool:
        """Validate inputs from previous agents.

        Stage 1: Validates DEG, Network, Pathway results
        Stage 2: Validates ML Prediction results
        """
        validation_stage = self.config.get('validation_stage', 1)
        self.logger.info(f"Running Validation Stage {validation_stage}")

        # Stage 2: ML Prediction validation
        if validation_stage == 2 or self.config.get('validate_ml_prediction', False):
            return self._validate_ml_inputs()

        # Stage 1: DEG/Network/Pathway validation (original behavior)
        # Required: DEG results
        self.deg_significant = self.load_csv("deg_significant.csv")
        if self.deg_significant is None:
            return False

        # Required: Hub genes (can be empty)
        self.hub_genes = self.load_csv("hub_genes.csv", required=False)
        if self.hub_genes is None:
            self.hub_genes = pd.DataFrame(columns=['gene_id', 'hub_score', 'degree'])

        # Required: Gene to pathway mapping (can be empty)
        self.gene_to_pathway = self.load_csv("gene_to_pathway.csv", required=False)
        if self.gene_to_pathway is None:
            self.gene_to_pathway = pd.DataFrame(columns=['gene_id', 'pathway_count', 'pathway_names'])

        self.logger.info(f"DEGs to validate: {len(self.deg_significant)}")
        self.logger.info(f"Hub genes available: {len(self.hub_genes)}")
        self.logger.info(f"Genes with pathway info: {len(self.gene_to_pathway)}")

        # Convert Ensembl IDs to Gene Symbols if needed
        self._build_gene_id_mapping()

        return True

    def _validate_ml_inputs(self) -> bool:
        """Validate ML prediction inputs for Stage 2 validation."""
        # Load ML prediction results
        cancer_prediction_file = self.input_dir / "cancer_prediction.json"
        if not cancer_prediction_file.exists():
            # Try parent directory (accumulated folder structure)
            parent_file = self.input_dir.parent / "cancer_prediction.json"
            if parent_file.exists():
                cancer_prediction_file = parent_file
            else:
                self.logger.warning("No ML prediction results found - skipping ML validation")
                return True  # Not an error, just skip

        try:
            with open(cancer_prediction_file, 'r', encoding='utf-8') as f:
                self.cancer_prediction = json.load(f)
            self.logger.info(f"Loaded ML prediction: {self.cancer_prediction.get('predicted_cancer', 'Unknown')}")
        except Exception as e:
            self.logger.error(f"Failed to load ML prediction: {e}")
            return False

        # Also load integrated gene table for cross-validation
        self.deg_significant = self.load_csv("integrated_gene_table.csv", required=False)
        if self.deg_significant is None:
            self.deg_significant = self.load_csv("deg_significant.csv", required=False)

        return True

    def _build_gene_id_mapping(self):
        """Build Gene ID to Gene Symbol mapping using mygene.

        Supports:
        - Ensembl IDs (ENSG00000141510)
        - Entrez Gene IDs (7157)
        - Gene Symbols (TP53) - passed through as-is
        """
        gene_ids = self.deg_significant['gene_id'].tolist()

        # Detect ID type
        sample_ids = [str(g) for g in gene_ids[:100]]

        # Check for Ensembl format
        ensembl_count = sum(1 for g in sample_ids if str(g).startswith('ENSG'))

        # Check for numeric (Entrez) format
        numeric_count = sum(1 for g in sample_ids if str(g).isdigit())

        # Check if already symbols (non-numeric, non-ENSG)
        symbol_count = sum(1 for g in sample_ids if not str(g).isdigit() and not str(g).startswith('ENSG'))

        self.logger.info(f"Gene ID format detection: Ensembl={ensembl_count}, Entrez={numeric_count}, Symbol={symbol_count}")

        # If mostly symbols already, no conversion needed
        if symbol_count > len(sample_ids) * 0.5:
            self.logger.info("Gene IDs appear to be symbols, no conversion needed")
            for gene_id in gene_ids:
                self.ensembl_to_symbol[str(gene_id)] = str(gene_id)
            return

        if not HAS_MYGENE:
            self.logger.warning("mygene not installed. Install with: pip install mygene")
            self.logger.warning("DB matching will not work without gene ID conversion")
            # Map to themselves as fallback
            for gene_id in gene_ids:
                self.ensembl_to_symbol[str(gene_id)] = str(gene_id)
            return

        try:
            mg = mygene.MyGeneInfo()
            unique_ids = list(set([str(g).split('.')[0] for g in gene_ids]))

            # Determine scope based on ID type
            if ensembl_count > numeric_count:
                scope = 'ensembl.gene'
                self.logger.info(f"Converting {len(unique_ids)} Ensembl IDs to Gene Symbols via mygene...")
            else:
                scope = 'entrezgene'
                self.logger.info(f"Converting {len(unique_ids)} Entrez Gene IDs to Gene Symbols via mygene...")

            # Query in batches
            batch_size = 1000
            for i in range(0, len(unique_ids), batch_size):
                batch = unique_ids[i:i+batch_size]
                results = mg.querymany(
                    batch,
                    scopes=scope,
                    fields='symbol',
                    species='human',
                    verbose=False
                )

                for r in results:
                    if 'symbol' in r:
                        self.ensembl_to_symbol[r['query']] = r['symbol']

            # Map versioned IDs (for Ensembl) and original IDs to symbols
            for gene_id in gene_ids:
                clean_id = str(gene_id).split('.')[0]
                if clean_id in self.ensembl_to_symbol:
                    self.ensembl_to_symbol[str(gene_id)] = self.ensembl_to_symbol[clean_id]
                elif str(gene_id) not in self.ensembl_to_symbol:
                    # Fallback: keep original ID if conversion failed
                    self.ensembl_to_symbol[str(gene_id)] = str(gene_id)

            converted = len([g for g in gene_ids if self.ensembl_to_symbol.get(str(g), str(g)) != str(g)])
            self.logger.info(f"Converted {converted}/{len(gene_ids)} gene IDs to Gene Symbols")

            # Log some examples
            examples = [(str(g), self.ensembl_to_symbol.get(str(g), str(g)))
                       for g in gene_ids[:5]
                       if self.ensembl_to_symbol.get(str(g), str(g)) != str(g)]
            if examples:
                self.logger.info(f"Conversion examples: {examples}")

        except Exception as e:
            self.logger.error(f"Gene ID conversion failed: {e}")
            self.logger.warning("DB matching may not work correctly")
            # Fallback: map to themselves
            for gene_id in gene_ids:
                self.ensembl_to_symbol[str(gene_id)] = str(gene_id)

    def _check_database_match(self, gene: str) -> Dict[str, Any]:
        """Check if gene is in cancer databases.

        Converts Ensembl ID to Gene Symbol if needed before matching.
        """
        # Convert Ensembl ID to Gene Symbol
        gene_symbol = self.ensembl_to_symbol.get(str(gene), gene)

        result = {
            'gene_symbol': gene_symbol,
            'in_cosmic': gene_symbol in self.COSMIC_TIER1_GENES,
            'in_oncokb': gene_symbol in self.ONCOKB_GENES,
            'cosmic_tier': 1 if gene_symbol in self.COSMIC_TIER1_GENES else None,
            'db_sources': []
        }

        if result['in_cosmic']:
            result['db_sources'].append('COSMIC')
        if result['in_oncokb']:
            result['db_sources'].append('OncoKB')

        result['db_matched'] = len(result['db_sources']) > 0

        return result

    def _check_cancer_type_match(self, gene: str) -> bool:
        """Check if gene is associated with the specific cancer type."""
        gene_symbol = self.ensembl_to_symbol.get(str(gene), gene)
        cancer_type = self.config["cancer_type"]
        type_genes = self.CANCER_TYPE_GENES.get(cancer_type, set())
        return gene_symbol in type_genes

    def _check_tme_related(self, gene: str) -> bool:
        """Check if gene is TME-related."""
        gene_symbol = self.ensembl_to_symbol.get(str(gene), gene)
        return gene_symbol in self.TME_GENES

    def _get_hub_info(self, gene: str) -> Tuple[bool, float, int]:
        """Get hub gene information."""
        if len(self.hub_genes) == 0:
            return False, 0.0, 0

        hub_row = self.hub_genes[self.hub_genes['gene_id'] == gene]
        if len(hub_row) == 0:
            return False, 0.0, 0

        hub_score = hub_row['hub_score'].values[0] if 'hub_score' in hub_row.columns else 0.0
        degree = hub_row['degree'].values[0] if 'degree' in hub_row.columns else 0
        return True, float(hub_score), int(degree)

    def _get_pathway_info(self, gene: str) -> Tuple[int, str]:
        """Get pathway information for gene."""
        if len(self.gene_to_pathway) == 0:
            return 0, ""

        pathway_row = self.gene_to_pathway[self.gene_to_pathway['gene_id'] == gene]
        if len(pathway_row) == 0:
            return 0, ""

        count = pathway_row['pathway_count'].values[0] if 'pathway_count' in pathway_row.columns else 0
        names = pathway_row['pathway_names'].values[0] if 'pathway_names' in pathway_row.columns else ""
        return int(count), str(names)

    def _calculate_interpretation_score(self, checklist: InterpretationChecklist, db_matched: bool) -> float:
        """Calculate interpretation score based on checklist."""
        weights = self.config["score_weights"]
        score = 0.0

        # Hub gene bonus
        if checklist.is_hub:
            score += weights["is_hub"]

        # DB match bonus
        if db_matched:
            score += weights["db_matched"]

        # Cancer type specific match
        if checklist.cancer_type_match:
            score += weights["cancer_type_match"]

        # In key pathways
        if checklist.pathway_count >= 3:
            score += weights["high_pathway_count"]

        # Expression direction consistent with literature
        if checklist.expression_direction_consistent:
            score += weights["expression_consistent"]

        return score

    def _determine_confidence(self, score: float, db_matched: bool, is_hub: bool) -> str:
        """Determine confidence level based on score and flags."""
        thresholds = self.config["confidence_thresholds"]

        if db_matched and score >= thresholds["high"]:
            return ConfidenceLevel.HIGH.value
        elif db_matched and score >= thresholds["medium"]:
            return ConfidenceLevel.MEDIUM.value
        elif not db_matched and is_hub:
            return ConfidenceLevel.NOVEL.value
        elif score >= thresholds["low"]:
            return ConfidenceLevel.LOW.value
        else:
            return ConfidenceLevel.REQUIRES_VALIDATION.value

    def _generate_tags(self, checklist: InterpretationChecklist, db_matched: bool) -> List[str]:
        """Generate interpretation tags."""
        tags = []

        if checklist.confidence == ConfidenceLevel.HIGH.value:
            tags.append("HIGH_CONFIDENCE")
        if db_matched and checklist.cancer_type_match:
            tags.append("KNOWN_CANCER_GENE")
        if not db_matched and checklist.is_hub:
            tags.append("NOVEL_CANDIDATE")
        if checklist.tme_related:
            tags.append("TME_SIGNAL")
        if checklist.is_hub:
            tags.append("HUB_GENE")
        if checklist.pathway_count >= 5:
            tags.append("PATHWAY_CENTRAL")
        if checklist.confidence == ConfidenceLevel.REQUIRES_VALIDATION.value:
            tags.append("REQUIRES_VALIDATION")

        return tags

    def _generate_narrative(self, gene: str, checklist: InterpretationChecklist,
                           db_matched: bool, deg_info: Dict) -> str:
        """Generate v2.0 conservative interpretation narrative.

        Follows principles:
        - No causal claims (no "drives", "induces", "controls")
        - No labeling as novel regulators or therapeutic targets
        - Use non-causal language ("is consistent with", "may reflect", "has been associated with")
        - Every interpretation must be traceable to provided results
        """
        direction = "upregulated" if deg_info['direction'] == 'up' else "downregulated"
        log2fc = deg_info['log2FC']

        # Basic observation (non-causal)
        narrative = f"{gene} shows {direction} expression (log2FC={log2fc:.2f}, padj<0.05). "

        # Context from DB (non-causal)
        if db_matched:
            if checklist.cancer_type_match:
                narrative += (
                    f"This gene has been previously associated with {self.config['cancer_type']} in cancer databases. "
                    f"The observed expression change is consistent with prior reports."
                )
            else:
                narrative += (
                    f"While {gene} appears in cancer databases, "
                    f"it has not been specifically linked to {self.config['cancer_type']}. "
                    f"The functional relevance in this context remains to be established."
                )
        else:
            if checklist.is_hub:
                narrative += (
                    f"This gene is not catalogued in major cancer databases but shows high network connectivity. "
                    f"Network centrality alone does not imply biological importance; "
                    f"functional validation would be required to assess significance."
                )
            elif checklist.tme_related:
                narrative += (
                    f"This gene is associated with tumor microenvironment markers. "
                    f"The expression change may reflect stromal or immune cell composition "
                    f"rather than tumor-intrinsic alterations."
                )
            else:
                narrative += (
                    f"This gene is not present in major cancer databases. "
                    f"Database absence does not indicate irrelevance; "
                    f"the current data are insufficient to draw conclusions about its biological role."
                )

        # Pathway context (non-causal)
        if checklist.pathway_count > 0:
            narrative += (
                f" Pathway enrichment analysis places this gene in {checklist.pathway_count} term(s), "
                f"providing functional context but not establishing causality."
            )

        return narrative

    def _generate_v2_interpretation(
        self,
        integrated_df: pd.DataFrame,
        matched_interpretations: List[Dict],
        novel_candidates: List[Dict]
    ) -> Dict[str, Any]:
        """Generate v2.0 structured interpretation report.

        Output format:
        - Observation: Major patterns in DEG, network, pathway
        - Supporting Evidence: Specific data supporting observations
        - Interpretation: Conservative, non-causal interpretation
        - Limitations: Analysis constraints and caveats
        """
        total_deg = len(integrated_df)
        up_count = int((integrated_df['direction'] == 'up').sum())
        down_count = int((integrated_df['direction'] == 'down').sum())
        hub_count = int(integrated_df['is_hub'].sum())
        db_matched_count = int(integrated_df['db_matched'].sum())
        pathway_genes = int((integrated_df['pathway_count'] > 0).sum())

        # Observation - 패턴 중심, 유전자 특이적 서술 없음
        observations = []
        observations.append(
            f"총 {total_deg:,}개의 차등 발현 유전자가 확인되었습니다 "
            f"(상향 발현 {up_count:,}개, 하향 발현 {down_count:,}개)."
        )

        if hub_count > 0:
            observations.append(
                f"네트워크 분석에서 연결성이 높은 {hub_count}개의 허브 유전자 후보가 식별되었습니다. "
                f"참고: 네트워크 중심성은 위상학적 특성이며, 생물학적 중요성을 직접적으로 의미하지 않습니다."
            )

        if db_matched_count > 0:
            match_ratio = db_matched_count / total_deg * 100
            observations.append(
                f"{db_matched_count}개의 DEG ({match_ratio:.1f}%)가 암 유전자 데이터베이스에 등록되어 있습니다. "
                f"대부분의 DEG는 데이터베이스에 등록되지 않았으며, 이는 전체 전사체 분석에서 일반적인 현상입니다."
            )
        else:
            observations.append(
                "주요 암 유전자 데이터베이스(COSMIC, OncoKB)에 등록된 DEG가 없습니다. "
                "이것이 무관성을 의미하지는 않으며, 데이터베이스의 제한된 범위와 잘 연구된 유전자에 대한 편향이 원인일 수 있습니다."
            )

        if pathway_genes > 0:
            observations.append(
                f"{pathway_genes}개의 DEG가 농축된 경로에 매핑되어 기능적 맥락을 제공합니다."
            )

        # Supporting Evidence - 실제 결과에 기반한 증거
        evidence = {
            "deg_statistics": {
                "total": total_deg,
                "upregulated": up_count,
                "downregulated": down_count
            },
            "network_analysis": {
                "hub_candidates": hub_count,
                "note": "허브 상태는 네트워크 위상 기반"
            },
            "database_validation": {
                "matched": db_matched_count,
                "databases_used": ["COSMIC Tier1", "OncoKB"],
                "note": "데이터베이스 범위가 제한적; 미등록이 무관성을 의미하지 않음"
            },
            "pathway_enrichment": {
                "genes_with_pathway": pathway_genes,
                "note": "농축은 통계적 연관성이며 인과 메커니즘 아님"
            }
        }

        # Interpretation - 보수적, 비인과적 표현
        interpretation_text = []

        if total_deg > 1000:
            interpretation_text.append(
                f"다수의 DEG ({total_deg:,}개)는 비교 그룹 간의 광범위한 전사체 변화를 반영할 수 있습니다. "
                "이는 이질적인 생물학적 상태와 일관되지만, 추가적인 증거 없이 특정 기능적 드라이버를 식별하기 어렵습니다."
            )
        elif total_deg > 100:
            interpretation_text.append(
                "적절한 수의 DEG는 그룹 간 검출 가능한 전사체 차이를 시사합니다. "
                "이러한 변화의 기능적 중요성은 검증 연구를 통해 확립되어야 합니다."
            )
        else:
            interpretation_text.append(
                "상대적으로 적은 수의 DEG가 확인되었으며, 이는 비교 그룹 간의 미묘한 차이 또는 "
                "제한된 통계적 검정력을 나타낼 수 있습니다."
            )

        if db_matched_count > 0 and db_matched_count / total_deg > 0.1:
            interpretation_text.append(
                "DEG 중 알려진 암 유전자의 존재는 암 관련 생물학과 일관되지만, "
                "인과성이나 치료적 관련성을 확립하지는 않습니다."
            )

        if hub_count > 0 and db_matched_count == 0:
            interpretation_text.append(
                "암 데이터베이스에 없는 허브 유전자가 식별되었습니다. "
                "이들은 새로운 후보를 나타낼 수 있지만, 데이터베이스 미등록은 생물학적 새로움보다는 "
                "제한된 사전 연구를 반영할 수도 있습니다. 그 중요성을 평가하려면 기능적 검증이 필요합니다."
            )

        # Limitations - 제한사항 명시
        limitations = [
            "RNA-seq는 전사체 풍부도를 측정하며, 단백질 수준이나 활성은 측정하지 않습니다.",
            "차등 발현은 통계적 발견이며, 생물학적 중요성이 보장되지 않습니다.",
            "암 유전자 데이터베이스는 범위가 제한적이며 잘 연구된 유전자에 편향되어 있습니다.",
            "네트워크 허브 상태는 위상학적 특성이며 조절 중요성을 직접적으로 나타내지 않습니다.",
            "경로 농축은 통계적 연관성을 보여주며, 기계론적 인과관계가 아닙니다.",
            "이 분석은 종양 고유 신호와 미세환경 효과를 구분하지 않습니다.",
            "모든 발견은 상관관계이며 인과 추론을 위해 실험적 검증이 필요합니다."
        ]

        # 충분한 증거가 있는지 확인
        if total_deg < 10 and db_matched_count == 0 and hub_count == 0:
            insufficient_data_note = (
                "현재 데이터는 특정 생물학적 결론을 지지하기에 불충분합니다. "
                "추가 샘플이나 보완적인 데이터 유형이 필요할 수 있습니다."
            )
        else:
            insufficient_data_note = None

        return {
            "observation": " ".join(observations),
            "supporting_evidence": evidence,
            "interpretation": " ".join(interpretation_text),
            "limitations": limitations,
            "insufficient_data_note": insufficient_data_note,
            "methodology_note": (
                "이 해석은 보수적 원칙을 따릅니다: "
                "인과적 주장 없음, 새로운 조절자나 치료 표적 지정 없음, "
                "분석 제한사항의 명시적 인정."
            )
        }

    def _interpret_gene(self, gene: str, deg_row: pd.Series) -> Dict[str, Any]:
        """Apply full interpretation to a single gene."""
        # Check databases
        db_info = self._check_database_match(gene)

        # Check cancer type specificity
        cancer_type_match = self._check_cancer_type_match(gene)

        # Check TME
        tme_related = self._check_tme_related(gene)

        # Get hub info
        is_hub, hub_score, degree = self._get_hub_info(gene)

        # Get pathway info
        pathway_count, pathway_names = self._get_pathway_info(gene)

        # Build checklist
        checklist = InterpretationChecklist(
            cancer_type_match=cancer_type_match if db_info['db_matched'] else None,
            is_hub=is_hub,
            hub_score=hub_score,
            pathway_position=pathway_names[:100] if pathway_names else None,
            pathway_count=pathway_count,
            in_known_pathway=pathway_count > 0,
            tme_related=tme_related
        )

        # Calculate score
        score = self._calculate_interpretation_score(checklist, db_info['db_matched'])
        checklist.interpretation_score = score

        # Determine confidence
        confidence = self._determine_confidence(score, db_info['db_matched'], is_hub)
        checklist.confidence = confidence

        # Generate tags
        tags = self._generate_tags(checklist, db_info['db_matched'])
        checklist.tags = tags

        # Generate narrative
        deg_info = {
            'direction': deg_row['direction'],
            'log2FC': deg_row['log2FC'],
            'padj': deg_row['padj']
        }
        narrative = self._generate_narrative(gene, checklist, db_info['db_matched'], deg_info)

        return {
            'gene_id': gene,
            'db_info': db_info,
            'checklist': asdict(checklist),
            'narrative': narrative,
            'deg_info': deg_info
        }

    def _run_rag_interpretation(self, top_genes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run gene interpretation for top genes.

        Supports two modes:
        1. RAG Only: Internal VectorDB literature search (Hybrid: Dense + Sparse)
        2. Multi-Source: RAG + External API Context
           - RAG: VectorDB literature search (true RAG)
           - API: Structured data from OncoKB, CIViC, STRING, UniProt, KEGG, Reactome
                  (NOT RAG - this is structured data augmentation)

        Args:
            top_genes: List of gene dictionaries with gene_symbol, log2fc, direction

        Returns:
            Dictionary mapping gene_symbol to interpretation
        """
        if not self.config.get('enable_rag', True):
            self.logger.info("Gene interpretation disabled in config")
            return {}

        # Determine which interpreter to use
        use_multisource = self.config.get('enable_multisource', True) and HAS_MULTISOURCE
        use_external_apis = self.config.get('use_external_apis', True)

        if not use_multisource and not HAS_RAG:
            self.logger.warning("Interpretation module not available - skipping")
            return {}

        try:
            cancer_type = self.config.get('cancer_type', 'breast_cancer')

            if use_multisource:
                # Multi-source: RAG + External API Context
                self.logger.info(f"Using Multi-Source interpreter (RAG + API Context: {use_external_apis})")
                self.rag_interpreter = create_multisource_interpreter(
                    cancer_type=cancer_type,
                    use_llm=True,
                    use_external_apis=use_external_apis
                )
                interpreter_type = "Multi-Source (RAG + External API Context)"
            else:
                # RAG only: VectorDB literature search
                self.logger.info("Using RAG-only interpreter (VectorDB literature search)")
                self.rag_interpreter = create_interpreter(cancer_type)
                interpreter_type = "RAG Only (VectorDB)"

            self.logger.info(f"Running {interpreter_type} for {len(top_genes)} genes...")

            # Run interpretation
            interpretations = self.rag_interpreter.interpret_genes(
                top_genes,
                max_genes=self.config.get('rag_max_genes', 20)
            )

            # Build result dictionary
            result = {}
            for interp in interpretations:
                result[interp.gene_symbol] = interp.to_dict()

            self.logger.info(f"Interpretation complete: {len(result)} genes")

            # Log data sources used
            if use_multisource and hasattr(interp, 'sources_used'):
                sources = set()
                for g in result.values():
                    sources.update(g.get('sources_used', []))
                if sources:
                    self.logger.info(f"Data sources: {', '.join(sources)}")

            return result

        except Exception as e:
            self.logger.error(f"Gene interpretation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def run(self) -> Dict[str, Any]:
        """Execute validation and interpretation.

        Stage 1: Validates DEG/Network/Pathway results against cancer databases
        Stage 2: Validates ML prediction results and cross-validates with DEG findings
        """
        validation_stage = self.config.get('validation_stage', 1)

        # Stage 2: ML Prediction validation
        if validation_stage == 2 or self.config.get('validate_ml_prediction', False):
            return self._run_ml_validation()

        # Stage 1: Original DEG/Network/Pathway validation
        self.logger.info(f"Starting validation for cancer type: {self.config['cancer_type']}")

        # Process each DEG
        interpretations = []
        integrated_rows = []

        for _, row in self.deg_significant.iterrows():
            gene = row['gene_id']
            interp = self._interpret_gene(gene, row)
            interpretations.append(interp)

            # Build integrated row
            integrated_rows.append({
                'gene_id': gene,
                'gene_symbol': interp['db_info'].get('gene_symbol', gene),
                'log2FC': row['log2FC'],
                'padj': row['padj'],
                'direction': row['direction'],
                'is_hub': interp['checklist']['is_hub'],
                'hub_score': interp['checklist']['hub_score'],
                'pathway_count': interp['checklist']['pathway_count'],
                'db_matched': interp['db_info']['db_matched'],
                'db_sources': ';'.join(interp['db_info']['db_sources']),
                'cancer_type_match': interp['checklist']['cancer_type_match'],
                'tme_related': interp['checklist']['tme_related'],
                'interpretation_score': interp['checklist']['interpretation_score'],
                'confidence': interp['checklist']['confidence'],
                'tags': ';'.join(interp['checklist']['tags'])
            })

        # Create DataFrames
        integrated_df = pd.DataFrame(integrated_rows)
        integrated_df = integrated_df.sort_values('interpretation_score', ascending=False)

        # DB matched genes
        db_matched_df = integrated_df[integrated_df['db_matched'] == True].copy()

        # === RAG Interpretation for top genes ===
        # Priority: Hub genes first (from network analysis), then DB-matched genes
        # This ensures RAG validates our analysis-derived hub genes against literature
        top_genes_for_rag = []
        rag_max_genes = self.config.get('rag_max_genes', 20)

        # 1. First, add all hub genes (sorted by hub_score descending)
        hub_genes_df = integrated_df[integrated_df['is_hub'] == True].sort_values('hub_score', ascending=False)
        for _, row in hub_genes_df.iterrows():
            if row['gene_symbol'] and len(top_genes_for_rag) < rag_max_genes:
                top_genes_for_rag.append({
                    'gene_symbol': row['gene_symbol'],
                    'gene_id': row['gene_id'],
                    'log2fc': row['log2FC'],
                    'direction': row['direction'],
                    'padj': row['padj'],
                    'is_hub': True,
                    'hub_score': row['hub_score']
                })

        # 2. If still under limit, add DB-matched non-hub genes
        added_symbols = {g['gene_symbol'] for g in top_genes_for_rag}
        db_matched_non_hub = integrated_df[
            (integrated_df['db_matched'] == True) &
            (integrated_df['is_hub'] == False)
        ].sort_values('interpretation_score', ascending=False)

        for _, row in db_matched_non_hub.iterrows():
            if row['gene_symbol'] and row['gene_symbol'] not in added_symbols and len(top_genes_for_rag) < rag_max_genes:
                top_genes_for_rag.append({
                    'gene_symbol': row['gene_symbol'],
                    'gene_id': row['gene_id'],
                    'log2fc': row['log2FC'],
                    'direction': row['direction'],
                    'padj': row['padj'],
                    'is_hub': False,
                    'hub_score': 0.0
                })
                added_symbols.add(row['gene_symbol'])

        self.logger.info(f"RAG target genes: {len([g for g in top_genes_for_rag if g.get('is_hub')])} hub genes + "
                        f"{len([g for g in top_genes_for_rag if not g.get('is_hub')])} DB-matched genes")

        # Run RAG interpretation
        self.rag_interpretations = self._run_rag_interpretation(top_genes_for_rag)

        # Add RAG interpretation to integrated table
        integrated_df['rag_interpretation'] = integrated_df['gene_symbol'].apply(
            lambda x: self.rag_interpretations.get(x, {}).get('interpretation', '')
        )
        integrated_df['rag_citations'] = integrated_df['gene_symbol'].apply(
            lambda x: len(self.rag_interpretations.get(x, {}).get('citations', []))
        )
        integrated_df['rag_pmids'] = integrated_df['gene_symbol'].apply(
            lambda x: ';'.join(self.rag_interpretations.get(x, {}).get('pmids', []))
        )

        # Save outputs
        self.save_csv(integrated_df, "integrated_gene_table.csv")
        self.save_csv(db_matched_df, "db_matched_genes.csv")

        # Build interpretation report
        matched_interpretations = [i for i in interpretations if i['db_info']['db_matched']]
        unmatched_interpretations = [i for i in interpretations if not i['db_info']['db_matched']]

        # Categorize by confidence
        high_confidence = [i for i in interpretations if i['checklist']['confidence'] == 'high']
        novel_candidates = [i for i in interpretations if i['checklist']['confidence'] == 'novel_candidate']
        requires_validation = [i for i in interpretations if i['checklist']['confidence'] == 'requires_validation']

        # Generate v2.0 structured interpretation report
        v2_interpretation = self._generate_v2_interpretation(
            integrated_df, matched_interpretations, novel_candidates
        )

        report = {
            "cancer_type": self.config["cancer_type"],
            "total_deg_analyzed": len(self.deg_significant),
            "summary": {
                "db_matched_count": len(db_matched_df),
                "db_matched_ratio": len(db_matched_df) / len(self.deg_significant) if len(self.deg_significant) > 0 else 0,
                "hub_genes_count": int(integrated_df['is_hub'].sum()),
                "hub_and_matched": int(((integrated_df['is_hub']) & (integrated_df['db_matched'])).sum()),
                "novel_hub_candidates": int(((integrated_df['is_hub']) & (~integrated_df['db_matched'])).sum()),
                "high_confidence_count": len(high_confidence),
                "novel_candidates_count": len(novel_candidates),
                "requires_validation_count": len(requires_validation)
            },
            # v2.0 Structured Interpretation
            "v2_interpretation": v2_interpretation,
            "matched_genes": [
                {
                    "gene": i['gene_id'],
                    "checklist": i['checklist'],
                    "narrative": i['narrative']
                }
                for i in sorted(matched_interpretations,
                              key=lambda x: x['checklist']['interpretation_score'],
                              reverse=True)[:20]  # Top 20
            ],
            "unmatched_genes": [
                {
                    "gene": i['gene_id'],
                    "checklist": i['checklist'],
                    "narrative": i['narrative'],
                    "note": "requires functional validation"
                }
                for i in sorted(unmatched_interpretations,
                              key=lambda x: x['checklist']['interpretation_score'],
                              reverse=True)[:20]  # Top 20
            ],
            "high_confidence_genes": [i['gene_id'] for i in high_confidence],
            "novel_candidate_genes": [i['gene_id'] for i in novel_candidates],
            "interpretation_principles": [
                "No causal claims - only associations and patterns",
                "Hub status alone does not imply biological importance",
                "Database absence does not mean irrelevance",
                "Pathway enrichment provides context, not causality",
                "TME genes may reflect microenvironment composition",
                "All conclusions require experimental validation"
            ],
            # Gene interpretation section
            "gene_interpretation": {
                "enabled": self.config.get('enable_rag', True) and (HAS_RAG or HAS_MULTISOURCE),
                "mode": "multi-source" if (self.config.get('enable_multisource', True) and HAS_MULTISOURCE) else "rag-only",
                "rag_source": "VectorDB (Hybrid: Dense + Sparse)",
                "api_sources": list(set(
                    source
                    for interp in self.rag_interpretations.values()
                    for source in interp.get('sources_used', [])
                    if source != "Internal VectorDB"
                )) if self.rag_interpretations else [],
                "genes_interpreted": len(self.rag_interpretations),
                "interpretations": self.rag_interpretations
            }
        }

        self.save_json(report, "interpretation_report.json")

        # Save RAG interpretations separately for easier access
        if self.rag_interpretations:
            self.save_json({
                "cancer_type": self.config["cancer_type"],
                "genes_interpreted": len(self.rag_interpretations),
                "interpretations": self.rag_interpretations
            }, "rag_interpretations.json")

        # Log summary
        self.logger.info(f"Validation & Interpretation Complete:")
        self.logger.info(f"  Total DEGs: {len(self.deg_significant)}")
        self.logger.info(f"  DB matched: {len(db_matched_df)} ({len(db_matched_df)/len(self.deg_significant)*100:.1f}%)")
        self.logger.info(f"  High confidence: {len(high_confidence)}")
        self.logger.info(f"  Novel candidates (hub but not in DB): {len(novel_candidates)}")
        self.logger.info(f"  Requires validation: {len(requires_validation)}")
        self.logger.info(f"  RAG interpretations: {len(self.rag_interpretations)}")

        return {
            "total_deg": len(self.deg_significant),
            "db_matched_count": len(db_matched_df),
            "db_matched_ratio": float(len(db_matched_df) / len(self.deg_significant)) if len(self.deg_significant) > 0 else 0,
            "hub_and_matched": int(((integrated_df['is_hub']) & (integrated_df['db_matched'])).sum()),
            "novel_hub_candidates": int(((integrated_df['is_hub']) & (~integrated_df['db_matched'])).sum()),
            "high_confidence_count": len(high_confidence),
            "confidence_distribution": {
                "high": len([i for i in interpretations if i['checklist']['confidence'] == 'high']),
                "medium": len([i for i in interpretations if i['checklist']['confidence'] == 'medium']),
                "low": len([i for i in interpretations if i['checklist']['confidence'] == 'low']),
                "novel_candidate": len(novel_candidates),
                "requires_validation": len(requires_validation)
            },
            "rag_interpretation": {
                "enabled": self.config.get('enable_rag', True) and HAS_RAG,
                "genes_interpreted": len(self.rag_interpretations)
            }
        }

    def _run_ml_validation(self) -> Dict[str, Any]:
        """Execute Stage 2 validation: ML Prediction validation.

        Validates ML prediction results against:
        1. TCGA cancer type characteristics
        2. DEG expression patterns
        3. Known cancer-specific gene signatures
        """
        self.logger.info("=" * 60)
        self.logger.info("Validation Stage 2: ML Prediction Validation")
        self.logger.info("=" * 60)

        if self.cancer_prediction is None:
            self.logger.warning("No ML prediction to validate")
            return {
                "validation_stage": 2,
                "ml_prediction_validated": False,
                "reason": "No ML prediction found"
            }

        predicted_cancer = self.cancer_prediction.get('predicted_cancer', 'Unknown')
        confidence = self.cancer_prediction.get('confidence', 0)
        agreement_ratio = self.cancer_prediction.get('agreement_ratio', 0)

        self.logger.info(f"Predicted cancer type: {predicted_cancer}")
        self.logger.info(f"Confidence: {confidence:.1%}")
        self.logger.info(f"Sample agreement: {agreement_ratio:.1%}")

        # Validation checks
        validation_results = {
            "validation_stage": 2,
            "predicted_cancer": predicted_cancer,
            "confidence": confidence,
            "agreement_ratio": agreement_ratio,
            "checks": {}
        }

        # Check 1: Confidence threshold
        confidence_check = confidence >= 0.7
        validation_results["checks"]["confidence_threshold"] = {
            "passed": confidence_check,
            "value": confidence,
            "threshold": 0.7,
            "message": "High confidence prediction" if confidence_check else "Low confidence - consider alternative diagnoses"
        }

        # Check 2: Sample agreement
        agreement_check = agreement_ratio >= 0.8
        validation_results["checks"]["sample_agreement"] = {
            "passed": agreement_check,
            "value": agreement_ratio,
            "threshold": 0.8,
            "message": "Strong sample agreement" if agreement_check else "Mixed predictions across samples"
        }

        # Check 3: Cancer-type specific gene expression (if DEG data available)
        if self.deg_significant is not None and len(self.deg_significant) > 0:
            cancer_type_genes = self.CANCER_TYPE_GENES.get(predicted_cancer.lower().replace('-', '_').replace(' ', '_'), set())
            if cancer_type_genes:
                # Check if cancer-specific genes are in DEGs
                gene_col = 'gene_symbol' if 'gene_symbol' in self.deg_significant.columns else 'gene_id'
                deg_genes = set(self.deg_significant[gene_col].tolist())
                matching_genes = cancer_type_genes.intersection(deg_genes)
                gene_match_ratio = len(matching_genes) / len(cancer_type_genes) if cancer_type_genes else 0

                validation_results["checks"]["cancer_gene_signature"] = {
                    "passed": gene_match_ratio >= 0.2,
                    "matching_genes": list(matching_genes),
                    "expected_genes": list(cancer_type_genes)[:10],  # Top 10
                    "match_ratio": gene_match_ratio,
                    "message": f"Found {len(matching_genes)}/{len(cancer_type_genes)} cancer-specific genes in DEGs"
                }

        # Check 4: Confusable cancer pairs warning
        confusable_pairs = [
            ({'HNSC', 'LUSC', 'SKCM'}, 'Squamous cell carcinomas'),
            ({'LUAD', 'PAAD'}, 'Adenocarcinomas'),
            ({'COAD', 'STAD'}, 'GI tract cancers'),
            ({'OV', 'UCEC'}, 'Gynecologic cancers')
        ]

        for pair_set, pair_name in confusable_pairs:
            if predicted_cancer in pair_set:
                top_k = self.cancer_prediction.get('top_k_summary', [])
                other_predictions = [p['cancer'] for p in top_k if p['cancer'] != predicted_cancer]
                confusable_detected = any(p in pair_set for p in other_predictions[:2])

                if confusable_detected:
                    validation_results["checks"]["confusable_pair_warning"] = {
                        "passed": False,
                        "pair_name": pair_name,
                        "cancers_in_pair": list(pair_set),
                        "message": f"Warning: {predicted_cancer} is part of a confusable pair ({pair_name}). Consider tissue-specific markers for confirmation."
                    }
                    break

        # Overall validation score
        passed_checks = sum(1 for check in validation_results["checks"].values() if check.get("passed", False))
        total_checks = len(validation_results["checks"])
        validation_results["overall_score"] = passed_checks / total_checks if total_checks > 0 else 0
        validation_results["ml_prediction_validated"] = validation_results["overall_score"] >= 0.5

        # Save validation report
        self.save_json(validation_results, "ml_validation_report.json")

        self.logger.info(f"ML Validation Complete:")
        self.logger.info(f"  Checks passed: {passed_checks}/{total_checks}")
        self.logger.info(f"  Overall score: {validation_results['overall_score']:.1%}")
        self.logger.info(f"  Validated: {validation_results['ml_prediction_validated']}")

        return validation_results

    def validate_outputs(self) -> bool:
        """Validate interpretation outputs."""
        validation_stage = self.config.get('validation_stage', 1)

        # Stage 2 has different output files
        if validation_stage == 2:
            required_files = ["ml_validation_report.json"]
        else:
            required_files = [
                "integrated_gene_table.csv",
                "db_matched_genes.csv",
                "interpretation_report.json"
            ]

        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing output file: {filename}")
                return False

        # Stage 2: Validate ML validation report
        if validation_stage == 2:
            with open(self.output_dir / "ml_validation_report.json", 'r') as f:
                report = json.load(f)
            required_keys = ['validation_stage', 'checks']
            for key in required_keys:
                if key not in report:
                    self.logger.error(f"Missing key in ML validation report: {key}")
                    return False
            return True

        # Stage 1: Validate interpretation report structure
        with open(self.output_dir / "interpretation_report.json", 'r') as f:
            report = json.load(f)

        required_keys = ['summary', 'matched_genes', 'unmatched_genes']
        for key in required_keys:
            if key not in report:
                self.logger.error(f"Missing key in interpretation report: {key}")
                return False

        return True
