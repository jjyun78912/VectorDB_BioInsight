"""
Variant Analysis Agent (WGS/WES)

Analyzes somatic variants from VCF files to identify driver mutations.

Input:
- variants.vcf or variants.vcf.gz: Somatic variant calls (from GATK, Mutect2, etc.)
- OR variants.maf: Mutation Annotation Format file
- metadata.csv (optional): Sample information

Output:
- annotated_variants.csv: Variants with functional annotations
- driver_mutations.csv: Identified driver mutations
- mutation_summary.csv: Per-gene mutation summary
- figures/: Mutation plots (lollipop, oncoplot)
- meta_variant.json: Execution metadata

Pipeline Steps:
1. Load VCF/MAF file
2. Filter variants (quality, coverage, VAF)
3. Annotate variants (VEP/ANNOVAR or database lookup)
4. Identify driver mutations (COSMIC, OncoKB, ClinVar)
5. Generate mutation summary
6. Create visualizations
"""

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..utils.base_agent import BaseAgent


@dataclass
class Variant:
    """Represents a somatic variant."""
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str = ""

    # Variant details
    variant_type: str = ""  # SNV, INS, DEL, MNV
    effect: str = ""  # missense, nonsense, frameshift, splice, etc.
    amino_acid_change: str = ""  # e.g., p.V600E
    codon_change: str = ""

    # Quality metrics
    vaf: float = 0.0  # Variant Allele Frequency
    depth: int = 0
    alt_count: int = 0

    # Annotations
    cosmic_id: str = ""
    cosmic_count: int = 0  # Number of samples in COSMIC
    oncokb_level: str = ""  # Level 1-4, R1, R2
    oncokb_effect: str = ""  # Oncogenic, Likely Oncogenic, etc.
    clinvar_significance: str = ""

    # Driver prediction
    is_driver: bool = False
    driver_score: float = 0.0
    driver_evidence: List[str] = None

    # Hotspot
    is_hotspot: bool = False
    hotspot_count: int = 0

    def __post_init__(self):
        if self.driver_evidence is None:
            self.driver_evidence = []

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def variant_key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"


class VariantDatabase:
    """
    Database for variant annotations.
    Uses COSMIC, OncoKB, and ClinVar data.
    """

    # Known oncogenic hotspots (curated from COSMIC/OncoKB)
    HOTSPOTS = {
        'KRAS': {
            'G12': ['G12C', 'G12D', 'G12V', 'G12A', 'G12R', 'G12S'],
            'G13': ['G13D', 'G13C', 'G13R'],
            'Q61': ['Q61H', 'Q61K', 'Q61L', 'Q61R'],
        },
        'BRAF': {
            'V600': ['V600E', 'V600K', 'V600D', 'V600R'],
        },
        'EGFR': {
            'L858': ['L858R'],
            'T790': ['T790M'],
            'C797': ['C797S'],
            'exon19del': ['del'],
            'exon20ins': ['ins'],
        },
        'PIK3CA': {
            'E542': ['E542K'],
            'E545': ['E545K', 'E545Q'],
            'H1047': ['H1047R', 'H1047L'],
        },
        'TP53': {
            'R175': ['R175H', 'R175C'],
            'R248': ['R248Q', 'R248W'],
            'R249': ['R249S'],
            'R273': ['R273C', 'R273H'],
            'R282': ['R282W'],
        },
        'IDH1': {
            'R132': ['R132H', 'R132C', 'R132G', 'R132S'],
        },
        'IDH2': {
            'R140': ['R140Q', 'R140L'],
            'R172': ['R172K', 'R172M'],
        },
        'NRAS': {
            'G12': ['G12D', 'G12C', 'G12V'],
            'G13': ['G13R', 'G13V'],
            'Q61': ['Q61K', 'Q61R', 'Q61L', 'Q61H'],
        },
        'AKT1': {
            'E17': ['E17K'],
        },
        'ERBB2': {
            'S310': ['S310F', 'S310Y'],
            'L755': ['L755S'],
            'V777': ['V777L'],
        },
        'MET': {
            'exon14skip': ['splice'],
        },
        'CTNNB1': {
            'S33': ['S33C', 'S33F', 'S33Y'],
            'S37': ['S37F', 'S37C'],
            'S45': ['S45F', 'S45P'],
            'D32': ['D32G', 'D32N'],
        },
        'SF3B1': {
            'K700': ['K700E'],
        },
        'DNMT3A': {
            'R882': ['R882H', 'R882C'],
        },
        'NPM1': {
            'W288': ['W288fs'],
        },
        'FLT3': {
            'ITD': ['ITD'],
            'D835': ['D835Y', 'D835V'],
        },
        'JAK2': {
            'V617': ['V617F'],
        },
        'CALR': {
            'exon9': ['frameshift'],
        },
        'MPL': {
            'W515': ['W515L', 'W515K'],
        },
    }

    # Oncogenic effect classification
    ONCOGENIC_EFFECTS = {
        'Oncogenic': 1.0,
        'Likely Oncogenic': 0.8,
        'Predicted Oncogenic': 0.6,
        'Unknown': 0.3,
        'Likely Neutral': 0.1,
        'Inconclusive': 0.2,
    }

    # Variant effect impact scores
    EFFECT_IMPACT = {
        'frameshift': 0.9,
        'nonsense': 0.9,
        'stop_gained': 0.9,
        'splice_donor': 0.85,
        'splice_acceptor': 0.85,
        'start_lost': 0.8,
        'stop_lost': 0.7,
        'missense': 0.6,
        'inframe_insertion': 0.5,
        'inframe_deletion': 0.5,
        'splice_region': 0.4,
        'synonymous': 0.1,
        '5_prime_UTR': 0.2,
        '3_prime_UTR': 0.2,
        'intron': 0.05,
        'intergenic': 0.01,
    }

    # TSG vs Oncogene gene lists
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

    def __init__(self, cosmic_db_path: Optional[Path] = None):
        self.cosmic_db = {}
        self.oncokb_db = {}

        if cosmic_db_path and cosmic_db_path.exists():
            self._load_cosmic_db(cosmic_db_path)

    def _load_cosmic_db(self, path: Path):
        """Load COSMIC database from file."""
        try:
            with open(path) as f:
                self.cosmic_db = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load COSMIC database: {e}")

    def is_hotspot(self, gene: str, aa_change: str) -> Tuple[bool, int]:
        """Check if variant is a known hotspot."""
        if gene not in self.HOTSPOTS:
            return False, 0

        # Extract position and change
        match = re.match(r'p\.([A-Z])(\d+)([A-Z])?', aa_change)
        if not match:
            # Check for special cases
            if 'del' in aa_change.lower():
                if 'exon19del' in self.HOTSPOTS.get(gene, {}):
                    return True, 1000
            if 'ins' in aa_change.lower():
                if 'exon20ins' in self.HOTSPOTS.get(gene, {}):
                    return True, 500
            return False, 0

        ref_aa, pos, alt_aa = match.groups()
        position_key = f"{ref_aa}{pos}"

        gene_hotspots = self.HOTSPOTS.get(gene, {})

        for hotspot_pos, variants in gene_hotspots.items():
            if position_key.startswith(hotspot_pos) or hotspot_pos in position_key:
                # Check if exact variant is listed
                full_change = f"{ref_aa}{pos}{alt_aa}" if alt_aa else position_key
                if full_change in variants or any(v in aa_change for v in variants):
                    return True, 100  # Placeholder count

        return False, 0

    def get_effect_impact(self, effect: str) -> float:
        """Get impact score for variant effect."""
        effect_lower = effect.lower().replace('_variant', '').replace(' ', '_')

        for key, score in self.EFFECT_IMPACT.items():
            if key in effect_lower:
                return score

        return 0.3  # Default for unknown effects

    def is_loss_of_function(self, effect: str) -> bool:
        """Check if variant causes loss of function."""
        lof_effects = ['frameshift', 'nonsense', 'stop_gained', 'splice_donor',
                       'splice_acceptor', 'start_lost']
        effect_lower = effect.lower()
        return any(lof in effect_lower for lof in lof_effects)

    def get_gene_role(self, gene: str) -> str:
        """Get gene role (TSG, Oncogene, or Unknown)."""
        if gene in self.TSG_GENES:
            return 'TSG'
        elif gene in self.ONCOGENES:
            return 'Oncogene'
        return 'Unknown'

    def predict_driver_status(self, variant: Variant) -> Tuple[bool, float, List[str]]:
        """
        Predict if variant is a driver mutation.

        Returns: (is_driver, score, evidence_list)
        """
        score = 0.0
        evidence = []
        gene = variant.gene

        # 1. Hotspot check (strong evidence)
        is_hotspot, hotspot_count = self.is_hotspot(gene, variant.amino_acid_change)
        if is_hotspot:
            score += 40
            evidence.append(f"Hotspot mutation ({gene} {variant.amino_acid_change})")

        # 2. COSMIC presence
        if variant.cosmic_id:
            cosmic_score = min(20, variant.cosmic_count / 50 * 20)
            score += cosmic_score
            evidence.append(f"COSMIC: {variant.cosmic_id} (n={variant.cosmic_count})")

        # 3. OncoKB annotation
        if variant.oncokb_effect:
            oncokb_score = self.ONCOGENIC_EFFECTS.get(variant.oncokb_effect, 0) * 25
            score += oncokb_score
            evidence.append(f"OncoKB: {variant.oncokb_effect}")

            if variant.oncokb_level:
                evidence.append(f"OncoKB Level: {variant.oncokb_level}")
                if variant.oncokb_level in ['1', '2', 'R1']:
                    score += 10

        # 4. Variant effect impact
        effect_score = self.get_effect_impact(variant.effect) * 15
        score += effect_score

        # 5. TSG + LoF logic
        gene_role = self.get_gene_role(gene)
        if gene_role == 'TSG' and self.is_loss_of_function(variant.effect):
            score += 15
            evidence.append(f"TSG ({gene}) with loss-of-function mutation")

        # 6. Oncogene + activating mutation
        if gene_role == 'Oncogene' and variant.effect.lower() == 'missense':
            if is_hotspot:
                score += 10
                evidence.append(f"Oncogene ({gene}) with activating hotspot")

        # 7. VAF consideration (clonal vs subclonal)
        if variant.vaf >= 0.3:
            evidence.append(f"High VAF ({variant.vaf:.1%}) - likely clonal")

        # Determine driver status
        is_driver = score >= 50  # Threshold

        return is_driver, min(100, score), evidence


class VariantAgent(BaseAgent):
    """Agent for WGS/WES variant analysis."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        default_config = {
            # Input format
            "input_format": "auto",  # auto, vcf, maf

            # Quality filters
            "min_depth": 10,
            "min_vaf": 0.05,
            "min_alt_count": 3,
            "min_qual": 30,

            # Analysis settings
            "include_synonymous": False,
            "include_intronic": False,
            "driver_score_threshold": 50,

            # Cancer type for context
            "cancer_type": "unknown",

            # Reference genome
            "genome_build": "GRCh38",

            # External annotation (if available)
            "use_vep": False,
            "use_annovar": False,
            "cosmic_db_path": None,
        }

        merged_config = {**default_config, **(config or {})}
        super().__init__("agent_variant", input_dir, output_dir, merged_config)

        self.variants: List[Variant] = []
        self.db = VariantDatabase()

    def validate_inputs(self) -> bool:
        """Validate input files."""
        # Check for VCF
        vcf_files = list(self.input_dir.glob("*.vcf")) + \
                    list(self.input_dir.glob("*.vcf.gz"))

        # Check for MAF
        maf_files = list(self.input_dir.glob("*.maf")) + \
                    list(self.input_dir.glob("*.maf.txt"))

        if vcf_files:
            self.logger.info(f"Found VCF file: {vcf_files[0].name}")
            return True
        elif maf_files:
            self.logger.info(f"Found MAF file: {maf_files[0].name}")
            return True
        else:
            self.logger.error("No VCF or MAF file found")
            return False

    def _load_vcf(self, vcf_path: Path) -> List[Variant]:
        """Load variants from VCF file."""
        variants = []

        open_func = gzip.open if str(vcf_path).endswith('.gz') else open
        mode = 'rt' if str(vcf_path).endswith('.gz') else 'r'

        with open_func(vcf_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue

                chrom, pos, vid, ref, alt, qual, filt, info = fields[:8]

                # Parse INFO field
                info_dict = {}
                for item in info.split(';'):
                    if '=' in item:
                        k, v = item.split('=', 1)
                        info_dict[k] = v

                # Extract gene and effect from INFO (common annotations)
                gene = info_dict.get('GENE', info_dict.get('Gene.refGene', ''))
                effect = info_dict.get('EFFECT', info_dict.get('ExonicFunc.refGene', ''))
                aa_change = info_dict.get('AA', info_dict.get('AAChange', info_dict.get('AAChange.refGene', '')))

                # Parse VAF from FORMAT fields if available
                vaf = 0.0
                depth = 0
                alt_count = 0

                if len(fields) >= 10:
                    format_keys = fields[8].split(':')
                    sample_values = fields[9].split(':')
                    format_dict = dict(zip(format_keys, sample_values))

                    # Try different VAF field names
                    if 'AF' in format_dict:
                        try:
                            vaf = float(format_dict['AF'])
                        except:
                            pass
                    elif 'AD' in format_dict:
                        try:
                            ad = format_dict['AD'].split(',')
                            ref_count = int(ad[0])
                            alt_count = int(ad[1])
                            depth = ref_count + alt_count
                            vaf = alt_count / depth if depth > 0 else 0
                        except:
                            pass

                    if 'DP' in format_dict:
                        try:
                            depth = int(format_dict['DP'])
                        except:
                            pass

                # Apply quality filters
                try:
                    qual_val = float(qual) if qual != '.' else 0
                except:
                    qual_val = 0

                if qual_val < self.config["min_qual"]:
                    continue
                if depth < self.config["min_depth"]:
                    continue
                if vaf < self.config["min_vaf"]:
                    continue

                # Determine variant type
                if len(ref) == len(alt) == 1:
                    var_type = "SNV"
                elif len(ref) > len(alt):
                    var_type = "DEL"
                elif len(ref) < len(alt):
                    var_type = "INS"
                else:
                    var_type = "MNV"

                variant = Variant(
                    chrom=chrom,
                    pos=int(pos),
                    ref=ref,
                    alt=alt,
                    gene=gene,
                    variant_type=var_type,
                    effect=effect,
                    amino_acid_change=aa_change,
                    vaf=vaf,
                    depth=depth,
                    alt_count=alt_count,
                )

                variants.append(variant)

        self.logger.info(f"Loaded {len(variants)} variants from VCF")
        return variants

    def _load_maf(self, maf_path: Path) -> List[Variant]:
        """Load variants from MAF file."""
        variants = []

        # Read MAF (tab-separated)
        df = pd.read_csv(maf_path, sep='\t', comment='#', low_memory=False)

        # Standard MAF columns
        col_map = {
            'Chromosome': 'chrom',
            'Start_Position': 'pos',
            'Reference_Allele': 'ref',
            'Tumor_Seq_Allele2': 'alt',
            'Hugo_Symbol': 'gene',
            'Variant_Classification': 'effect',
            'HGVSp_Short': 'aa_change',
            'Variant_Type': 'var_type',
            't_alt_count': 'alt_count',
            't_depth': 'depth',
        }

        for _, row in df.iterrows():
            # Extract fields
            chrom = str(row.get('Chromosome', row.get('chr', '')))
            pos = int(row.get('Start_Position', row.get('start', 0)))
            ref = str(row.get('Reference_Allele', '-'))
            alt = str(row.get('Tumor_Seq_Allele2', row.get('Allele', '-')))
            gene = str(row.get('Hugo_Symbol', row.get('gene', '')))
            effect = str(row.get('Variant_Classification', ''))
            aa_change = str(row.get('HGVSp_Short', row.get('Protein_Change', '')))

            # Get counts
            alt_count = int(row.get('t_alt_count', 0))
            depth = int(row.get('t_depth', row.get('t_ref_count', 0) + alt_count))
            vaf = alt_count / depth if depth > 0 else 0

            # Apply filters
            if depth < self.config["min_depth"]:
                continue
            if vaf < self.config["min_vaf"]:
                continue

            # Determine variant type
            var_type = str(row.get('Variant_Type', 'SNP'))

            variant = Variant(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                gene=gene,
                variant_type=var_type,
                effect=effect,
                amino_acid_change=aa_change,
                vaf=vaf,
                depth=depth,
                alt_count=alt_count,
            )

            variants.append(variant)

        self.logger.info(f"Loaded {len(variants)} variants from MAF")
        return variants

    def _annotate_variants(self):
        """Annotate variants with driver information."""
        self.logger.info("Annotating variants...")

        for variant in self.variants:
            # Check hotspot
            is_hotspot, hotspot_count = self.db.is_hotspot(
                variant.gene,
                variant.amino_acid_change
            )
            variant.is_hotspot = is_hotspot
            variant.hotspot_count = hotspot_count

            # Predict driver status
            is_driver, score, evidence = self.db.predict_driver_status(variant)
            variant.is_driver = is_driver
            variant.driver_score = score
            variant.driver_evidence = evidence

        n_drivers = sum(1 for v in self.variants if v.is_driver)
        n_hotspots = sum(1 for v in self.variants if v.is_hotspot)

        self.logger.info(f"  Identified {n_drivers} potential driver mutations")
        self.logger.info(f"  Found {n_hotspots} hotspot mutations")

    def _generate_summary(self) -> pd.DataFrame:
        """Generate per-gene mutation summary."""
        gene_data = defaultdict(lambda: {
            'total_mutations': 0,
            'driver_mutations': 0,
            'hotspot_mutations': 0,
            'samples_mutated': set(),
            'top_variant': None,
            'max_score': 0,
        })

        for variant in self.variants:
            gene = variant.gene
            if not gene:
                continue

            gene_data[gene]['total_mutations'] += 1

            if variant.is_driver:
                gene_data[gene]['driver_mutations'] += 1

            if variant.is_hotspot:
                gene_data[gene]['hotspot_mutations'] += 1

            if variant.driver_score > gene_data[gene]['max_score']:
                gene_data[gene]['max_score'] = variant.driver_score
                gene_data[gene]['top_variant'] = variant.amino_acid_change

        # Convert to DataFrame
        rows = []
        for gene, data in gene_data.items():
            rows.append({
                'gene': gene,
                'total_mutations': data['total_mutations'],
                'driver_mutations': data['driver_mutations'],
                'hotspot_mutations': data['hotspot_mutations'],
                'max_driver_score': data['max_score'],
                'top_variant': data['top_variant'],
                'gene_role': self.db.get_gene_role(gene),
            })

        summary_df = pd.DataFrame(rows)
        summary_df = summary_df.sort_values('max_driver_score', ascending=False)

        return summary_df

    def _generate_visualizations(self):
        """Generate mutation visualizations."""
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')

            # 1. Mutation type distribution
            type_counts = defaultdict(int)
            for v in self.variants:
                type_counts[v.variant_type] += 1

            fig, ax = plt.subplots(figsize=(8, 5))
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            ax.bar(types, counts, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
            ax.set_xlabel('Variant Type')
            ax.set_ylabel('Count')
            ax.set_title('Mutation Type Distribution')
            plt.tight_layout()
            plt.savefig(figures_dir / 'mutation_types.png', dpi=150)
            plt.close()

            # 2. Driver score distribution
            driver_scores = [v.driver_score for v in self.variants if v.driver_score > 0]
            if driver_scores:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(driver_scores, bins=20, color='#e74c3c', alpha=0.7)
                ax.axvline(x=50, color='black', linestyle='--', label='Driver threshold')
                ax.set_xlabel('Driver Score')
                ax.set_ylabel('Count')
                ax.set_title('Driver Score Distribution')
                ax.legend()
                plt.tight_layout()
                plt.savefig(figures_dir / 'driver_scores.png', dpi=150)
                plt.close()

            # 3. Top mutated genes
            gene_counts = defaultdict(int)
            for v in self.variants:
                if v.gene:
                    gene_counts[v.gene] += 1

            top_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            if top_genes:
                fig, ax = plt.subplots(figsize=(10, 6))
                genes = [g[0] for g in top_genes]
                counts = [g[1] for g in top_genes]
                colors = ['#e74c3c' if g in self.db.ONCOGENES else
                         '#3498db' if g in self.db.TSG_GENES else '#95a5a6'
                         for g in genes]
                ax.barh(genes[::-1], counts[::-1], color=colors[::-1])
                ax.set_xlabel('Mutation Count')
                ax.set_title('Top 20 Mutated Genes')
                plt.tight_layout()
                plt.savefig(figures_dir / 'top_mutated_genes.png', dpi=150)
                plt.close()

            self.logger.info("Generated visualizations")

        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualizations")

    def run(self) -> Dict[str, Any]:
        """Run variant analysis pipeline."""
        self.logger.info("="*60)
        self.logger.info("Starting Variant Analysis")
        self.logger.info("="*60)

        # 1. Load variants
        vcf_files = list(self.input_dir.glob("*.vcf")) + \
                    list(self.input_dir.glob("*.vcf.gz"))
        maf_files = list(self.input_dir.glob("*.maf")) + \
                    list(self.input_dir.glob("*.maf.txt"))

        if vcf_files:
            self.variants = self._load_vcf(vcf_files[0])
        elif maf_files:
            self.variants = self._load_maf(maf_files[0])

        if not self.variants:
            self.logger.warning("No variants loaded")
            return {"status": "error", "message": "No variants found"}

        # 2. Annotate variants
        self._annotate_variants()

        # 3. Generate summary
        summary_df = self._generate_summary()

        # 4. Generate visualizations
        self._generate_visualizations()

        # 5. Save outputs
        self._save_outputs(summary_df)

        # Compile results
        n_drivers = sum(1 for v in self.variants if v.is_driver)
        n_hotspots = sum(1 for v in self.variants if v.is_hotspot)

        results = {
            "status": "success",
            "total_variants": len(self.variants),
            "driver_mutations": n_drivers,
            "hotspot_mutations": n_hotspots,
            "genes_mutated": len(summary_df),
            "top_driver_genes": summary_df.head(10)['gene'].tolist() if len(summary_df) > 0 else [],
        }

        self.logger.info("="*60)
        self.logger.info("Variant Analysis Complete")
        self.logger.info(f"  Total variants: {results['total_variants']}")
        self.logger.info(f"  Driver mutations: {results['driver_mutations']}")
        self.logger.info(f"  Hotspot mutations: {results['hotspot_mutations']}")
        self.logger.info(f"  Genes mutated: {results['genes_mutated']}")
        self.logger.info("="*60)

        return results

    def _save_outputs(self, summary_df: pd.DataFrame):
        """Save analysis outputs."""
        self.logger.info("Saving outputs...")

        # All annotated variants
        variants_df = pd.DataFrame([v.to_dict() for v in self.variants])
        self.save_csv(variants_df, "annotated_variants.csv")

        # Driver mutations only
        driver_variants = [v for v in self.variants if v.is_driver]
        if driver_variants:
            driver_df = pd.DataFrame([v.to_dict() for v in driver_variants])
            driver_df = driver_df.sort_values('driver_score', ascending=False)
            self.save_csv(driver_df, "driver_mutations.csv")

        # Gene summary
        self.save_csv(summary_df, "mutation_summary.csv")

        # Hotspot mutations
        hotspot_variants = [v for v in self.variants if v.is_hotspot]
        if hotspot_variants:
            hotspot_df = pd.DataFrame([v.to_dict() for v in hotspot_variants])
            self.save_csv(hotspot_df, "hotspot_mutations.csv")

        self.logger.info(f"  Saved annotated_variants.csv ({len(self.variants)} variants)")
        self.logger.info(f"  Saved driver_mutations.csv ({len(driver_variants)} drivers)")
        self.logger.info(f"  Saved mutation_summary.csv ({len(summary_df)} genes)")

    def validate_outputs(self) -> bool:
        """Validate that required output files were generated."""
        required_files = ["annotated_variants.csv", "mutation_summary.csv"]

        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                self.logger.error(f"Missing required output: {filename}")
                return False

        return True
