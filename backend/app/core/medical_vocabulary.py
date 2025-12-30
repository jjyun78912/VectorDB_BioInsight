"""
Medical Vocabulary Mapper with MeSH Term Support.

Provides standardized disease term mapping for precise literature search.
Uses MeSH (Medical Subject Headings) terminology as primary vocabulary.
"""
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class MeSHTerm:
    """MeSH term with synonyms and search patterns."""
    primary: str  # Official MeSH heading
    mesh_id: str  # MeSH unique ID (e.g., D001289)
    synonyms: list[str] = field(default_factory=list)
    abbreviations: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)  # Terms to NOT match

    def get_all_terms(self) -> list[str]:
        """Get all searchable terms."""
        terms = [self.primary] + self.synonyms + self.abbreviations
        return [t.lower() for t in terms]

    def to_pubmed_query(self, field: str = "tiab") -> str:
        """
        Generate PubMed-style Boolean query.

        Args:
            field: PubMed field tag (tiab=title/abstract, mesh=MeSH)

        Returns:
            Boolean query string
        """
        parts = []

        # MeSH term (highest priority)
        parts.append(f'"{self.primary}"[MeSH]')

        # Title/Abstract terms
        for term in [self.primary] + self.synonyms:
            parts.append(f'"{term}"[{field}]')

        # Abbreviations
        for abbr in self.abbreviations:
            parts.append(f'{abbr}[{field}]')

        query = " OR ".join(parts)

        # Add exclusions
        if self.exclusions:
            exclusion_parts = [f'"{e}"[{field}]' for e in self.exclusions]
            query = f"({query}) NOT ({' OR '.join(exclusion_parts)})"

        return query


# ============== Disease MeSH Vocabulary ==============

DISEASE_MESH_TERMS: dict[str, MeSHTerm] = {
    # Neurological
    "adhd": MeSHTerm(
        primary="Attention Deficit Disorder with Hyperactivity",
        mesh_id="D001289",
        synonyms=[
            "attention deficit hyperactivity disorder",
            "hyperkinetic disorder",
            "attention deficit syndrome"
        ],
        abbreviations=["ADHD", "ADD"],
        related_terms=["hyperactivity", "inattention", "impulsivity"],
        exclusions=[]
    ),

    "alzheimer": MeSHTerm(
        primary="Alzheimer Disease",
        mesh_id="D000544",
        synonyms=[
            "Alzheimer's disease",
            "Alzheimer dementia",
            "senile dementia of Alzheimer type",
            "presenile dementia"
        ],
        abbreviations=["AD"],
        related_terms=["amyloid", "tau", "neurodegeneration", "dementia"],
        exclusions=[]
    ),

    "glioblastoma": MeSHTerm(
        primary="Glioblastoma",
        mesh_id="D005909",
        synonyms=[
            "glioblastoma multiforme",
            "grade IV astrocytoma",
            "GBM tumor"
        ],
        abbreviations=["GBM"],
        related_terms=["brain tumor", "astrocytoma", "glioma"],
        exclusions=[]
    ),

    # Cancers
    "pancreatic_cancer": MeSHTerm(
        primary="Pancreatic Neoplasms",
        mesh_id="D010190",
        synonyms=[
            "pancreatic cancer",
            "pancreatic carcinoma",
            "pancreatic ductal adenocarcinoma",
            "pancreatic adenocarcinoma",
            "cancer of pancreas"
        ],
        abbreviations=["PDAC", "PC"],
        related_terms=["pancreas", "ductal", "exocrine"],
        exclusions=["pancreatitis"]  # Inflammation, not cancer
    ),

    "blood_cancer": MeSHTerm(
        primary="Hematologic Neoplasms",
        mesh_id="D019337",
        synonyms=[
            "blood cancer",
            "hematological malignancy",
            "hematopoietic neoplasms"
        ],
        abbreviations=[],
        related_terms=["leukemia", "lymphoma", "myeloma", "bone marrow"],
        exclusions=[]
    ),

    "breast_cancer": MeSHTerm(
        primary="Breast Neoplasms",
        mesh_id="D001943",
        synonyms=[
            "breast cancer",
            "breast carcinoma",
            "mammary cancer",
            "breast tumor"
        ],
        abbreviations=["BC"],
        related_terms=["mammary", "HER2", "BRCA", "triple negative"],
        exclusions=[]
    ),

    # Endocrine
    "pcos": MeSHTerm(
        primary="Polycystic Ovary Syndrome",
        mesh_id="D011085",
        synonyms=[
            "polycystic ovarian syndrome",
            "Stein-Leventhal syndrome",
            "polycystic ovaries"
        ],
        abbreviations=["PCOS"],
        related_terms=["ovarian", "androgen", "insulin resistance", "anovulation"],
        exclusions=[]
    ),

    "pheochromocytoma": MeSHTerm(
        primary="Pheochromocytoma",
        mesh_id="D010673",
        synonyms=[
            "phaeochromocytoma",
            "adrenal medulla tumor",
            "chromaffin cell tumor"
        ],
        abbreviations=["PHEO", "PCC"],
        related_terms=["catecholamine", "adrenal", "paraganglioma", "VHL", "SDH"],
        exclusions=[]
    ),
}

# Common disease aliases for lookup
DISEASE_ALIASES: dict[str, str] = {
    # ADHD variants
    "attention deficit": "adhd",
    "hyperactivity": "adhd",
    "add": "adhd",

    # Alzheimer variants
    "alzheimer's": "alzheimer",
    "alzheimers": "alzheimer",
    "dementia": "alzheimer",

    # GBM variants
    "gbm": "glioblastoma",
    "brain cancer": "glioblastoma",
    "brain tumor": "glioblastoma",

    # Pancreatic cancer variants
    "pdac": "pancreatic_cancer",
    "pancreas cancer": "pancreatic_cancer",

    # Blood cancer variants
    "leukemia": "blood_cancer",
    "lymphoma": "blood_cancer",
    "myeloma": "blood_cancer",

    # PCOS variants
    "polycystic ovary": "pcos",
    "polycystic ovarian": "pcos",

    # Pheochromocytoma variants
    "pheo": "pheochromocytoma",
    "paraganglioma": "pheochromocytoma",
}


class MedicalVocabulary:
    """
    Medical vocabulary mapper for standardized disease term matching.
    """

    def __init__(self):
        self.mesh_terms = DISEASE_MESH_TERMS
        self.aliases = DISEASE_ALIASES

    def normalize_disease(self, query: str) -> Optional[str]:
        """
        Normalize a disease query to standard key.

        Args:
            query: User query or disease name

        Returns:
            Normalized disease key or None
        """
        query_lower = query.lower().strip()

        # Direct match
        if query_lower in self.mesh_terms:
            return query_lower

        # Alias match
        for alias, disease_key in self.aliases.items():
            if alias in query_lower:
                return disease_key

        # Fuzzy match against MeSH terms
        for key, mesh in self.mesh_terms.items():
            all_terms = mesh.get_all_terms()
            for term in all_terms:
                if term in query_lower or query_lower in term:
                    return key

        return None

    def get_mesh_term(self, disease_key: str) -> Optional[MeSHTerm]:
        """Get MeSH term for a disease key."""
        return self.mesh_terms.get(disease_key.lower())

    def extract_disease_from_query(self, query: str) -> tuple[Optional[str], str]:
        """
        Extract disease term from a query and return normalized form.

        Args:
            query: Full search query

        Returns:
            Tuple of (disease_key, remaining_query)
        """
        query_lower = query.lower()

        # Check each disease term
        for key, mesh in self.mesh_terms.items():
            all_terms = mesh.get_all_terms()
            for term in all_terms:
                if term in query_lower:
                    # Remove disease term from query
                    remaining = re.sub(
                        re.escape(term),
                        "",
                        query,
                        flags=re.IGNORECASE
                    ).strip()
                    return key, remaining

        # Check aliases
        for alias, disease_key in self.aliases.items():
            if alias in query_lower:
                remaining = re.sub(
                    re.escape(alias),
                    "",
                    query,
                    flags=re.IGNORECASE
                ).strip()
                return disease_key, remaining

        return None, query

    def build_search_query(
        self,
        disease_key: str,
        modifiers: list[str] = None,
        exclusions: list[str] = None
    ) -> dict:
        """
        Build a structured search query for a disease.

        Args:
            disease_key: Normalized disease key
            modifiers: Additional search terms (e.g., "treatment", "genetics")
            exclusions: Terms to exclude

        Returns:
            Structured query dict with disease terms and search pattern
        """
        mesh = self.get_mesh_term(disease_key)

        if not mesh:
            return {
                "disease_key": disease_key,
                "primary_terms": [disease_key],
                "synonyms": [],
                "abbreviations": [],
                "modifiers": modifiers or [],
                "exclusions": exclusions or [],
                "pubmed_query": None
            }

        all_exclusions = list(mesh.exclusions)
        if exclusions:
            all_exclusions.extend(exclusions)

        return {
            "disease_key": disease_key,
            "mesh_id": mesh.mesh_id,
            "primary_terms": [mesh.primary],
            "synonyms": mesh.synonyms,
            "abbreviations": mesh.abbreviations,
            "related_terms": mesh.related_terms,
            "modifiers": modifiers or [],
            "exclusions": all_exclusions,
            "pubmed_query": mesh.to_pubmed_query()
        }

    def match_score(self, text: str, disease_key: str) -> dict:
        """
        Calculate match score for text against disease vocabulary.

        Args:
            text: Text to analyze (title, abstract, etc.)
            disease_key: Disease to match against

        Returns:
            Match details with scores
        """
        mesh = self.get_mesh_term(disease_key)
        if not mesh:
            return {"score": 0, "matches": [], "field_type": "none"}

        text_lower = text.lower()
        matches = []

        # Check primary term (highest weight)
        if mesh.primary.lower() in text_lower:
            matches.append({"term": mesh.primary, "type": "primary", "weight": 1.0})

        # Check synonyms
        for syn in mesh.synonyms:
            if syn.lower() in text_lower:
                matches.append({"term": syn, "type": "synonym", "weight": 0.9})

        # Check abbreviations
        for abbr in mesh.abbreviations:
            # Use word boundary matching for abbreviations
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                matches.append({"term": abbr, "type": "abbreviation", "weight": 0.85})

        # Check related terms (lower weight)
        for rel in mesh.related_terms:
            if rel.lower() in text_lower:
                matches.append({"term": rel, "type": "related", "weight": 0.5})

        # Check exclusions (negative score)
        for exc in mesh.exclusions:
            if exc.lower() in text_lower:
                matches.append({"term": exc, "type": "exclusion", "weight": -0.5})

        # Calculate total score
        total_score = sum(m["weight"] for m in matches)
        max_possible = 1.0 + len(mesh.synonyms) * 0.9 + len(mesh.abbreviations) * 0.85
        normalized_score = max(0, min(1, total_score / max_possible)) if max_possible > 0 else 0

        return {
            "score": normalized_score,
            "raw_score": total_score,
            "matches": matches,
            "has_primary": any(m["type"] == "primary" for m in matches),
            "has_exclusion": any(m["type"] == "exclusion" for m in matches)
        }


# Singleton instance
_vocabulary = None

def get_medical_vocabulary() -> MedicalVocabulary:
    """Get or create the medical vocabulary singleton."""
    global _vocabulary
    if _vocabulary is None:
        _vocabulary = MedicalVocabulary()
    return _vocabulary
