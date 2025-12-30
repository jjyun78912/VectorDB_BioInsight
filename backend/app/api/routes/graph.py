"""
Knowledge Graph API endpoints for 3D visualization.
Supports multiple disease domains with domain-specific keywords.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Set
import sys
from pathlib import Path
import re
from collections import defaultdict

# Add project root to path

router = APIRouter()

# Domain-specific biomedical keywords
DOMAIN_KEYWORDS = {
    "pancreatic_cancer": {
        # Genes & Mutations
        "KRAS", "TP53", "CDKN2A", "SMAD4", "BRCA1", "BRCA2", "ATM", "PALB2",
        "STK11", "MLH1", "MSH2", "PRSS1", "SPINK1", "CTRC", "CPA1",
        # Proteins & Markers
        "CA19-9", "CEA", "MUC1", "MUC4", "mesothelin", "claudin",
        # Pathways
        "Wnt signaling", "TGF-beta", "Notch pathway", "Hedgehog", "NF-kB",
        "PI3K-AKT", "MAPK pathway", "JAK-STAT",
        # Disease terms
        "PDAC", "pancreatic adenocarcinoma", "PanIN", "IPMN", "MCN",
        "metastasis", "chemoresistance", "desmoplasia", "stroma",
        # Treatments
        "FOLFIRINOX", "gemcitabine", "nab-paclitaxel", "Whipple procedure",
        "immunotherapy", "checkpoint inhibitor", "radiation therapy",
        # Clinical
        "survival", "prognosis", "early detection", "screening", "biomarker",
    },
    "blood_cancer": {
        # Genes & Mutations
        "BCR-ABL", "FLT3", "NPM1", "CEBPA", "IDH1", "IDH2", "DNMT3A", "TET2",
        "TP53", "RUNX1", "EZH2", "ASXL1", "JAK2", "CALR", "MPL", "BCL2", "MYC",
        # Types
        "AML", "ALL", "CML", "CLL", "lymphoma", "myeloma", "leukemia",
        "Hodgkin", "non-Hodgkin", "MDS", "MPN",
        # Treatments
        "imatinib", "dasatinib", "venetoclax", "CAR-T", "stem cell transplant",
        "HSCT", "chemotherapy", "targeted therapy", "immunotherapy",
        # Pathways & Mechanisms
        "apoptosis", "BCL2 family", "tyrosine kinase", "epigenetic",
        "minimal residual disease", "MRD", "blast", "cytogenetics",
        # Clinical
        "remission", "relapse", "refractory", "prognosis", "survival",
    },
    "glioblastoma": {
        # Genes & Mutations
        "MGMT", "IDH1", "IDH2", "EGFR", "PTEN", "TP53", "TERT", "ATRX",
        "PDGFRA", "CDK4", "MDM2", "RB1", "NF1", "PIK3CA", "PIK3R1",
        # Markers & Classification
        "EGFRvIII", "1p/19q codeletion", "H3K27M", "G-CIMP",
        "proneural", "classical", "mesenchymal", "neural",
        # Pathways
        "RTK signaling", "PI3K pathway", "Rb pathway", "p53 pathway",
        "Wnt signaling", "VEGF", "angiogenesis", "invasion",
        # Treatments
        "temozolomide", "bevacizumab", "radiation", "tumor treating fields",
        "TTFields", "immunotherapy", "vaccine therapy", "CAR-T",
        # Disease terms
        "GBM", "glioma", "astrocytoma", "blood-brain barrier", "recurrence",
        "pseudoprogression", "edema", "necrosis",
        # Clinical
        "survival", "progression-free", "Karnofsky", "KPS",
    },
    "alzheimer": {
        # Genes & Proteins
        "APP", "PSEN1", "PSEN2", "APOE", "APOE4", "TREM2", "CLU", "CR1",
        "PICALM", "BIN1", "CD33", "SORL1", "ABCA7",
        # Pathology
        "amyloid beta", "Abeta", "tau", "neurofibrillary tangles", "NFT",
        "amyloid plaques", "neurodegeneration", "synaptic loss",
        "neuroinflammation", "microglia", "astrocyte",
        # Biomarkers
        "CSF biomarkers", "PET imaging", "amyloid PET", "tau PET",
        "p-tau", "t-tau", "Abeta42", "NfL", "GFAP",
        # Treatments
        "cholinesterase inhibitor", "donepezil", "memantine",
        "anti-amyloid", "lecanemab", "aducanumab", "immunotherapy",
        # Clinical
        "MCI", "mild cognitive impairment", "dementia", "MMSE",
        "cognitive decline", "memory loss", "early onset", "late onset",
        # Mechanisms
        "oxidative stress", "mitochondrial dysfunction", "autophagy",
        "blood-brain barrier", "vascular", "inflammation",
    },
    "pcos": {
        # Hormones & Markers
        "androgen", "testosterone", "LH", "FSH", "insulin", "SHBG",
        "AMH", "estrogen", "progesterone", "cortisol", "DHEA",
        # Genes
        "INSR", "FSHR", "LHCGR", "CYP17A1", "CYP19A1", "AR", "SRD5A2",
        "DENND1A", "THADA", "FTO", "TCF7L2",
        # Metabolic
        "insulin resistance", "hyperinsulinemia", "glucose intolerance",
        "metabolic syndrome", "obesity", "dyslipidemia", "type 2 diabetes",
        # Reproductive
        "anovulation", "oligomenorrhea", "amenorrhea", "infertility",
        "ovarian cyst", "follicle", "hirsutism", "acne", "alopecia",
        # Treatments
        "metformin", "clomiphene", "letrozole", "oral contraceptives",
        "spironolactone", "GnRH", "IVF", "lifestyle intervention",
        # Clinical
        "Rotterdam criteria", "hyperandrogenism", "polycystic ovaries",
        "menstrual irregularity", "weight management",
        # Complications
        "cardiovascular risk", "endometrial cancer", "sleep apnea",
        "depression", "anxiety", "fatty liver",
    },
}

# Fallback general keywords
GENERAL_KEYWORDS = {
    "mutation", "variant", "germline", "somatic", "expression",
    "prognosis", "diagnosis", "treatment", "therapy", "survival",
    "biomarker", "gene", "protein", "pathway", "signaling",
}


def get_keywords_for_domain(domain: str) -> Set[str]:
    """Get keywords for a specific domain."""
    domain_specific = DOMAIN_KEYWORDS.get(domain, set())
    return domain_specific | GENERAL_KEYWORDS


def create_keyword_map(domain: str) -> Dict[str, str]:
    """Create lowercase mapping for case-insensitive matching."""
    keywords = get_keywords_for_domain(domain)
    return {kw.lower(): kw for kw in keywords}


class GraphNode(BaseModel):
    """Graph node representing a keyword or paper."""
    id: str
    name: str
    type: str  # 'keyword', 'paper', 'gene', 'disease', 'pathway', etc.
    size: float = 1.0
    color: Optional[str] = None
    metadata: Optional[Dict] = None


class GraphLink(BaseModel):
    """Graph link representing a connection."""
    source: str
    target: str
    strength: float = 1.0


class KnowledgeGraphResponse(BaseModel):
    """Knowledge graph response."""
    nodes: List[GraphNode]
    links: List[GraphLink]
    stats: Dict


def extract_keywords(text: str, domain: str) -> Set[str]:
    """Extract biomedical keywords from text for a specific domain."""
    text_lower = text.lower()
    found = set()
    keyword_map = create_keyword_map(domain)

    for kw_lower, kw_original in keyword_map.items():
        # Use word boundary matching for multi-word terms
        if len(kw_lower.split()) > 1:
            if kw_lower in text_lower:
                found.add(kw_original)
        else:
            # Single word - use regex for word boundary
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            if re.search(pattern, text_lower):
                found.add(kw_original)

    return found


def get_node_type(keyword: str, domain: str) -> str:
    """Determine the type of a keyword for coloring."""
    kw_lower = keyword.lower()

    # Genes (usually uppercase short strings or known gene patterns)
    gene_patterns = ["kras", "tp53", "brca", "egfr", "idh", "mgmt", "apoe",
                     "bcr-abl", "flt3", "npm1", "psen", "insr"]
    if keyword.isupper() and len(keyword) <= 10:
        return "gene"
    if any(gene in kw_lower for gene in gene_patterns):
        return "gene"

    # Diseases
    disease_terms = ["cancer", "leukemia", "lymphoma", "myeloma", "glioma",
                     "alzheimer", "dementia", "syndrome", "disease", "disorder"]
    if any(term in kw_lower for term in disease_terms):
        return "disease"

    # Pathways
    pathway_terms = ["pathway", "signaling", "cascade"]
    if any(term in kw_lower for term in pathway_terms):
        return "pathway"

    # Treatments
    treatment_terms = ["therapy", "treatment", "surgery", "transplant",
                       "chemotherapy", "immunotherapy", "radiation"]
    if any(term in kw_lower for term in treatment_terms):
        return "treatment"

    # Biomarkers
    biomarker_terms = ["marker", "biomarker", "antigen", "ca19", "cea"]
    if any(term in kw_lower for term in biomarker_terms):
        return "biomarker"

    # Proteins
    protein_terms = ["protein", "enzyme", "kinase", "receptor"]
    if any(term in kw_lower for term in protein_terms):
        return "protein"

    # Mechanisms
    mechanism_terms = ["apoptosis", "necrosis", "inflammation", "resistance",
                       "metastasis", "angiogenesis", "autophagy"]
    if any(term in kw_lower for term in mechanism_terms):
        return "mechanism"

    return "keyword"


def get_node_color(node_type: str) -> str:
    """Get color for node type - space/universe theme."""
    colors = {
        "paper": "#ff6b6b",      # Red - stars
        "gene": "#ffd93d",       # Yellow/Gold - suns
        "disease": "#6bcb77",    # Green - planets
        "pathway": "#4d96ff",    # Blue - nebulae
        "treatment": "#ff9f43",  # Orange - supernovas
        "biomarker": "#a66cff",  # Purple - galaxies
        "protein": "#45b7d1",    # Cyan - blue stars
        "mechanism": "#f9a8d4",  # Pink - cosmic dust
        "keyword": "#95d5b2",    # Light green - comets
    }
    return colors.get(node_type, "#888888")


@router.get("/", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    domain: str = Query("pancreatic_cancer", description="Disease domain"),
    include_papers: bool = Query(True, description="Include paper nodes"),
    min_connections: int = Query(1, ge=1, description="Minimum connections for a keyword")
):
    """
    Generate a knowledge graph from indexed papers.
    Returns nodes (keywords, papers) and links (co-occurrence relationships).
    Perfect for 3D universe-style visualization.
    """
    try:
        from backend.app.core.vector_store import create_vector_store

        vector_store = create_vector_store(disease_domain=domain)

        if vector_store.count == 0:
            return KnowledgeGraphResponse(
                nodes=[],
                links=[],
                stats={"total_nodes": 0, "total_links": 0, "domain": domain}
            )

        # Get all papers
        papers = vector_store.get_all_papers()

        # Track keyword occurrences and co-occurrences
        keyword_papers: Dict[str, Set[str]] = defaultdict(set)
        paper_keywords: Dict[str, Set[str]] = {}

        # Get chunks for each paper and extract keywords
        collection = vector_store.collection
        all_results = collection.get(include=["documents", "metadatas"])

        # Group chunks by paper
        paper_texts: Dict[str, str] = defaultdict(str)
        for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
            paper_title = meta.get("paper_title", "Unknown")
            paper_texts[paper_title] += " " + doc

        # Extract keywords from each paper
        for paper_title, text in paper_texts.items():
            keywords = extract_keywords(text, domain)
            paper_keywords[paper_title] = keywords

            for kw in keywords:
                keyword_papers[kw].add(paper_title)

        # Build graph
        nodes: List[GraphNode] = []
        links: List[GraphLink] = []
        node_ids: Set[str] = set()

        # Filter keywords by minimum connections
        filtered_keywords = {
            kw: papers_set for kw, papers_set in keyword_papers.items()
            if len(papers_set) >= min_connections
        }

        # Add paper nodes (as central "planets")
        if include_papers:
            for paper in papers:
                paper_id = f"paper_{hash(paper['title']) % 100000}"
                if paper_id not in node_ids:
                    nodes.append(GraphNode(
                        id=paper_id,
                        name=paper["title"][:50] + "..." if len(paper["title"]) > 50 else paper["title"],
                        type="paper",
                        size=2.5,  # Papers are larger
                        color=get_node_color("paper"),
                        metadata={
                            "full_title": paper["title"],
                            "doi": paper.get("doi"),
                            "year": paper.get("year")
                        }
                    ))
                    node_ids.add(paper_id)

        # Add keyword nodes (as "stars" orbiting)
        for keyword, paper_set in filtered_keywords.items():
            kw_id = f"kw_{keyword.replace(' ', '_').replace('-', '_')}"
            node_type = get_node_type(keyword, domain)

            # Size based on occurrence count
            size = 0.5 + len(paper_set) * 0.5

            nodes.append(GraphNode(
                id=kw_id,
                name=keyword,
                type=node_type,
                size=min(size, 5.0),
                color=get_node_color(node_type),
                metadata={"occurrences": len(paper_set), "type": node_type}
            ))
            node_ids.add(kw_id)

            # Link keywords to papers
            if include_papers:
                for paper_title in paper_set:
                    paper_id = f"paper_{hash(paper_title) % 100000}"
                    if paper_id in node_ids:
                        links.append(GraphLink(
                            source=kw_id,
                            target=paper_id,
                            strength=0.3
                        ))

        # Create keyword-keyword links based on co-occurrence
        keyword_list = list(filtered_keywords.keys())
        for i, kw1 in enumerate(keyword_list):
            for kw2 in keyword_list[i+1:]:
                common_papers = keyword_papers[kw1] & keyword_papers[kw2]
                if common_papers:
                    kw1_id = f"kw_{kw1.replace(' ', '_').replace('-', '_')}"
                    kw2_id = f"kw_{kw2.replace(' ', '_').replace('-', '_')}"

                    strength = len(common_papers) / max(len(keyword_papers[kw1]), len(keyword_papers[kw2]))

                    links.append(GraphLink(
                        source=kw1_id,
                        target=kw2_id,
                        strength=strength
                    ))

        # Categorize nodes by type
        type_counts = defaultdict(int)
        for node in nodes:
            type_counts[node.type] += 1

        return KnowledgeGraphResponse(
            nodes=nodes,
            links=links,
            stats={
                "total_nodes": len(nodes),
                "total_links": len(links),
                "total_papers": len(papers),
                "total_keywords": len(filtered_keywords),
                "domain": domain,
                "node_types": dict(type_counts)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keywords")
async def list_keywords(
    domain: str = Query("pancreatic_cancer", description="Disease domain")
):
    """List all extracted keywords with their occurrence counts."""
    try:
        from backend.app.core.vector_store import create_vector_store

        vector_store = create_vector_store(disease_domain=domain)

        if vector_store.count == 0:
            return {"keywords": [], "total": 0, "domain": domain}

        collection = vector_store.collection
        all_results = collection.get(include=["documents", "metadatas"])

        # Group chunks by paper
        paper_texts: Dict[str, str] = defaultdict(str)
        for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
            paper_title = meta.get("paper_title", "Unknown")
            paper_texts[paper_title] += " " + doc

        # Count keywords
        keyword_counts: Dict[str, int] = defaultdict(int)
        for text in paper_texts.values():
            keywords = extract_keywords(text, domain)
            for kw in keywords:
                keyword_counts[kw] += 1

        # Sort by count
        sorted_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "keywords": [
                {
                    "keyword": kw,
                    "count": count,
                    "type": get_node_type(kw, domain),
                    "color": get_node_color(get_node_type(kw, domain))
                }
                for kw, count in sorted_keywords
            ],
            "total": len(sorted_keywords),
            "domain": domain
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_stats(
    domain: str = Query("pancreatic_cancer", description="Disease domain")
):
    """Get statistics about the knowledge graph for a domain."""
    try:
        from backend.app.core.vector_store import create_vector_store
        from backend.app.core.config import PAPERS_DIR

        vector_store = create_vector_store(disease_domain=domain)

        papers_dir = PAPERS_DIR / domain
        paper_count = len(list(papers_dir.glob("*.json"))) if papers_dir.exists() else 0

        return {
            "domain": domain,
            "total_chunks": vector_store.count,
            "total_papers": paper_count,
            "available_keyword_types": list(get_keywords_for_domain(domain))[:20],
            "keyword_categories": [
                "gene", "disease", "pathway", "treatment",
                "biomarker", "protein", "mechanism"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
