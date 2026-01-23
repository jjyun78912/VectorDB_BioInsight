"""
Gene interpretation module for RNA-seq pipeline.

Two interpretation modes:

1. RAG Only (GeneRAGInterpreter)
   - Internal VectorDB literature search
   - Hybrid: Dense (PubMedBERT) + Sparse (BM25)

2. Multi-Source (MultiSourceGeneInterpreter)
   - RAG: VectorDB literature search (true RAG)
   - API Context: External biological databases (NOT RAG - structured data)
     → OncoKB, CIViC, STRING, UniProt, KEGG, Reactome

Note: "RAG" = Retrieval-Augmented Generation (vector similarity search)
      External API calls are "Structured Data Augmentation", not RAG.
"""
from .gene_interpreter import GeneRAGInterpreter, GeneInterpretation, create_interpreter
from .enhanced_interpreter import (
    MultiSourceGeneInterpreter,
    MultiSourceGeneInterpretation,
    create_multisource_interpreter
)

__all__ = [
    # RAG Only (VectorDB literature search)
    "GeneRAGInterpreter",
    "GeneInterpretation",
    "create_interpreter",
    # Multi-Source (RAG + External API Context)
    "MultiSourceGeneInterpreter",
    "MultiSourceGeneInterpretation",
    "create_multisource_interpreter",
]
