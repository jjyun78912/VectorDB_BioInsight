"""
RAG interpretation module for RNA-seq pipeline.

Includes:
- GeneRAGInterpreter: Basic RAG with internal VectorDB
- EnhancedGeneInterpreter: RAG + External APIs (OncoKB, CIViC, STRING, UniProt, KEGG, Reactome)
"""
from .gene_interpreter import GeneRAGInterpreter, GeneInterpretation, create_interpreter
from .enhanced_interpreter import (
    EnhancedGeneInterpreter,
    EnhancedGeneInterpretation,
    create_enhanced_interpreter
)

__all__ = [
    # Basic RAG
    "GeneRAGInterpreter",
    "GeneInterpretation",
    "create_interpreter",
    # Enhanced RAG (with external APIs)
    "EnhancedGeneInterpreter",
    "EnhancedGeneInterpretation",
    "create_enhanced_interpreter",
]
