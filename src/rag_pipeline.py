"""
RAG Pipeline for Bio Paper Q&A.

Features:
- Vector search with ChromaDB
- LLM answer generation with Gemini
- Citation tracking with source references
- Context-aware responses for biomedical research
"""
from dataclasses import dataclass, field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .config import GOOGLE_API_KEY, GEMINI_MODEL, TOP_K_RESULTS
from .vector_store import BioVectorStore, SearchResult, create_vector_store


@dataclass
class Citation:
    """A citation reference to a source document."""
    paper_title: str
    section: str
    content_preview: str
    relevance_score: float
    doi: str = ""
    year: str = ""

    def format(self, index: int) -> str:
        """Format citation for display."""
        doi_str = f" (DOI: {self.doi})" if self.doi else ""
        year_str = f" ({self.year})" if self.year else ""
        return f"[{index}] {self.paper_title}{year_str}{doi_str}\n    Section: {self.section}"


@dataclass
class RAGResponse:
    """Response from RAG pipeline with answer and citations."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    query: str = ""
    context_used: str = ""

    def format(self) -> str:
        """Format full response with citations."""
        output = f"{self.answer}\n\n"

        if self.citations:
            output += "─" * 60 + "\n"
            output += "📚 References:\n"
            for i, citation in enumerate(self.citations, 1):
                output += f"{citation.format(i)}\n"

        return output


class BioRAGPipeline:
    """
    RAG Pipeline for biomedical paper Q&A.

    Combines vector search with LLM generation to provide
    accurate, citation-backed answers to research questions.
    """

    def __init__(
        self,
        disease_domain: str,
        model_name: str = GEMINI_MODEL,
        api_key: str | None = None,
        top_k: int = TOP_K_RESULTS,
        temperature: float = 0.3
    ):
        """
        Initialize RAG pipeline.

        Args:
            disease_domain: Disease domain for vector store
            model_name: Gemini model name
            api_key: Google API key (uses env var if None)
            top_k: Number of documents to retrieve
            temperature: LLM temperature (lower = more focused)
        """
        self.disease_domain = disease_domain
        self.top_k = top_k

        # Initialize vector store
        self.vector_store = create_vector_store(disease_domain=disease_domain)

        # Configure Gemini
        api_key = api_key or GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Add it to .env file.")

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )

        # Build RAG chain
        self._build_chain()

    def _build_chain(self):
        """Build the LangChain RAG chain."""

        # System prompt for biomedical Q&A
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biomedical research assistant specialized in {disease_domain}.
Your role is to answer research questions based on the provided scientific literature context.

Guidelines:
1. Base your answers ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Use scientific terminology appropriately
4. When referencing specific findings, indicate which source (e.g., "According to [1]...")
5. Be precise and accurate - this is for research purposes
6. If there are conflicting findings in the sources, mention both perspectives

Context from indexed papers:
{context}"""),
            ("human", "{question}")
        ])

        # Build chain
        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _retrieve(self, query: str, section: str | None = None) -> list[SearchResult]:
        """Retrieve relevant documents from vector store."""
        if section:
            return self.vector_store.search_by_section(query, section, top_k=self.top_k)
        return self.vector_store.search(query, top_k=self.top_k)

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results as context for LLM."""
        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(results, 1):
            paper = result.metadata.get("paper_title", "Unknown Paper")
            section = result.metadata.get("section", "Unknown Section")
            year = result.metadata.get("year", "")

            year_str = f" ({year})" if year else ""
            context_parts.append(
                f"[{i}] Source: {paper}{year_str}\n"
                f"    Section: {section}\n"
                f"    Content: {result.content}\n"
            )

        return "\n".join(context_parts)

    def _create_citations(self, results: list[SearchResult]) -> list[Citation]:
        """Create citation objects from search results."""
        citations = []
        for result in results:
            citations.append(Citation(
                paper_title=result.metadata.get("paper_title", "Unknown"),
                section=result.metadata.get("section", "Unknown"),
                content_preview=result.content[:150] + "...",
                relevance_score=result.relevance_score,
                doi=result.metadata.get("doi", ""),
                year=result.metadata.get("year", "")
            ))
        return citations

    def query(
        self,
        question: str,
        section: str | None = None,
        include_context: bool = False
    ) -> RAGResponse:
        """
        Answer a research question using RAG.

        Args:
            question: The research question
            section: Optional section filter (e.g., "Methods")
            include_context: Whether to include raw context in response

        Returns:
            RAGResponse with answer and citations
        """
        # Retrieve relevant documents
        results = self._retrieve(question, section)

        if not results:
            return RAGResponse(
                answer="죄송합니다. 해당 질문에 관련된 문서를 찾을 수 없습니다. "
                       "다른 키워드로 검색하거나 더 많은 논문을 인덱싱해주세요.",
                query=question
            )

        # Format context
        context = self._format_context(results)

        # Generate answer
        answer = self.chain.invoke({
            "disease_domain": self.disease_domain,
            "context": context,
            "question": question
        })

        # Create citations
        citations = self._create_citations(results)

        return RAGResponse(
            answer=answer,
            citations=citations,
            query=question,
            context_used=context if include_context else ""
        )

    def query_with_followup(
        self,
        question: str,
        previous_context: str = "",
        section: str | None = None
    ) -> RAGResponse:
        """
        Answer a follow-up question with conversation context.

        Args:
            question: The follow-up question
            previous_context: Context from previous Q&A
            section: Optional section filter

        Returns:
            RAGResponse with answer and citations
        """
        # Combine with previous context for better understanding
        enhanced_question = question
        if previous_context:
            enhanced_question = f"Previous context: {previous_context}\n\nCurrent question: {question}"

        return self.query(enhanced_question, section)


def create_rag_pipeline(
    disease_domain: str,
    **kwargs
) -> BioRAGPipeline:
    """Convenience function to create a RAG pipeline."""
    return BioRAGPipeline(disease_domain=disease_domain, **kwargs)
