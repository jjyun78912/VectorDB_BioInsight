"""
Paper-Specific Chat Agent - Intelligent Q&A for individual papers.

Features:
- Session-based paper isolation
- Intelligent context filtering
- Junk response detection
- Multi-turn conversation support
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .vector_store import BioVectorStore, create_vector_store, SearchResult
from .embeddings import get_embedder
from .config import GOOGLE_API_KEY, GEMINI_MODEL, CHROMA_DIR


class JunkContentValidator:
    """Validates and filters out junk content like author contributions and references."""

    JUNK_KEYWORDS = [
        "writing", "editing", "review", "conceptualization",
        "methodology", "validation", "investigation", "supervision",
        "data curation", "visualization", "funding acquisition",
        "competing interests", "conflict of interest"
    ]

    STOP_WORDS = {
        "what", "is", "the", "are", "how", "does", "do", "can",
        "this", "that", "paper", "study", "research", "finding"
    }

    # If 4+ junk keywords appear, likely author contributions section
    JUNK_KEYWORD_THRESHOLD = 4

    @classmethod
    def is_junk(cls, text: str) -> bool:
        """Check if content is junk (author contributions, etc.)."""
        text_lower = text.lower()
        keyword_count = sum(1 for kw in cls.JUNK_KEYWORDS if kw in text_lower)

        if keyword_count >= cls.JUNK_KEYWORD_THRESHOLD:
            return True

        # Check for reference-style content (e.g., "1. Smith A, Jones B.")
        if re.match(r'^\d+\.\s+[A-Z][a-z]+\s+[A-Z]{1,2}[,.]', text):
            return True

        return False

    @classmethod
    def is_relevant_to_question(cls, content: str, question: str) -> bool:
        """Basic relevance check between content and question."""
        question_lower = question.lower()
        question_terms = set(question_lower.split()) - cls.STOP_WORDS
        content_lower = content.lower()

        # Check if any question term appears in content
        matches = sum(1 for term in question_terms if term in content_lower)

        # At least one key term should match
        return matches >= 1 or len(question_terms) == 0


class ConfidenceEstimator:
    """Estimates confidence scores for LLM-generated answers."""

    BASE_CONFIDENCE = 0.7
    UNCERTAINTY_PENALTY = 0.15
    CITATION_BONUS = 0.05
    MAX_CITATION_BONUS = 0.2

    UNCERTAINTY_PHRASES = [
        "cannot find", "not mentioned", "unclear",
        "may", "might", "possibly"
    ]

    @classmethod
    def estimate(cls, answer: str, context: str) -> float:
        """Estimate confidence in the answer based on characteristics."""
        confidence = cls.BASE_CONFIDENCE

        # Lower confidence if answer indicates uncertainty
        for phrase in cls.UNCERTAINTY_PHRASES:
            if phrase in answer.lower():
                confidence -= cls.UNCERTAINTY_PENALTY
                break  # Only apply penalty once

        # Higher confidence if answer has citations
        citation_count = len(re.findall(r'\[\d+\]', answer))
        confidence += min(citation_count * cls.CITATION_BONUS, cls.MAX_CITATION_BONUS)

        return max(0.0, min(1.0, confidence))


class PromptTemplates:
    """Language-specific prompt templates for paper summarization."""

    QA_SYSTEM_PROMPT = """You are a research assistant. Give BRIEF, helpful answers about this paper.

RULES:
- 2-3 sentences max
- Start with the key point
- Cite sources: [1], [2], [3]
- Synthesize information from multiple sources if needed
- Use simple language

Example: "The study shows X is effective for treating Y [1][2]. Key side effects include A and B [3]." """

    QA_HUMAN_PROMPT = """Context from paper (each source is numbered [Source N]):
{context}

Question: {question}

Answer (remember to cite sources using [1], [2], etc.):"""

    SUMMARIZE_PROMPTS = {
        "en": {
            "system": """You are a research paper summarizer. Analyze the paper content and provide a structured summary.

Output format (JSON):
{{
  "summary": "2-3 sentence overview of what the paper is about and its main contribution",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "methodology": "Brief description of the research approach/methods used"
}}

Be concise and focus on the most important aspects.""",
            "human": """Paper: {title}

Content from paper:
{context}

Provide a structured summary in JSON format:"""
        },
        "ko": {
            "system": """당신은 연구 논문 요약 전문가입니다. 논문 내용을 분석하고 구조화된 요약을 제공하세요.

출력 형식 (JSON):
{{
  "summary": "논문의 주제와 주요 기여에 대한 2-3문장 개요",
  "key_findings": ["핵심 발견 1", "핵심 발견 2", "핵심 발견 3"],
  "methodology": "연구 방법론에 대한 간략한 설명"
}}

중요: 반드시 한국어로 작성하세요. 핵심적인 내용에 집중하고 간결하게 작성하세요.""",
            "human": """논문: {title}

논문 내용:
{context}

JSON 형식으로 구조화된 요약을 한국어로 제공하세요:"""
        }
    }


@dataclass
class PaperSession:
    """A chat session for a specific paper."""
    session_id: str
    paper_title: str
    collection_name: str
    chunks_count: int
    created_at: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


@dataclass
class CitedSource:
    """A source with citation index for inline referencing."""
    citation_index: int  # [1], [2], etc.
    paper_title: str
    section: str
    excerpt: str
    full_content: str  # Full content for expanded view
    relevance_score: float


@dataclass
class AgentResponse:
    """Response from the paper agent."""
    answer: str
    sources: list[dict]
    confidence: float  # 0-1 score
    is_answerable: bool  # Whether the question can be answered from context


class PaperAgent:
    """
    Intelligent agent for paper-specific Q&A.

    - Creates isolated collection per paper session
    - Filters out irrelevant/junk responses
    - Provides confidence scores
    """

    # Keywords that indicate unanswerable questions
    UNANSWERABLE_INDICATORS = [
        "cannot answer", "don't have", "no information",
        "not mentioned", "not discussed", "not provided",
        "does not contain", "outside the scope",
        "author contributions", "funding", "acknowledgment"
    ]

    # Minimum relevance score to consider (0-100 scale, lower threshold to include more results)
    MIN_RELEVANCE_SCORE = 10.0

    def __init__(self, session_id: str, paper_title: str):
        """Initialize agent for a specific paper session."""
        self.session_id = session_id
        self.paper_title = paper_title
        self.collection_name = f"paper_session_{session_id}"

        # Create session-specific vector store
        self.vector_store = BioVectorStore(
            collection_name=self.collection_name,
            persist_directory=CHROMA_DIR / "sessions"
        )

        # LLM for response generation
        self._llm = None

    @property
    def llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not configured")

            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
            )
        return self._llm

    def add_chunks(self, chunks: list) -> int:
        """Add paper chunks to the session store."""
        return self.vector_store.add_chunks(chunks, show_progress=False)

    def query(self, question: str, top_k: int = 5) -> AgentResponse:
        """
        Answer a question about the paper.

        Uses intelligent filtering to:
        1. Filter out low-relevance results
        2. Detect junk content in results
        3. Determine if question is answerable
        """
        # Search for relevant chunks
        results = self.vector_store.search(question, top_k=top_k * 2)  # Get more for filtering

        # Filter results
        filtered_results = self._filter_results(results, question)

        if not filtered_results:
            return AgentResponse(
                answer="I couldn't find relevant information in this paper to answer your question. Try asking about specific topics covered in the paper.",
                sources=[],
                confidence=0.0,
                is_answerable=False
            )

        # Generate answer
        context = self._build_context(filtered_results[:top_k])
        answer, confidence = self._generate_answer(question, context)

        # Check if answer indicates inability to respond
        is_answerable = not any(
            indicator in answer.lower()
            for indicator in self.UNANSWERABLE_INDICATORS
        )

        # Build sources with citation indices for inline referencing
        sources = [
            {
                "citation_index": i + 1,  # 1-based index for [1], [2], etc.
                "paper_title": r.metadata.get("paper_title", self.paper_title),
                "section": r.metadata.get("section", "Unknown"),
                "excerpt": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "full_content": r.content,  # Full content for expanded view
                "relevance_score": r.relevance_score
            }
            for i, r in enumerate(filtered_results[:top_k])
        ]

        return AgentResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            is_answerable=is_answerable
        )

    def _filter_results(self, results: list[SearchResult], question: str) -> list[SearchResult]:
        """
        Filter out low-quality and irrelevant results.

        Steps:
        1. Filter by relevance score and junk content
        2. If nothing passes filters, return top non-junk results as fallback
        """
        # Step 1: Apply strict filters
        filtered = [
            r for r in results
            if r.relevance_score >= self.MIN_RELEVANCE_SCORE
            and not JunkContentValidator.is_junk(r.content)
        ]

        # Step 2: Fallback if all filtered out
        if filtered:
            return filtered

        # Return top 3 non-junk results even if low relevance
        if not results:
            return []

        return [r for r in results[:3] if not JunkContentValidator.is_junk(r.content)]

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results with numbered citations."""
        context_parts = []

        for i, r in enumerate(results, 1):
            section = r.metadata.get("section", "Unknown")
            # Format: [citation_number] (Section Name): Content
            # This helps LLM use correct citation numbers in response
            context_parts.append(f"[Source {i}] Section: {section}\nContent: {r.content}")

        return "\n\n---\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> tuple[str, float]:
        """Generate answer using LLM with confidence estimation."""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", PromptTemplates.QA_SYSTEM_PROMPT),
            ("human", PromptTemplates.QA_HUMAN_PROMPT)
        ])

        chain = prompt | self.llm

        try:
            result = chain.invoke({
                "context": context,
                "question": question
            })

            answer = result.content
            confidence = ConfidenceEstimator.estimate(answer, context)

            return answer, confidence

        except Exception as e:
            return f"Error generating response: {str(e)}", 0.0

    def _gather_summary_context(self) -> str:
        """Gather diverse chunks from different sections for summarization."""
        intro_results = self.vector_store.search("introduction background purpose", top_k=2)
        method_results = self.vector_store.search("methods methodology approach", top_k=2)
        result_results = self.vector_store.search("results findings outcomes", top_k=3)
        conclusion_results = self.vector_store.search("conclusion discussion implications", top_k=2)

        # Combine unique chunks
        all_chunks = []
        seen_content = set()
        for r in intro_results + method_results + result_results + conclusion_results:
            content_key = r.content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                section = r.metadata.get("section", "Unknown")
                all_chunks.append(f"[{section}] {r.content}")

        if not all_chunks:
            return ""

        return "\n\n---\n\n".join(all_chunks[:8])  # Limit to 8 chunks

    def summarize(self, language: str = "en") -> dict:
        """
        Generate AI summary of the paper using the indexed chunks.

        Args:
            language: Output language - "en" for English, "ko" for Korean

        Returns:
            dict with summary, key_findings, methodology
        """
        from langchain_core.prompts import ChatPromptTemplate

        # Get diverse chunks from different sections
        context = self._gather_summary_context()

        if not context:
            no_content_msg = (
                "요약을 생성할 수 없습니다 - 인덱싱된 콘텐츠가 없습니다."
                if language == "ko"
                else "Unable to generate summary - no content indexed."
            )
            return {
                "summary": no_content_msg,
                "key_findings": [],
                "methodology": ""
            }

        # Get language-specific prompts
        prompts = PromptTemplates.SUMMARIZE_PROMPTS.get(language, PromptTemplates.SUMMARIZE_PROMPTS["en"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["system"]),
            ("human", prompts["human"])
        ])

        chain = prompt | self.llm

        try:
            result = chain.invoke({
                "title": self.paper_title,
                "context": context
            })

            return self._parse_summary_response(result.content)

        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "key_findings": [],
                "methodology": ""
            }

    def _parse_summary_response(self, response_text: str) -> dict:
        """Parse JSON response from summarization LLM."""
        import json

        response_text = response_text.strip()

        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        try:
            parsed = json.loads(response_text)
            return {
                "summary": parsed.get("summary", ""),
                "key_findings": parsed.get("key_findings", []),
                "methodology": parsed.get("methodology", "")
            }
        except json.JSONDecodeError:
            # Fallback: return raw response as summary
            return {
                "summary": response_text[:500],
                "key_findings": [],
                "methodology": ""
            }

    def get_session_info(self) -> dict:
        """Get information about this session."""
        return {
            "session_id": self.session_id,
            "paper_title": self.paper_title,
            "collection_name": self.collection_name,
            "chunks_count": self.vector_store.count
        }

    def cleanup(self):
        """Delete the session data."""
        try:
            self.vector_store.reset()
        except Exception as e:
            pass  # Cleanup failures are non-critical


# Session manager
_sessions: dict[str, PaperAgent] = {}


def create_paper_session(paper_title: str) -> str:
    """Create a new paper chat session."""
    session_id = str(uuid.uuid4())[:8]
    agent = PaperAgent(session_id, paper_title)
    _sessions[session_id] = agent
    return session_id


def get_paper_agent(session_id: str) -> Optional[PaperAgent]:
    """Get an existing paper agent by session ID."""
    return _sessions.get(session_id)


def delete_paper_session(session_id: str):
    """Delete a paper session and cleanup."""
    if session_id in _sessions:
        _sessions[session_id].cleanup()
        del _sessions[session_id]
