"""
LLM-based Research Interpretation Module.

Features:
- Novel contribution analysis
- Comparison with existing research
- Future direction suggestions
- Clinical implication extraction
"""
from dataclasses import dataclass, field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import GOOGLE_API_KEY, GEMINI_MODEL
from .vector_store import create_vector_store


@dataclass
class InterpretationReport:
    """LLM-generated interpretation report for a paper."""
    paper_title: str
    doi: str = ""

    # Core interpretations
    novel_contributions: list[str] = field(default_factory=list)
    differences_from_existing: list[str] = field(default_factory=list)
    future_directions: list[str] = field(default_factory=list)
    clinical_implications: list[str] = field(default_factory=list)
    methodological_strengths: list[str] = field(default_factory=list)
    limitations_analysis: list[str] = field(default_factory=list)

    # Comparison with other papers
    related_papers: list[dict] = field(default_factory=list)
    positioning_in_field: str = ""

    # Raw LLM response for reference
    raw_response: str = ""

    def format(self) -> str:
        """Format report for display."""
        output = []
        output.append("=" * 70)
        output.append(f"🔬 Research Interpretation Report")
        output.append(f"📄 {self.paper_title[:60]}...")
        if self.doi:
            output.append(f"   DOI: {self.doi}")
        output.append("=" * 70)

        if self.novel_contributions:
            output.append("\n✨ Novel Contributions (What's New?)")
            output.append("-" * 50)
            for i, item in enumerate(self.novel_contributions, 1):
                output.append(f"  {i}. {item}")

        if self.differences_from_existing:
            output.append("\n🔄 Differences from Existing Research")
            output.append("-" * 50)
            for i, item in enumerate(self.differences_from_existing, 1):
                output.append(f"  {i}. {item}")

        if self.future_directions:
            output.append("\n🚀 Suggested Future Directions")
            output.append("-" * 50)
            for i, item in enumerate(self.future_directions, 1):
                output.append(f"  {i}. {item}")

        if self.clinical_implications:
            output.append("\n🏥 Clinical Implications")
            output.append("-" * 50)
            for i, item in enumerate(self.clinical_implications, 1):
                output.append(f"  {i}. {item}")

        if self.methodological_strengths:
            output.append("\n💪 Methodological Strengths")
            output.append("-" * 50)
            for i, item in enumerate(self.methodological_strengths, 1):
                output.append(f"  {i}. {item}")

        if self.limitations_analysis:
            output.append("\n⚠️ Limitations Analysis")
            output.append("-" * 50)
            for i, item in enumerate(self.limitations_analysis, 1):
                output.append(f"  {i}. {item}")

        if self.positioning_in_field:
            output.append("\n📍 Position in Research Field")
            output.append("-" * 50)
            output.append(f"  {self.positioning_in_field}")

        if self.related_papers:
            output.append("\n📚 Related Papers for Context")
            output.append("-" * 50)
            for paper in self.related_papers:
                output.append(f"  • {paper.get('title', 'Unknown')[:50]}...")
                output.append(f"    Similarity: {paper.get('similarity', 0):.1%}")

        return "\n".join(output)


class ResearchInterpreter:
    """
    Generate LLM-based interpretations of research papers.

    Provides deep analysis including:
    - What's novel about this paper
    - How it differs from existing research
    - Recommended future research directions
    """

    def __init__(
        self,
        disease_domain: str,
        model_name: str = GEMINI_MODEL,
        api_key: str | None = None
    ):
        """Initialize interpreter."""
        self.disease_domain = disease_domain
        self.vector_store = create_vector_store(disease_domain=disease_domain)

        api_key = api_key or GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set.")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )

        self._build_prompts()

    def _build_prompts(self):
        """Build interpretation prompts."""
        self.interpret_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert biomedical research analyst specializing in {disease_domain}.
Your task is to provide a deep, insightful interpretation of a research paper.

Analyze the paper content and provide your response in the following JSON format:

{{
  "NOVEL_CONTRIBUTIONS": [
    "Specific novel finding or contribution 1",
    "Specific novel finding or contribution 2",
    "Up to 5 items"
  ],
  "DIFFERENCES_FROM_EXISTING": [
    "How this differs from previous research 1",
    "How this differs from previous research 2",
    "Up to 4 items"
  ],
  "FUTURE_DIRECTIONS": [
    "Suggested future research direction 1",
    "Suggested future research direction 2",
    "Up to 5 items - be specific and actionable"
  ],
  "CLINICAL_IMPLICATIONS": [
    "Clinical implication 1",
    "Clinical implication 2",
    "Up to 4 items"
  ],
  "METHODOLOGICAL_STRENGTHS": [
    "Strength of the methodology 1",
    "Strength of the methodology 2"
  ],
  "LIMITATIONS_ANALYSIS": [
    "Limitation or concern 1",
    "Limitation or concern 2"
  ],
  "POSITIONING": "2-3 sentences describing where this paper fits in the broader research landscape"
}}

Be specific, cite evidence from the paper, and provide actionable insights."""),
            ("human", """Paper Title: {title}

Paper Content:
{content}

Context from Related Papers:
{context}

Please provide a comprehensive interpretation:""")
        ])

        self.chain = self.interpret_prompt | self.llm | StrOutputParser()

    def interpret_paper(self, paper_title: str) -> InterpretationReport:
        """
        Generate interpretation report for an indexed paper.

        Args:
            paper_title: Title of the paper to interpret

        Returns:
            InterpretationReport with analysis
        """
        # Get paper content
        results = self.vector_store.collection.get(
            where={"paper_title": paper_title},
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            raise ValueError(f"Paper not found: {paper_title}")

        # Combine content
        content = "\n\n".join(results["documents"])
        if len(content) > 25000:
            content = content[:25000] + "\n...[truncated]"

        metadata = results["metadatas"][0] if results["metadatas"] else {}

        # Get context from similar papers
        context = self._get_related_context(paper_title, content[:2000])

        # Generate interpretation
        try:
            response = self.chain.invoke({
                "disease_domain": self.disease_domain,
                "title": paper_title,
                "content": content,
                "context": context["text"]
            })
        except Exception as e:
            return InterpretationReport(
                paper_title=paper_title,
                doi=metadata.get("doi", ""),
                raw_response=f"Error: {str(e)}"
            )

        # Parse response
        report = self._parse_response(response, paper_title, metadata)
        report.related_papers = context["papers"]

        return report

    def _get_related_context(self, exclude_title: str, query_text: str) -> dict:
        """Get context from related papers for comparison."""
        # Search for similar content
        results = self.vector_store.search(query_text, top_k=10)

        # Filter out the target paper and get unique papers
        seen = set()
        related = []
        context_parts = []

        for result in results:
            title = result.metadata.get("paper_title", "")
            if title == exclude_title or title in seen:
                continue
            seen.add(title)

            related.append({
                "title": title,
                "doi": result.metadata.get("doi", ""),
                "similarity": result.relevance_score / 100
            })

            context_parts.append(f"[{title}]: {result.content[:500]}...")

            if len(related) >= 3:
                break

        return {
            "papers": related,
            "text": "\n\n".join(context_parts) if context_parts else "No related papers found."
        }

    def _parse_response(
        self,
        response: str,
        title: str,
        metadata: dict
    ) -> InterpretationReport:
        """Parse LLM response into report."""
        import json
        import re

        report = InterpretationReport(
            paper_title=title,
            doi=metadata.get("doi", ""),
            raw_response=response
        )

        # Try JSON parsing
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                report.novel_contributions = data.get("NOVEL_CONTRIBUTIONS", [])
                report.differences_from_existing = data.get("DIFFERENCES_FROM_EXISTING", [])
                report.future_directions = data.get("FUTURE_DIRECTIONS", [])
                report.clinical_implications = data.get("CLINICAL_IMPLICATIONS", [])
                report.methodological_strengths = data.get("METHODOLOGICAL_STRENGTHS", [])
                report.limitations_analysis = data.get("LIMITATIONS_ANALYSIS", [])
                report.positioning_in_field = data.get("POSITIONING", "")
        except (json.JSONDecodeError, AttributeError):
            # Fallback: treat entire response as positioning
            report.positioning_in_field = response[:500]

        return report

    def compare_papers(self, paper_titles: list[str]) -> str:
        """Compare multiple papers and generate comparison report."""
        if len(paper_titles) < 2:
            raise ValueError("Need at least 2 papers to compare")

        papers_content = []
        for title in paper_titles[:3]:  # Limit to 3 papers
            results = self.vector_store.collection.get(
                where={"paper_title": title},
                include=["documents"]
            )
            if results["ids"]:
                content = "\n".join(results["documents"][:5])  # First 5 chunks
                papers_content.append(f"[{title}]\n{content[:3000]}")

        compare_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at comparing biomedical research papers.
Compare the following papers and provide:
1. Common themes and findings
2. Key differences in approach or conclusions
3. Which paper contributes most to each aspect
4. How they complement each other"""),
            ("human", "{papers}")
        ])

        chain = compare_prompt | self.llm | StrOutputParser()
        return chain.invoke({"papers": "\n\n---\n\n".join(papers_content)})


def create_interpreter(disease_domain: str, **kwargs) -> ResearchInterpreter:
    """Create an interpreter instance."""
    return ResearchInterpreter(disease_domain=disease_domain, **kwargs)
