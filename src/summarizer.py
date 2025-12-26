"""
Paper Summarization Service.

Features:
- Automatic extraction of research purpose, methods, results, conclusions
- Section-by-section summaries
- Structured summary output with citations
"""
from dataclasses import dataclass, field
from pathlib import Path
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import GOOGLE_API_KEY, GEMINI_MODEL
from .pdf_parser import BioPaperParser
from .vector_store import BioVectorStore, create_vector_store


@dataclass
class PaperSummary:
    """Structured summary of a research paper."""
    title: str
    doi: str = ""
    year: str = ""

    # Core sections
    purpose: str = ""           # 연구 목적
    background: str = ""        # 연구 배경/문제 정의
    methods: str = ""           # 방법론
    results: str = ""           # 주요 결과
    conclusions: str = ""       # 결론

    # Detailed content
    key_findings: list[str] = field(default_factory=list)           # 핵심 발견
    specific_recommendations: list[str] = field(default_factory=list)  # 구체적 권고/가이드라인
    clinical_implications: str = ""  # 임상적 의의
    novel_contributions: str = ""    # 새로운 기여점
    limitations: str = ""

    # Section summaries
    section_summaries: dict[str, str] = field(default_factory=dict)

    def format(self) -> str:
        """Format summary for display."""
        output = []
        output.append("=" * 70)
        output.append(f"📄 {self.title}")
        if self.doi:
            output.append(f"   DOI: {self.doi}")
        if self.year:
            output.append(f"   Year: {self.year}")
        output.append("=" * 70)

        output.append("\n🎯 연구 목적 (Purpose)")
        output.append("-" * 40)
        output.append(self.purpose or "N/A")

        if self.background:
            output.append("\n📖 연구 배경 (Background)")
            output.append("-" * 40)
            output.append(self.background)

        output.append("\n🔬 방법론 (Methods)")
        output.append("-" * 40)
        output.append(self.methods or "N/A")

        output.append("\n📊 주요 결과 (Key Results)")
        output.append("-" * 40)
        output.append(self.results or "N/A")

        output.append("\n💡 결론 (Conclusions)")
        output.append("-" * 40)
        output.append(self.conclusions or "N/A")

        if self.key_findings:
            output.append("\n🔑 핵심 발견 (Key Findings)")
            output.append("-" * 40)
            for i, finding in enumerate(self.key_findings, 1):
                output.append(f"  {i}. {finding}")

        if self.specific_recommendations:
            output.append("\n📋 구체적 권고사항/가이드라인 (Specific Recommendations)")
            output.append("-" * 40)
            for i, rec in enumerate(self.specific_recommendations, 1):
                output.append(f"  {i}. {rec}")

        if self.clinical_implications:
            output.append("\n🏥 임상적 의의 (Clinical Implications)")
            output.append("-" * 40)
            output.append(self.clinical_implications)

        if self.novel_contributions:
            output.append("\n✨ 새로운 기여 (Novel Contributions)")
            output.append("-" * 40)
            output.append(self.novel_contributions)

        if self.limitations:
            output.append("\n⚠️ 한계점 (Limitations)")
            output.append("-" * 40)
            output.append(self.limitations)

        if self.section_summaries:
            output.append("\n📑 섹션별 요약 (Section Summaries)")
            output.append("-" * 40)
            for section, summary in self.section_summaries.items():
                output.append(f"\n[{section}]")
                output.append(summary)

        return "\n".join(output)


class PaperSummarizer:
    """
    Service for summarizing research papers.

    Uses LLM to generate structured summaries from indexed papers
    or directly from PDF files.
    """

    def __init__(
        self,
        disease_domain: str | None = None,
        model_name: str = GEMINI_MODEL,
        api_key: str | None = None
    ):
        """
        Initialize summarizer.

        Args:
            disease_domain: Disease domain for vector store access
            model_name: Gemini model name
            api_key: Google API key
        """
        self.disease_domain = disease_domain

        # Initialize vector store if domain specified
        self.vector_store = None
        if disease_domain:
            self.vector_store = create_vector_store(disease_domain=disease_domain)

        # Configure Gemini
        api_key = api_key or GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Add it to .env file.")

        genai.configure(api_key=api_key)

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )

        # Build prompts
        self._build_prompts()

    def _build_prompts(self):
        """Build LLM prompts for summarization."""

        # Full paper summary prompt - Enhanced for detailed extraction
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert biomedical research analyst. Your task is to create a comprehensive, actionable summary that captures both the high-level insights AND the specific, practical details from research papers.

CRITICAL: Go beyond surface-level summaries. Extract SPECIFIC protocols, criteria, algorithms, numeric values, and actionable recommendations that researchers/clinicians can directly use.

Provide your summary in the following JSON structure:

{{
  "PURPOSE": "2-3 sentences: What problem does this paper solve? What gap does it address?",
  "BACKGROUND": "2-3 sentences: Why is this research important? What is the current state of knowledge?",
  "METHODS": "3-4 sentences: What specific approaches, techniques, or frameworks were used? Include specific tools, criteria, or protocols mentioned.",
  "RESULTS": "3-4 sentences: What were the main quantitative/qualitative findings? Include specific numbers, percentages, or comparisons when available.",
  "CONCLUSIONS": "2-3 sentences: What are the main takeaways? How should this change practice or understanding?",
  "KEY_FINDINGS": [
    "Specific finding 1 with concrete details (e.g., Drug X showed 85% response rate in patients with mutation Y)",
    "Specific finding 2",
    "Specific finding 3",
    "Specific finding 4",
    "Specific finding 5 (include up to 7 if important)"
  ],
  "SPECIFIC_RECOMMENDATIONS": [
    "Actionable recommendation 1 (e.g., For Cluster 1A patients, use 68Ga-DOTA-SSA PET/CT as first-line imaging)",
    "Actionable recommendation 2 (e.g., Patients with SDHx mutations should undergo annual screening starting at age 10)",
    "Actionable recommendation 3",
    "Include specific protocols, algorithms, decision criteria, or guidelines mentioned in the paper"
  ],
  "CLINICAL_IMPLICATIONS": "2-3 sentences: How should clinicians apply these findings? What changes in practice are recommended?",
  "NOVEL_CONTRIBUTIONS": "1-2 sentences: What is new or unique about this paper compared to existing literature?",
  "LIMITATIONS": "1-2 sentences describing study limitations, or Not explicitly stated if not mentioned"
}}

IMPORTANT GUIDELINES:
1. Extract SPECIFIC details: drug names, dosages, genetic markers, imaging protocols, diagnostic criteria
2. Include NUMBERS when available: percentages, p-values, confidence intervals, cutoff values
3. Capture DECISION ALGORITHMS: If X, then do Y type recommendations
4. Note COMPARISONS: Method A is superior to Method B for condition C
5. Identify CLASSIFICATION SYSTEMS or CATEGORIES presented in the paper
6. Extract any TABLES or FLOWCHARTS content conceptually
7. Be thorough - a good summary should enable someone to understand the practical applications without reading the full paper"""),
            ("human", """Please create a comprehensive, detailed summary of the following research paper:

Title: {title}

Content:
{content}

Remember: Extract specific, actionable details - not just general statements. Include concrete recommendations, protocols, and criteria that can be directly applied.""")
        ])

        # Section summary prompt
        self.section_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a scientific paper summarization expert.
Summarize the given section content in 2-3 concise sentences.
Focus on the key points and maintain scientific accuracy."""),
            ("human", """Section: {section}

Content:
{content}

Provide a brief summary:""")
        ])

        # Build chains
        self.summary_chain = self.summary_prompt | self.llm | StrOutputParser()
        self.section_chain = self.section_prompt | self.llm | StrOutputParser()

    def summarize_from_vectordb(
        self,
        paper_title: str,
        include_sections: bool = True
    ) -> PaperSummary:
        """
        Summarize a paper that's already indexed in the vector database.

        Args:
            paper_title: Title of the indexed paper
            include_sections: Whether to include section-by-section summaries

        Returns:
            PaperSummary object
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Provide disease_domain.")

        # Get all chunks for this paper
        results = self.vector_store.collection.get(
            where={"paper_title": paper_title},
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            raise ValueError(f"Paper not found: {paper_title}")

        # Organize content by section
        sections = {}
        metadata = results["metadatas"][0] if results["metadatas"] else {}

        for doc, meta in zip(results["documents"], results["metadatas"]):
            section = meta.get("section", "Unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(doc)

        # Combine all content for full summary
        all_content = "\n\n".join([
            f"[{section}]\n" + "\n".join(chunks)
            for section, chunks in sections.items()
        ])

        # Truncate if too long (Gemini context limit)
        if len(all_content) > 30000:
            all_content = all_content[:30000] + "\n...[truncated]"

        # Generate full summary
        summary_text = self.summary_chain.invoke({
            "title": paper_title,
            "content": all_content
        })

        # Parse summary
        summary = self._parse_summary(summary_text, paper_title, metadata)

        # Generate section summaries if requested
        if include_sections:
            for section, chunks in sections.items():
                if section in ["References", "Acknowledgments"]:
                    continue

                section_content = "\n".join(chunks)
                if len(section_content) > 8000:
                    section_content = section_content[:8000] + "..."

                try:
                    section_summary = self.section_chain.invoke({
                        "section": section,
                        "content": section_content
                    })
                    summary.section_summaries[section] = section_summary
                except Exception as e:
                    summary.section_summaries[section] = f"[Error: {str(e)}]"

        return summary

    def summarize_pdf(
        self,
        pdf_path: str | Path,
        include_sections: bool = True
    ) -> PaperSummary:
        """
        Summarize a PDF file directly (without indexing).

        Args:
            pdf_path: Path to the PDF file
            include_sections: Whether to include section-by-section summaries

        Returns:
            PaperSummary object
        """
        # Parse PDF
        parser = BioPaperParser()
        parsed = parser.parse(str(pdf_path))

        if parsed.get("error"):
            raise ValueError(f"Failed to parse PDF: {parsed['error']}")

        title = parsed.get("title", "Unknown")
        sections = parsed.get("sections", {})

        # Combine all content
        all_content = "\n\n".join([
            f"[{section}]\n{content}"
            for section, content in sections.items()
            if section not in ["References", "Acknowledgments"]
        ])

        # Truncate if too long
        if len(all_content) > 30000:
            all_content = all_content[:30000] + "\n...[truncated]"

        # Generate full summary
        summary_text = self.summary_chain.invoke({
            "title": title,
            "content": all_content
        })

        # Parse summary
        metadata = {
            "doi": parsed.get("doi", ""),
            "year": parsed.get("year", "")
        }
        summary = self._parse_summary(summary_text, title, metadata)

        # Generate section summaries if requested
        if include_sections:
            for section, content in sections.items():
                if section in ["References", "Acknowledgments"]:
                    continue

                if len(content) > 8000:
                    content = content[:8000] + "..."

                try:
                    section_summary = self.section_chain.invoke({
                        "section": section,
                        "content": content
                    })
                    summary.section_summaries[section] = section_summary
                except Exception as e:
                    summary.section_summaries[section] = f"[Error: {str(e)}]"

        return summary

    def _parse_summary(
        self,
        summary_text: str,
        title: str,
        metadata: dict
    ) -> PaperSummary:
        """Parse LLM output into PaperSummary object."""
        import json
        import re

        summary = PaperSummary(
            title=title,
            doi=metadata.get("doi", ""),
            year=metadata.get("year", "")
        )

        # Try to parse as JSON first
        try:
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', summary_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', summary_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = None

            if json_str:
                data = json.loads(json_str)
                summary.purpose = data.get("PURPOSE", "")
                summary.background = data.get("BACKGROUND", "")
                summary.methods = data.get("METHODS", "")
                summary.results = data.get("RESULTS", "")
                summary.conclusions = data.get("CONCLUSIONS", "")
                summary.key_findings = data.get("KEY_FINDINGS", [])
                summary.specific_recommendations = data.get("SPECIFIC_RECOMMENDATIONS", [])
                summary.clinical_implications = data.get("CLINICAL_IMPLICATIONS", "")
                summary.novel_contributions = data.get("NOVEL_CONTRIBUTIONS", "")
                summary.limitations = data.get("LIMITATIONS", "Not specified")
                return summary
        except (json.JSONDecodeError, AttributeError):
            pass  # Fall back to text parsing

        # Text-based parsing
        lines = summary_text.split("\n")
        current_section = None
        current_content = []

        # Define section markers (case-insensitive)
        section_markers = {
            "purpose": ["PURPOSE:", "PURPOSE", "**PURPOSE**", "**PURPOSE:**"],
            "methods": ["METHODS:", "METHODS", "**METHODS**", "**METHODS:**", "METHODOLOGY:"],
            "results": ["RESULTS:", "RESULTS", "**RESULTS**", "**RESULTS:**", "FINDINGS:"],
            "conclusions": ["CONCLUSIONS:", "CONCLUSIONS", "**CONCLUSIONS**", "**CONCLUSIONS:**", "CONCLUSION:"],
            "key_findings": ["KEY_FINDINGS:", "KEY FINDINGS:", "**KEY FINDINGS**", "**KEY_FINDINGS:**"],
            "limitations": ["LIMITATIONS:", "LIMITATIONS", "**LIMITATIONS**", "**LIMITATIONS:**"]
        }

        for line in lines:
            line = line.strip()
            line_upper = line.upper()

            # Check for section markers
            found_section = None
            for section, markers in section_markers.items():
                for marker in markers:
                    if line_upper.startswith(marker.upper()):
                        found_section = section
                        # Remove marker from line
                        remaining = line[len(marker):].strip()
                        break
                if found_section:
                    break

            if found_section:
                if current_section:
                    self._set_section(summary, current_section, current_content)
                current_section = found_section
                current_content = [remaining] if remaining else []
            elif line.startswith("-") or line.startswith("•") or line.startswith("*") or line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                # Bullet points or numbered lists
                cleaned = line.lstrip("-•*0123456789. ")
                if cleaned:
                    current_content.append(cleaned)
            elif line:
                current_content.append(line)

        # Set last section
        if current_section:
            self._set_section(summary, current_section, current_content)

        return summary

    def _set_section(self, summary: PaperSummary, section: str, content: list):
        """Set a section in the summary object."""
        if section == "purpose":
            summary.purpose = " ".join(content)
        elif section == "methods":
            summary.methods = " ".join(content)
        elif section == "results":
            summary.results = " ".join(content)
        elif section == "conclusions":
            summary.conclusions = " ".join(content)
        elif section == "key_findings":
            summary.key_findings = content
        elif section == "limitations":
            summary.limitations = " ".join(content)

    def list_indexed_papers(self) -> list[dict]:
        """List all papers available for summarization."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        return self.vector_store.get_all_papers()


def create_summarizer(
    disease_domain: str | None = None,
    **kwargs
) -> PaperSummarizer:
    """Convenience function to create a summarizer."""
    return PaperSummarizer(disease_domain=disease_domain, **kwargs)
