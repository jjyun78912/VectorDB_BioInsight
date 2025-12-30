"""
Chat/RAG API endpoints - AI-powered paper Q&A and summarization.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import json
import tempfile
import shutil

# Add project root to path

router = APIRouter()


# ============== Paper Agent Endpoints ==============

class PaperUploadResponse(BaseModel):
    """Response for paper upload to agent."""
    success: bool
    session_id: str
    paper_title: str
    chunks_indexed: int
    message: str


@router.post("/agent/upload", response_model=PaperUploadResponse)
async def upload_paper_to_agent(
    file: UploadFile = File(...),
):
    """
    Upload a PDF and create a dedicated chat agent for it.
    Returns a session_id to use for subsequent queries.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        from backend.app.core.pdf_parser import BioPaperParser
        from backend.app.core.text_splitter import BioPaperSplitter
        from backend.app.core.paper_agent import create_paper_session, get_paper_agent

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # Parse PDF
            parser = BioPaperParser()
            metadata, sections = parser.parse_pdf(tmp_path)

            # Split into chunks
            splitter = BioPaperSplitter()
            chunks = splitter.split_paper(metadata, sections)

            if not chunks:
                return PaperUploadResponse(
                    success=False,
                    session_id="",
                    paper_title=metadata.title or file.filename,
                    chunks_indexed=0,
                    message="Could not extract text from PDF. The file may be image-based."
                )

            # Create agent session
            session_id = create_paper_session(metadata.title or file.filename)
            agent = get_paper_agent(session_id)

            # Index chunks
            chunks_added = agent.add_chunks(chunks)

            return PaperUploadResponse(
                success=True,
                session_id=session_id,
                paper_title=metadata.title or file.filename,
                chunks_indexed=chunks_added,
                message=f"Paper indexed successfully with {chunks_added} searchable sections"
            )

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AgentQueryRequest(BaseModel):
    """Request to query paper agent."""
    session_id: str
    question: str
    top_k: int = 5


class AgentQueryResponse(BaseModel):
    """Response from paper agent."""
    question: str
    answer: str
    sources: List[dict]
    confidence: float
    is_answerable: bool


@router.post("/agent/ask", response_model=AgentQueryResponse)
async def ask_paper_agent(request: AgentQueryRequest):
    """
    Ask a question to a paper-specific agent.
    Uses the session_id from upload to query only that paper.
    """
    try:
        from backend.app.core.paper_agent import get_paper_agent

        agent = get_paper_agent(request.session_id)
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found. Please upload a paper first."
            )

        response = agent.query(request.question, top_k=request.top_k)

        return AgentQueryResponse(
            question=request.question,
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            is_answerable=response.is_answerable
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AgentSessionInfo(BaseModel):
    """Agent session information."""
    session_id: str
    paper_title: str
    chunks_count: int


@router.get("/agent/session/{session_id}", response_model=AgentSessionInfo)
async def get_agent_session(session_id: str):
    """Get information about a paper agent session."""
    from backend.app.core.paper_agent import get_paper_agent

    agent = get_paper_agent(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")

    info = agent.get_session_info()
    return AgentSessionInfo(
        session_id=info["session_id"],
        paper_title=info["paper_title"],
        chunks_count=info["chunks_count"]
    )


@router.delete("/agent/session/{session_id}")
async def delete_agent_session(session_id: str):
    """Delete a paper agent session."""
    from backend.app.core.paper_agent import delete_paper_session, get_paper_agent

    if not get_paper_agent(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    delete_paper_session(session_id)
    return {"success": True, "message": "Session deleted"}


@router.get("/agent/session/{session_id}/debug")
async def debug_agent_session(session_id: str):
    """Debug endpoint to see actual stored chunks."""
    from backend.app.core.paper_agent import get_paper_agent

    agent = get_paper_agent(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get all chunks from vector store
    try:
        collection = agent.vector_store._collection
        all_data = collection.get(include=["documents", "metadatas"])

        chunks_info = []
        for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
            chunks_info.append({
                "index": i,
                "section": meta.get("section", "Unknown"),
                "content_preview": doc[:200] if doc else "EMPTY",
                "content_length": len(doc) if doc else 0
            })

        # Group by section
        section_counts = {}
        for chunk in chunks_info:
            section = chunk["section"]
            section_counts[section] = section_counts.get(section, 0) + 1

        return {
            "session_id": session_id,
            "paper_title": agent.paper_title,
            "total_chunks": len(chunks_info),
            "chunks_by_section": section_counts,
            "sample_chunks": chunks_info[:10]  # First 10 chunks
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "total_chunks": agent.vector_store.count
        }


# ============== Original Endpoints ==============


class ChatRequest(BaseModel):
    """Chat request model."""
    question: str
    domain: str = "pancreatic_cancer"
    section: Optional[str] = None
    top_k: int = 5


class Source(BaseModel):
    """Source reference model with citation index for inline referencing."""
    citation_index: int = 0  # 1-based index for [1], [2], etc.
    paper_title: str
    section: str
    relevance_score: float
    excerpt: str
    full_content: Optional[str] = None  # Full content for expanded view
    pmid: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    question: str
    answer: str
    sources: List[Source]
    domain: str


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    Ask a question using RAG pipeline.
    Returns AI-generated answer with source citations.
    """
    try:
        from backend.app.core.rag_pipeline import create_rag_pipeline

        rag = create_rag_pipeline(
            disease_domain=request.domain,
            top_k=request.top_k
        )

        if rag.vector_store.count == 0:
            return ChatResponse(
                question=request.question,
                answer=f"No documents indexed yet for {request.domain}. Please index some papers first.",
                sources=[],
                domain=request.domain
            )

        response = rag.query(
            question=request.question,
            section=request.section
        )

        sources = [
            Source(
                paper_title=cite.paper_title,
                section=cite.section,
                relevance_score=cite.relevance_score,
                excerpt=cite.content_preview,
                pmid=cite.doi.split("/")[-1] if cite.doi else None
            )
            for cite in response.citations
        ]

        return ChatResponse(
            question=request.question,
            answer=response.answer,
            sources=sources,
            domain=request.domain
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SummarizeRequest(BaseModel):
    """Summarize request model."""
    pmid: str
    domain: str = "pancreatic_cancer"


class PaperSummary(BaseModel):
    """Paper summary model."""
    pmid: str
    title: str
    executive_summary: str
    key_findings: List[str]
    methodology: Optional[str] = None
    conclusions: Optional[str] = None
    keywords: List[str] = []


@router.post("/summarize", response_model=PaperSummary)
async def summarize_paper(request: SummarizeRequest):
    """
    Generate an AI summary of a specific paper.
    """
    try:
        from backend.app.core.config import PAPERS_DIR, GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        papers_dir = PAPERS_DIR / request.domain
        paper_file = papers_dir / f"{request.pmid}.json"

        if not paper_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Paper {request.pmid} not found in {request.domain}"
            )

        with open(paper_file, 'r', encoding='utf-8') as f:
            paper = json.load(f)

        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "")
        full_text = paper.get("full_text", "")

        # Use abstract if no full text
        content = full_text[:8000] if full_text else abstract

        if not content:
            return PaperSummary(
                pmid=request.pmid,
                title=title,
                executive_summary="No content available for summarization.",
                key_findings=[],
                keywords=paper.get("keywords", [])
            )

        # Generate summary using Gemini
        if not GOOGLE_API_KEY:
            # Return basic info without AI summary
            return PaperSummary(
                pmid=request.pmid,
                title=title,
                executive_summary=abstract[:500] if abstract else "No abstract available.",
                key_findings=[],
                methodology=None,
                conclusions=None,
                keywords=paper.get("keywords", [])
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biomedical research assistant. Analyze the following research paper and provide:
1. A concise executive summary (2-3 sentences)
2. Key findings (3-5 bullet points)
3. Methodology summary (1-2 sentences)
4. Main conclusions (1-2 sentences)

Be scientific and precise. Focus on the most important information."""),
            ("human", """Paper Title: {title}

Content:
{content}

Please provide the structured analysis.""")
        ])

        chain = prompt | llm

        result = chain.invoke({
            "title": title,
            "content": content
        })

        # Parse the response
        response_text = result.content

        # Simple parsing - extract sections
        lines = response_text.split("\n")
        executive_summary = ""
        key_findings = []
        methodology = ""
        conclusions = ""

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower_line = line.lower()
            if "executive summary" in lower_line or "summary:" in lower_line:
                current_section = "summary"
                continue
            elif "key finding" in lower_line or "findings:" in lower_line:
                current_section = "findings"
                continue
            elif "methodology" in lower_line or "methods:" in lower_line:
                current_section = "methodology"
                continue
            elif "conclusion" in lower_line:
                current_section = "conclusions"
                continue

            if current_section == "summary":
                executive_summary += line + " "
            elif current_section == "findings":
                if line.startswith(("-", "*", "•", "1", "2", "3", "4", "5")):
                    finding = line.lstrip("-*•0123456789. ")
                    if finding:
                        key_findings.append(finding)
            elif current_section == "methodology":
                methodology += line + " "
            elif current_section == "conclusions":
                conclusions += line + " "

        # Fallback if parsing failed
        if not executive_summary:
            executive_summary = response_text[:500]

        return PaperSummary(
            pmid=request.pmid,
            title=title,
            executive_summary=executive_summary.strip(),
            key_findings=key_findings[:5],
            methodology=methodology.strip() if methodology else None,
            conclusions=conclusions.strip() if conclusions else None,
            keywords=paper.get("keywords", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AbstractSummaryRequest(BaseModel):
    """Request to summarize abstract directly."""
    title: str
    abstract: str


class AbstractSummaryResponse(BaseModel):
    """Abstract summary response."""
    title: str
    summary: str
    key_points: List[str]


@router.post("/summarize-abstract", response_model=AbstractSummaryResponse)
async def summarize_abstract(request: AbstractSummaryRequest):
    """
    Generate a summary from paper title and abstract directly.
    Used for PubMed papers that aren't in local DB.
    """
    try:
        from backend.app.core.config import GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        if not request.abstract:
            return AbstractSummaryResponse(
                title=request.title,
                summary="No abstract available for summarization.",
                key_points=[]
            )

        if not GOOGLE_API_KEY:
            # Return truncated abstract if no API key
            return AbstractSummaryResponse(
                title=request.title,
                summary=request.abstract[:500],
                key_points=[]
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biomedical research expert. Summarize the following paper abstract in 2-3 clear sentences that capture the main purpose, methodology, and findings.

Then list 3-4 key points as bullet points.

Format your response as:
SUMMARY: [your 2-3 sentence summary]

KEY POINTS:
- Point 1
- Point 2
- Point 3"""),
            ("human", """Paper Title: {title}

Abstract: {abstract}""")
        ])

        chain = prompt | llm
        result = chain.invoke({
            "title": request.title,
            "abstract": request.abstract
        })

        response_text = result.content

        # Parse response
        summary = ""
        key_points = []

        lines = response_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("SUMMARY:"):
                current_section = "summary"
                summary = line[8:].strip()
            elif "KEY POINT" in line.upper() or line.upper().startswith("KEY POINTS"):
                current_section = "points"
            elif current_section == "summary" and not line.startswith("-"):
                summary += " " + line
            elif current_section == "points" and line.startswith(("-", "*", "•")):
                point = line.lstrip("-*• ").strip()
                if point:
                    key_points.append(point)

        # Fallback
        if not summary:
            summary = response_text[:400]

        return AbstractSummaryResponse(
            title=request.title,
            summary=summary.strip(),
            key_points=key_points[:4]
        )

    except Exception as e:
        return AbstractSummaryResponse(
            title=request.title,
            summary=f"Error generating summary: {str(e)[:100]}",
            key_points=[]
        )


class AbstractQARequest(BaseModel):
    """Request to ask question about abstract."""
    title: str
    abstract: str
    question: str


class AbstractQAResponse(BaseModel):
    """Response to abstract Q&A."""
    question: str
    answer: str


@router.post("/ask-abstract", response_model=AbstractQAResponse)
async def ask_about_abstract(request: AbstractQARequest):
    """
    Answer questions about a paper based on its abstract.
    Used for PubMed papers that aren't in local DB.
    """
    try:
        from backend.app.core.config import GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        if not request.abstract:
            return AbstractQAResponse(
                question=request.question,
                answer="No abstract available to answer questions."
            )

        if not GOOGLE_API_KEY:
            return AbstractQAResponse(
                question=request.question,
                answer="API key not configured. Unable to answer questions."
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biomedical research expert. Answer the user's question based ONLY on the provided paper abstract.

If the abstract doesn't contain enough information to answer the question, say so clearly.
Keep your answer concise (2-4 sentences) and scientifically accurate."""),
            ("human", """Paper Title: {title}

Abstract: {abstract}

Question: {question}""")
        ])

        chain = prompt | llm
        result = chain.invoke({
            "title": request.title,
            "abstract": request.abstract,
            "question": request.question
        })

        return AbstractQAResponse(
            question=request.question,
            answer=result.content.strip()
        )

    except Exception as e:
        return AbstractQAResponse(
            question=request.question,
            answer=f"Error: {str(e)[:100]}"
        )


class TranslateRequest(BaseModel):
    """Request to translate search query."""
    text: str
    source_lang: str = "auto"  # auto-detect or ko/en
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    """Translation response."""
    original: str
    translated: str
    detected_lang: str
    is_biomedical: bool = True


@router.post("/translate", response_model=TranslateResponse)
async def translate_query(request: TranslateRequest):
    """
    Translate search query from Korean to English for PubMed search.
    Auto-detects Korean text and translates to English with biomedical context.
    """
    import re

    # Check if text contains Korean characters
    def contains_korean(text: str) -> bool:
        return bool(re.search(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', text))

    # If no Korean, return as-is
    if not contains_korean(request.text):
        return TranslateResponse(
            original=request.text,
            translated=request.text,
            detected_lang="en",
            is_biomedical=True
        )

    try:
        from backend.app.core.config import GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        if not GOOGLE_API_KEY:
            return TranslateResponse(
                original=request.text,
                translated=request.text,
                detected_lang="ko",
                is_biomedical=True
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biomedical translation expert. Translate the given Korean search query to English for PubMed search.

Rules:
1. Use proper medical/scientific terminology
2. Keep disease names, gene names, and drug names in their standard English form
3. Return ONLY the translated text, nothing else
4. If it's already in English, return as-is

Examples:
- "췌장암 치료" → "pancreatic cancer treatment"
- "BRCA1 돌연변이" → "BRCA1 mutation"
- "알츠하이머 예방" → "Alzheimer's prevention"
- "당뇨병 인슐린 저항성" → "diabetes insulin resistance"
- "폐암 면역치료" → "lung cancer immunotherapy" """),
            ("human", "{text}")
        ])

        chain = prompt | llm
        result = chain.invoke({"text": request.text})

        translated = result.content.strip()
        # Remove any quotes if the model wrapped the response
        translated = translated.strip('"\'')

        return TranslateResponse(
            original=request.text,
            translated=translated,
            detected_lang="ko",
            is_biomedical=True
        )

    except Exception as e:
        # Fallback: return original text
        return TranslateResponse(
            original=request.text,
            translated=request.text,
            detected_lang="ko",
            is_biomedical=True
        )


class AnalyzeRequest(BaseModel):
    """Paper analysis request."""
    pmid: str
    domain: str = "pancreatic_cancer"
    analysis_type: str = "full"  # full, methods, results, comparison


class PaperAnalysis(BaseModel):
    """Detailed paper analysis."""
    pmid: str
    title: str
    analysis_type: str
    content: str
    highlights: List[str] = []
    related_papers: List[str] = []


@router.post("/analyze", response_model=PaperAnalysis)
async def analyze_paper(request: AnalyzeRequest):
    """
    Perform detailed analysis of a paper.
    """
    try:
        from backend.app.core.config import PAPERS_DIR, GOOGLE_API_KEY
        from backend.app.core.vector_store import create_vector_store
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        papers_dir = PAPERS_DIR / request.domain
        paper_file = papers_dir / f"{request.pmid}.json"

        if not paper_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Paper {request.pmid} not found"
            )

        with open(paper_file, 'r', encoding='utf-8') as f:
            paper = json.load(f)

        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "")
        full_text = paper.get("full_text", "")

        content = full_text[:10000] if full_text else abstract

        if not content:
            return PaperAnalysis(
                pmid=request.pmid,
                title=title,
                analysis_type=request.analysis_type,
                content="No content available for analysis.",
                highlights=[],
                related_papers=[]
            )

        # Analysis prompts based on type
        analysis_prompts = {
            "full": "Provide a comprehensive analysis of this research paper, including its significance, methodology, results, and implications for the field.",
            "methods": "Analyze the methodology used in this paper. Discuss the experimental design, techniques used, sample size, and any limitations.",
            "results": "Analyze the results presented in this paper. What are the main findings? How statistically significant are they? What do they mean?",
            "comparison": "How does this paper compare to other work in the field? What novel contributions does it make? What gaps does it address?"
        }

        prompt_text = analysis_prompts.get(request.analysis_type, analysis_prompts["full"])

        if not GOOGLE_API_KEY:
            return PaperAnalysis(
                pmid=request.pmid,
                title=title,
                analysis_type=request.analysis_type,
                content=abstract[:1000] if abstract else "API key not configured for AI analysis.",
                highlights=[],
                related_papers=[]
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a biomedical research expert. {prompt_text}

Provide your analysis in a clear, scientific manner. Include specific details from the paper.
At the end, list 3-5 key highlights as bullet points."""),
            ("human", """Paper: {title}

Content:
{content}""")
        ])

        chain = prompt | llm
        result = chain.invoke({"title": title, "content": content})

        response_text = result.content

        # Extract highlights
        highlights = []
        lines = response_text.split("\n")
        in_highlights = False
        main_content = []

        for line in lines:
            if "highlight" in line.lower() or "key point" in line.lower():
                in_highlights = True
                continue

            if in_highlights and line.strip().startswith(("-", "*", "•")):
                highlights.append(line.strip().lstrip("-*• "))
            elif not in_highlights:
                main_content.append(line)

        # Find related papers using vector search
        related_papers = []
        try:
            vector_store = create_vector_store(disease_domain=request.domain)
            results = vector_store.search(title + " " + abstract[:200], top_k=5)

            seen_titles = {title}
            for r in results:
                paper_title = r.metadata.get("paper_title", "")
                if paper_title and paper_title not in seen_titles:
                    related_papers.append(paper_title[:80])
                    seen_titles.add(paper_title)

            related_papers = related_papers[:3]
        except:
            pass

        return PaperAnalysis(
            pmid=request.pmid,
            title=title,
            analysis_type=request.analysis_type,
            content="\n".join(main_content).strip(),
            highlights=highlights[:5],
            related_papers=related_papers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
