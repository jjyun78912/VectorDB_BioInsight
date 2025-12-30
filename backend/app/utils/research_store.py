"""
Research Asset Storage Module.

Features:
- Save preprocessed text, embeddings, summaries
- Q&A session logging
- Auto-comparison when new papers are added
- Research notes automation
- Export/Import functionality
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import json
import hashlib


@dataclass
class QALogEntry:
    """A single Q&A interaction log."""
    timestamp: str
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    section_filter: Optional[str] = None
    confidence_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QALogEntry":
        return cls(**data)


@dataclass
class PaperAsset:
    """Stored assets for a single paper."""
    paper_id: str
    title: str
    doi: str = ""
    year: str = ""
    indexed_at: str = ""

    # Stored assets
    summary: dict = field(default_factory=dict)
    interpretation: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)

    # Relationships
    similar_papers: list[dict] = field(default_factory=list)
    related_keywords: list[str] = field(default_factory=list)

    # Notes
    research_notes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PaperAsset":
        return cls(**data)


@dataclass
class ResearchSession:
    """A research session with Q&A logs and notes."""
    session_id: str
    disease_domain: str
    created_at: str
    updated_at: str

    # Session data
    qa_logs: list[QALogEntry] = field(default_factory=list)
    search_history: list[dict] = field(default_factory=list)
    bookmarks: list[str] = field(default_factory=list)  # Paper IDs
    notes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["qa_logs"] = [log.to_dict() for log in self.qa_logs]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchSession":
        qa_logs = [QALogEntry.from_dict(log) for log in data.pop("qa_logs", [])]
        session = cls(**data)
        session.qa_logs = qa_logs
        return session


class ResearchStore:
    """
    Persistent storage for research assets.

    Stores:
    - Paper summaries, interpretations, validations
    - Q&A session logs
    - Research notes
    - Cross-paper comparisons
    """

    def __init__(self, disease_domain: str, storage_dir: str = "./data/research_store"):
        """Initialize research store."""
        self.disease_domain = disease_domain
        self.storage_path = Path(storage_dir) / disease_domain
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage files
        self.papers_file = self.storage_path / "papers.json"
        self.sessions_file = self.storage_path / "sessions.json"
        self.index_file = self.storage_path / "index.json"

        # Load existing data
        self._load()

    def _load(self):
        """Load existing data from storage."""
        # Papers
        if self.papers_file.exists():
            data = json.loads(self.papers_file.read_text())
            self.papers = {k: PaperAsset.from_dict(v) for k, v in data.items()}
        else:
            self.papers = {}

        # Sessions
        if self.sessions_file.exists():
            data = json.loads(self.sessions_file.read_text())
            self.sessions = {k: ResearchSession.from_dict(v) for k, v in data.items()}
        else:
            self.sessions = {}

        # Index
        if self.index_file.exists():
            self.index = json.loads(self.index_file.read_text())
        else:
            self.index = {
                "last_updated": "",
                "total_papers": 0,
                "total_qa_logs": 0,
                "keywords": {}
            }

    def _save(self):
        """Save all data to storage."""
        # Papers
        papers_data = {k: v.to_dict() for k, v in self.papers.items()}
        self.papers_file.write_text(json.dumps(papers_data, indent=2, ensure_ascii=False))

        # Sessions
        sessions_data = {k: v.to_dict() for k, v in self.sessions.items()}
        self.sessions_file.write_text(json.dumps(sessions_data, indent=2, ensure_ascii=False))

        # Update and save index
        self.index["last_updated"] = datetime.now().isoformat()
        self.index["total_papers"] = len(self.papers)
        self.index["total_qa_logs"] = sum(len(s.qa_logs) for s in self.sessions.values())
        self.index_file.write_text(json.dumps(self.index, indent=2, ensure_ascii=False))

    def _generate_paper_id(self, title: str, doi: str = "") -> str:
        """Generate unique paper ID."""
        content = f"{title}:{doi}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # ==================== Paper Asset Management ====================

    def save_paper_summary(
        self,
        title: str,
        summary: dict,
        doi: str = "",
        year: str = ""
    ) -> str:
        """Save a paper summary."""
        paper_id = self._generate_paper_id(title, doi)

        if paper_id not in self.papers:
            self.papers[paper_id] = PaperAsset(
                paper_id=paper_id,
                title=title,
                doi=doi,
                year=year,
                indexed_at=datetime.now().isoformat()
            )

        self.papers[paper_id].summary = summary
        self._save()
        return paper_id

    def save_paper_interpretation(self, paper_id: str, interpretation: dict):
        """Save interpretation for a paper."""
        if paper_id in self.papers:
            self.papers[paper_id].interpretation = interpretation
            self._save()

    def save_paper_validation(self, paper_id: str, validation: dict):
        """Save validation result for a paper."""
        if paper_id in self.papers:
            self.papers[paper_id].validation = validation
            self._save()

    def add_research_note(self, paper_id: str, note: str, category: str = "general"):
        """Add a research note to a paper."""
        if paper_id in self.papers:
            self.papers[paper_id].research_notes.append({
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "content": note
            })
            self._save()

    def get_paper_asset(self, paper_id: str) -> Optional[PaperAsset]:
        """Get all stored assets for a paper."""
        return self.papers.get(paper_id)

    def get_paper_by_title(self, title: str) -> Optional[PaperAsset]:
        """Find paper by title (partial match)."""
        title_lower = title.lower()
        for paper in self.papers.values():
            if title_lower in paper.title.lower():
                return paper
        return None

    def list_papers(self) -> list[dict]:
        """List all stored papers with basic info."""
        return [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "doi": p.doi,
                "year": p.year,
                "has_summary": bool(p.summary),
                "has_interpretation": bool(p.interpretation),
                "notes_count": len(p.research_notes)
            }
            for p in self.papers.values()
        ]

    # ==================== Session Management ====================

    def create_session(self) -> str:
        """Create a new research session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        now = datetime.now().isoformat()

        self.sessions[session_id] = ResearchSession(
            session_id=session_id,
            disease_domain=self.disease_domain,
            created_at=now,
            updated_at=now
        )
        self._save()
        return session_id

    def log_qa(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[str] = None,
        section_filter: str = None,
        confidence_score: float = 0.0
    ):
        """Log a Q&A interaction."""
        if session_id not in self.sessions:
            session_id = self.create_session()

        entry = QALogEntry(
            timestamp=datetime.now().isoformat(),
            question=question,
            answer=answer,
            sources=sources or [],
            section_filter=section_filter,
            confidence_score=confidence_score
        )

        self.sessions[session_id].qa_logs.append(entry)
        self.sessions[session_id].updated_at = datetime.now().isoformat()
        self._save()

    def log_search(self, session_id: str, query: str, result_count: int):
        """Log a search query."""
        if session_id in self.sessions:
            self.sessions[session_id].search_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "result_count": result_count
            })
            self._save()

    def add_bookmark(self, session_id: str, paper_id: str):
        """Bookmark a paper in the current session."""
        if session_id in self.sessions:
            if paper_id not in self.sessions[session_id].bookmarks:
                self.sessions[session_id].bookmarks.append(paper_id)
                self._save()

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get a research session."""
        return self.sessions.get(session_id)

    def get_latest_session(self) -> Optional[ResearchSession]:
        """Get the most recent session."""
        if not self.sessions:
            return None
        latest_id = max(self.sessions.keys())
        return self.sessions[latest_id]

    def get_qa_history(self, session_id: str = None) -> list[QALogEntry]:
        """Get Q&A history for a session or all sessions."""
        if session_id:
            session = self.sessions.get(session_id)
            return session.qa_logs if session else []

        # Return all Q&A logs
        all_logs = []
        for session in self.sessions.values():
            all_logs.extend(session.qa_logs)
        return sorted(all_logs, key=lambda x: x.timestamp, reverse=True)

    # ==================== Export/Import ====================

    def export_session(self, session_id: str, output_path: str = None) -> str:
        """Export a session to JSON file."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        output_path = output_path or str(self.storage_path / f"session_{session_id}.json")

        # Include related paper summaries
        export_data = {
            "session": session.to_dict(),
            "papers": {
                pid: self.papers[pid].to_dict()
                for pid in session.bookmarks
                if pid in self.papers
            }
        }

        Path(output_path).write_text(json.dumps(export_data, indent=2, ensure_ascii=False))
        return output_path

    def export_all(self, output_path: str = None) -> str:
        """Export all research data."""
        output_path = output_path or str(self.storage_path / "full_export.json")

        export_data = {
            "domain": self.disease_domain,
            "exported_at": datetime.now().isoformat(),
            "papers": {k: v.to_dict() for k, v in self.papers.items()},
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "index": self.index
        }

        Path(output_path).write_text(json.dumps(export_data, indent=2, ensure_ascii=False))
        return output_path

    def import_data(self, import_path: str):
        """Import research data from JSON file."""
        data = json.loads(Path(import_path).read_text())

        # Import papers
        for paper_id, paper_data in data.get("papers", {}).items():
            if paper_id not in self.papers:
                self.papers[paper_id] = PaperAsset.from_dict(paper_data)

        # Import sessions
        for session_id, session_data in data.get("sessions", {}).items():
            if session_id not in self.sessions:
                self.sessions[session_id] = ResearchSession.from_dict(session_data)

        self._save()

    # ==================== Analytics ====================

    def get_statistics(self) -> dict:
        """Get research store statistics."""
        total_notes = sum(len(p.research_notes) for p in self.papers.values())
        total_qa = sum(len(s.qa_logs) for s in self.sessions.values())

        papers_with_summary = sum(1 for p in self.papers.values() if p.summary)
        papers_with_interpretation = sum(1 for p in self.papers.values() if p.interpretation)

        return {
            "disease_domain": self.disease_domain,
            "total_papers": len(self.papers),
            "papers_with_summary": papers_with_summary,
            "papers_with_interpretation": papers_with_interpretation,
            "total_research_notes": total_notes,
            "total_sessions": len(self.sessions),
            "total_qa_interactions": total_qa,
            "last_updated": self.index.get("last_updated", "Never")
        }

    def find_similar_queries(self, query: str, limit: int = 5) -> list[QALogEntry]:
        """Find similar previous Q&A interactions (simple keyword match)."""
        query_words = set(query.lower().split())
        scored = []

        for session in self.sessions.values():
            for log in session.qa_logs:
                log_words = set(log.question.lower().split())
                overlap = len(query_words & log_words)
                if overlap > 0:
                    scored.append((overlap, log))

        scored.sort(key=lambda x: -x[0])
        return [log for _, log in scored[:limit]]


def create_research_store(disease_domain: str, **kwargs) -> ResearchStore:
    """Create a research store instance."""
    return ResearchStore(disease_domain=disease_domain, **kwargs)
