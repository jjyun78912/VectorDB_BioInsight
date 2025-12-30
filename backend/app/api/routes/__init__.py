"""
API Routes for BioInsight.
"""
from fastapi import APIRouter

from .paper import router as paper_router
from .search import router as search_router
from .chat import router as chat_router
from .graph import router as graph_router
from .crawler import router as crawler_router
from .trends import router as trends_router
from .citations import router as citations_router
from .dynamic_trends import router as dynamic_trends_router
from .research_gaps import router as research_gaps_router

router = APIRouter()

router.include_router(paper_router, prefix="/papers", tags=["papers"])
router.include_router(search_router, prefix="/search", tags=["search"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(graph_router, prefix="/graph", tags=["graph"])
router.include_router(crawler_router, prefix="/crawler", tags=["crawler"])
router.include_router(trends_router, prefix="/trends", tags=["trends"])
router.include_router(citations_router, prefix="/citations", tags=["citations"])
router.include_router(dynamic_trends_router, prefix="/dynamic-trends", tags=["dynamic-trends"])
router.include_router(research_gaps_router, prefix="/research-gaps", tags=["research-gaps"])
