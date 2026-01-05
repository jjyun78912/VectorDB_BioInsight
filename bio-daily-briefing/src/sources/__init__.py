"""
Multi-source fetchers for BIO Daily Briefing v2

Sources:
1. FDA - Drug approvals, safety warnings (RSS)
2. ClinicalTrials.gov - Phase 3 results, new trials (API)
3. bioRxiv/medRxiv - Preprints (API)
4. PubMed - Peer-reviewed papers (existing)
"""

from .fda_fetcher import FDAFetcher, FDANews
from .clinicaltrials_fetcher import ClinicalTrialsFetcher, ClinicalTrial
from .biorxiv_fetcher import BioRxivFetcher, Preprint

__all__ = [
    "FDAFetcher",
    "FDANews",
    "ClinicalTrialsFetcher",
    "ClinicalTrial",
    "BioRxivFetcher",
    "Preprint",
]
