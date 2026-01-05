"""
ClinicalTrials.gov Fetcher

Fetches Phase 3 results, new trials, and trial updates from ClinicalTrials.gov API v2.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import httpx


@dataclass
class ClinicalTrial:
    """Represents a clinical trial."""
    nct_id: str
    title: str
    phase: str
    status: str
    conditions: List[str]
    interventions: List[str]
    sponsor: str
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    results_date: Optional[str] = None
    enrollment: int = 0
    brief_summary: str = ""
    primary_outcome: str = ""
    has_results: bool = False
    result_type: str = ""  # positive, negative, mixed, pending
    priority: int = 0

    def to_dict(self) -> dict:
        return {
            "source": "ClinicalTrials",
            "type": self.result_type or f"phase_{self.phase}",
            "nct_id": self.nct_id,
            "title": self.title,
            "phase": self.phase,
            "status": self.status,
            "conditions": self.conditions,
            "interventions": self.interventions,
            "sponsor": self.sponsor,
            "start_date": self.start_date,
            "completion_date": self.completion_date,
            "results_date": self.results_date,
            "enrollment": self.enrollment,
            "has_results": self.has_results,
            "result_type": self.result_type,
            "priority": self.priority,
            "link": f"https://clinicaltrials.gov/study/{self.nct_id}",
        }


class ClinicalTrialsFetcher:
    """Fetches clinical trial data from ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    # Big pharma sponsors for priority scoring
    BIG_PHARMA = [
        "Pfizer", "Novartis", "Roche", "Eli Lilly", "Novo Nordisk",
        "Merck", "Bristol-Myers Squibb", "AstraZeneca", "Johnson & Johnson",
        "Sanofi", "GSK", "GlaxoSmithKline", "AbbVie", "Gilead",
        "Amgen", "Moderna", "BioNTech", "Regeneron", "Vertex",
    ]

    # Hot conditions for priority scoring
    HOT_CONDITIONS = [
        "obesity", "diabetes", "alzheimer", "cancer", "melanoma",
        "breast cancer", "lung cancer", "lymphoma", "leukemia",
        "solid tumor", "covid", "sars-cov-2",
    ]

    # Hot drug keywords
    HOT_DRUGS = [
        "semaglutide", "tirzepatide", "glp-1", "car-t", "car t",
        "crispr", "mrna", "adc", "antibody-drug conjugate",
        "bispecific", "pd-1", "pd-l1", "checkpoint",
    ]

    def __init__(self):
        self.timeout = httpx.Timeout(60.0)

    async def fetch_phase3_with_results(self, days: int = 30) -> List[ClinicalTrial]:
        """
        Fetch Phase 3 trials that have results posted recently.

        Args:
            days: Look back period for results posting

        Returns:
            List of ClinicalTrial objects with results
        """
        print(f"\n[ClinicalTrials - Phase 3 결과 (최근 {days}일)]")

        # Query for completed Phase 3 studies with results
        # Using simpler query without date range filter (API v2 format)
        params = {
            "query.term": "AREA[Phase]PHASE3 AND AREA[OverallStatus]COMPLETED AND AREA[ResultsFirstPostDate]RANGE[MIN,MAX]",
            "pageSize": 50,
            "sort": "ResultsFirstPostDate:desc",
        }

        trials = await self._fetch_studies(params)
        print(f"  Phase 3 결과 발표: {len(trials)}건")
        return trials

    async def fetch_new_phase3_trials(self, days: int = 14) -> List[ClinicalTrial]:
        """
        Fetch newly started Phase 3 trials.

        Args:
            days: Look back period for new trials

        Returns:
            List of new ClinicalTrial objects
        """
        print(f"\n[ClinicalTrials - 신규 Phase 3 (최근 {days}일)]")

        # Query for recruiting Phase 3 studies sorted by start date
        params = {
            "query.term": "AREA[Phase]PHASE3 AND AREA[OverallStatus]RECRUITING",
            "pageSize": 30,
            "sort": "StudyFirstPostDate:desc",
        }

        trials = await self._fetch_studies(params)
        print(f"  신규 Phase 3 등록: {len(trials)}건")
        return trials

    async def fetch_terminated_trials(self, days: int = 30) -> List[ClinicalTrial]:
        """
        Fetch recently terminated or suspended trials (potential failures).

        Args:
            days: Look back period

        Returns:
            List of terminated/suspended ClinicalTrial objects
        """
        print(f"\n[ClinicalTrials - 중단/실패 (최근 {days}일)]")

        # Query for terminated Phase 3 studies
        params = {
            "query.term": "AREA[Phase]PHASE3 AND (AREA[OverallStatus]TERMINATED OR AREA[OverallStatus]SUSPENDED)",
            "pageSize": 20,
            "sort": "LastUpdatePostDate:desc",
        }

        trials = await self._fetch_studies(params)

        for trial in trials:
            trial.result_type = "terminated"

        print(f"  중단/실패 임상: {len(trials)}건")
        return trials

    async def _fetch_studies(self, params: Dict) -> List[ClinicalTrial]:
        """Fetch studies from API with given parameters."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.BASE_URL, params=params)

                if response.status_code != 200:
                    print(f"  ⚠️ API 오류: {response.status_code}")
                    return []

                data = response.json()
                studies = data.get("studies", [])
                return [self._parse_study(s) for s in studies]

        except httpx.TimeoutException:
            print("  ⚠️ API 타임아웃")
            return []
        except Exception as e:
            print(f"  ⚠️ 오류: {e}")
            return []

    def _parse_study(self, study: Dict) -> ClinicalTrial:
        """Parse a study from API response."""
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        desc_module = protocol.get("descriptionModule", {})
        design_module = protocol.get("designModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        outcomes_module = protocol.get("outcomesModule", {})

        # Extract interventions
        interventions = []
        for arm in arms_module.get("interventions", []):
            name = arm.get("name", "")
            if name:
                interventions.append(name)

        # Extract primary outcomes
        primary_outcomes = []
        for outcome in outcomes_module.get("primaryOutcomes", []):
            measure = outcome.get("measure", "")
            if measure:
                primary_outcomes.append(measure)

        trial = ClinicalTrial(
            nct_id=id_module.get("nctId", ""),
            title=id_module.get("briefTitle", ""),
            phase=self._extract_phase(design_module),
            status=status_module.get("overallStatus", ""),
            conditions=conditions_module.get("conditions", []),
            interventions=interventions[:5],  # Limit
            sponsor=sponsor_module.get("leadSponsor", {}).get("name", ""),
            start_date=self._extract_date(status_module.get("startDateStruct")),
            completion_date=self._extract_date(status_module.get("completionDateStruct")),
            results_date=self._extract_date(study.get("resultsSection", {}).get("resultsFirstSubmitDateStruct")),
            enrollment=design_module.get("enrollmentInfo", {}).get("count", 0),
            brief_summary=desc_module.get("briefSummary", "")[:500],
            primary_outcome="; ".join(primary_outcomes[:2]),
            has_results=bool(study.get("resultsSection")),
        )

        # Calculate priority
        trial.priority = self._calculate_priority(trial)

        return trial

    def _extract_phase(self, design_module: Dict) -> str:
        """Extract phase from design module."""
        phases = design_module.get("phases", [])
        if phases:
            return phases[0].replace("PHASE", "Phase ")
        return "Unknown"

    def _extract_date(self, date_struct: Optional[Dict]) -> Optional[str]:
        """Extract date from date structure."""
        if not date_struct:
            return None
        return date_struct.get("date")

    def _calculate_priority(self, trial: ClinicalTrial) -> int:
        """Calculate trial priority score."""
        score = 0
        text = f"{trial.title} {' '.join(trial.conditions)} {' '.join(trial.interventions)}".lower()

        # Phase 3 bonus
        if "phase 3" in trial.phase.lower():
            score += 50

        # Has results bonus
        if trial.has_results:
            score += 40

        # Big pharma sponsor
        sponsor_lower = trial.sponsor.lower()
        for company in self.BIG_PHARMA:
            if company.lower() in sponsor_lower:
                score += 30
                break

        # Hot conditions
        for condition in self.HOT_CONDITIONS:
            if condition in text:
                score += 20
                break

        # Hot drugs
        for drug in self.HOT_DRUGS:
            if drug in text:
                score += 25
                break

        # Large enrollment bonus
        if trial.enrollment >= 1000:
            score += 20
        elif trial.enrollment >= 500:
            score += 10

        return score

    async def fetch_all(self, results_days: int = 30, new_trials_days: int = 14) -> Dict[str, List[ClinicalTrial]]:
        """
        Fetch all types of clinical trial updates.

        Returns:
            Dict with keys: phase3_results, new_trials, terminated
        """
        results = await self.fetch_phase3_with_results(days=results_days)
        new_trials = await self.fetch_new_phase3_trials(days=new_trials_days)
        terminated = await self.fetch_terminated_trials(days=results_days)

        return {
            "phase3_results": results,
            "new_trials": new_trials,
            "terminated": terminated,
        }


async def main():
    """Test ClinicalTrials fetcher."""
    print("=" * 60)
    print("ClinicalTrials.gov Fetcher Test")
    print("=" * 60)

    fetcher = ClinicalTrialsFetcher()

    # Fetch all
    data = await fetcher.fetch_all(results_days=60, new_trials_days=30)

    # Show Phase 3 results
    results = data["phase3_results"]
    if results:
        print(f"\n📊 Phase 3 결과 발표 ({len(results)}건):")
        for trial in sorted(results, key=lambda x: x.priority, reverse=True)[:5]:
            print(f"\n  [{trial.nct_id}] {trial.title[:60]}...")
            print(f"    Sponsor: {trial.sponsor}")
            print(f"    Conditions: {', '.join(trial.conditions[:2])}")
            print(f"    Interventions: {', '.join(trial.interventions[:2])}")
            print(f"    Priority: {trial.priority}")

    # Show new trials
    new_trials = data["new_trials"]
    if new_trials:
        print(f"\n🆕 신규 Phase 3 임상 ({len(new_trials)}건):")
        for trial in sorted(new_trials, key=lambda x: x.priority, reverse=True)[:3]:
            print(f"\n  [{trial.nct_id}] {trial.title[:60]}...")
            print(f"    Sponsor: {trial.sponsor}")

    # Show terminated
    terminated = data["terminated"]
    if terminated:
        print(f"\n❌ 중단/실패 임상 ({len(terminated)}건):")
        for trial in terminated[:3]:
            print(f"  • [{trial.nct_id}] {trial.title[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
