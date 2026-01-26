"""
Report Section Components

Modular section generators for RNA-seq reports.
"""

from .common import (
    CoverSection,
    SummarySection,
    AbstractSection,
    QCSection,
    DriverSection,
    ClinicalSection,
    FollowUpSection,
    MethodsSection,
    ResearchSection,
    ReferencesSection,
    AppendixSection,
)

from .bulk import (
    DEGSection,
    PathwaySection,
    NetworkSection,
    VolcanoSection,
    HeatmapSection,
)

from .singlecell import (
    CellTypeSection,
    MarkerSection,
    TrajectorySection,
    TMESection,
    GRNSection,
    PloidySection,
    InteractionSection,
)

__all__ = [
    # Common
    "CoverSection",
    "SummarySection",
    "AbstractSection",
    "QCSection",
    "DriverSection",
    "ClinicalSection",
    "FollowUpSection",
    "MethodsSection",
    "ResearchSection",
    "ReferencesSection",
    "AppendixSection",
    # Bulk
    "DEGSection",
    "PathwaySection",
    "NetworkSection",
    "VolcanoSection",
    "HeatmapSection",
    # Single-cell
    "CellTypeSection",
    "MarkerSection",
    "TrajectorySection",
    "TMESection",
    "GRNSection",
    "PloidySection",
    "InteractionSection",
]
