"""
Unified Report Generation System for RNA-seq Pipeline

Supports both Bulk RNA-seq and Single-cell RNA-seq with:
- Common base template and styling
- Modular section components
- Data-type specific sections
"""

from .base_report import BaseReportGenerator, ReportConfig, ReportData
from .unified_report import UnifiedReportGenerator

__all__ = [
    "BaseReportGenerator",
    "ReportConfig",
    "ReportData",
    "UnifiedReportGenerator",
]
