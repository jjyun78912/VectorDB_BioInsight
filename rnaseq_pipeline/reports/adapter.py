"""
Report Adapter

Provides compatibility layer between existing agent classes and the new
unified report system. Allows gradual migration from old to new system.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base_report import ReportConfig, ReportData
from .unified_report import UnifiedReportGenerator


class ReportAdapter:
    """
    Adapter to use unified report system from existing agents.

    Usage in agent6_report.py:
        from rnaseq_pipeline.reports.adapter import ReportAdapter

        class ReportAgent(BaseAgent):
            def run(self) -> Dict[str, Any]:
                # Option 1: Use unified system
                if self.config.get("use_unified_report", False):
                    adapter = ReportAdapter(self.input_dir, self.output_dir)
                    return adapter.generate_bulk_report(
                        cancer_type=self.config.get("cancer_type", "unknown")
                    )

                # Option 2: Use existing system (default)
                ...existing code...

    Usage in agent_singlecell_report.py:
        from rnaseq_pipeline.reports.adapter import ReportAdapter

        class SingleCellReportAgent(BaseAgent):
            def run(self) -> Dict[str, Any]:
                if self.config.get("use_unified_report", False):
                    adapter = ReportAdapter(self.input_dir, self.output_dir)
                    return adapter.generate_singlecell_report(
                        cancer_type=self.config.get("cancer_type", "unknown")
                    )
                ...
    """

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def generate_bulk_report(
        self,
        cancer_type: str = "unknown",
        report_title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate bulk RNA-seq report using unified system."""
        if report_title is None:
            report_title = f"Bulk RNA-seq Analysis Report"
            if cancer_type != "unknown":
                report_title = f"{cancer_type.upper()} {report_title}"

        config = ReportConfig(
            report_title=report_title,
            data_type="bulk",
            cancer_type=cancer_type,
            **kwargs
        )

        generator = UnifiedReportGenerator(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            config=config
        )

        report_path = generator.generate()

        return {
            "report_path": str(report_path),
            "system": "unified_v3",
            "data_type": "bulk",
        }

    def generate_singlecell_report(
        self,
        cancer_type: str = "unknown",
        report_title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate single-cell RNA-seq report using unified system."""
        if report_title is None:
            report_title = f"Single-cell RNA-seq Analysis Report"
            if cancer_type != "unknown":
                report_title = f"{cancer_type.upper()} {report_title}"

        config = ReportConfig(
            report_title=report_title,
            data_type="singlecell",
            cancer_type=cancer_type,
            **kwargs
        )

        generator = UnifiedReportGenerator(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            config=config
        )

        report_path = generator.generate()

        return {
            "report_path": str(report_path),
            "system": "unified_v3",
            "data_type": "singlecell",
        }

    def generate_multiomic_report(
        self,
        cancer_type: str = "unknown",
        report_title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate multi-omic report using unified system."""
        if report_title is None:
            report_title = f"Multi-omic Analysis Report"
            if cancer_type != "unknown":
                report_title = f"{cancer_type.upper()} {report_title}"

        config = ReportConfig(
            report_title=report_title,
            data_type="multiomic",
            cancer_type=cancer_type,
            **kwargs
        )

        generator = UnifiedReportGenerator(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            config=config
        )

        report_path = generator.generate()

        return {
            "report_path": str(report_path),
            "system": "unified_v3",
            "data_type": "multiomic",
        }


def migrate_to_unified(
    input_dir: Path,
    output_dir: Path,
    data_type: str = "bulk",
    cancer_type: str = "unknown",
) -> Path:
    """
    Convenience function for one-off migration to unified report.

    Args:
        input_dir: Directory with analysis outputs
        output_dir: Directory to save report
        data_type: "bulk" or "singlecell"
        cancer_type: Cancer type code

    Returns:
        Path to generated report
    """
    adapter = ReportAdapter(input_dir, output_dir)

    if data_type == "singlecell":
        result = adapter.generate_singlecell_report(cancer_type=cancer_type)
    elif data_type == "multiomic":
        result = adapter.generate_multiomic_report(cancer_type=cancer_type)
    else:
        result = adapter.generate_bulk_report(cancer_type=cancer_type)

    return Path(result["report_path"])
