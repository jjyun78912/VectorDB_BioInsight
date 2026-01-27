"""
Unified Report Generator for RNA-seq Pipeline

Generates HTML reports for both Bulk and Single-cell RNA-seq analysis
using a common template and modular section components.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base_report import BaseReportGenerator, ReportConfig, ReportData
from .sections.common import (
    BaseSection,
    CoverSection,
    SummarySection,
    AbstractSection,
    QCSection,
    DriverSection,
    MLPredictionSection,
    ClinicalSection,
    FollowUpSection,
    MethodsSection,
    ReferencesSection,
    AppendixSection,
)
from .sections.bulk import (
    DEGSection,
    PathwaySection,
    NetworkSection,
)
from .sections.singlecell import (
    CellTypeSection,
    MarkerSection,
    TrajectorySection,
    TMESection,
    GRNSection,
    PloidySection,
    InteractionSection,
)


class UnifiedReportGenerator(BaseReportGenerator):
    """Unified report generator for both Bulk and Single-cell RNA-seq."""

    # Section order for Bulk RNA-seq
    # Matches original agent6_report.py structure:
    # 1. 연구 개요, (Abstract), 2. QC, 3. DEG, 4. Pathway, 5. Driver, 6. Network,
    # 7. Clinical, 8. 후속 연구 제안, 9. Methods, 10. 문헌 기반 해석, 11. 부록
    BULK_SECTIONS: List[Type[BaseSection]] = [
        CoverSection,
        SummarySection,       # 1. 연구 개요
        AbstractSection,      # 연구 요약 (Extended Abstract) - LLM 기반
        QCSection,            # 2. 데이터 품질 관리
        DEGSection,           # 3. 차등발현 분석
        PathwaySection,       # 4. 경로 및 기능 분석
        DriverSection,        # 5. Driver 유전자 분석
        NetworkSection,       # 6. 네트워크 분석
        MLPredictionSection,  # (선택) ML 암종 예측
        ClinicalSection,      # 7. 임상적 시사점
        FollowUpSection,      # 8. 후속 연구 제안 (치료 타겟, 약물 재목적화, 실험 검증, 추천 논문)
        MethodsSection,       # 9. 분석 방법
        ReferencesSection,    # 10. 문헌 기반 해석
        AppendixSection,      # 11. 부록
    ]

    # Section order for Single-cell RNA-seq
    SINGLECELL_SECTIONS: List[Type[BaseSection]] = [
        CoverSection,
        SummarySection,       # 1. 연구 개요
        AbstractSection,      # 연구 요약 (Extended Abstract)
        QCSection,            # 2. 데이터 품질 관리
        CellTypeSection,      # 3. 세포 유형 분석
        MarkerSection,        # 4. 마커 유전자
        DriverSection,        # 5. Driver 유전자 분석
        TrajectorySection,    # (선택) Trajectory
        TMESection,           # (선택) TME
        GRNSection,           # (선택) GRN
        PloidySection,        # (선택) Ploidy
        InteractionSection,   # (선택) Cell-cell interaction
        MLPredictionSection,  # (선택) Pseudo-bulk ML 예측
        ClinicalSection,      # 7. 임상적 시사점
        FollowUpSection,      # 8. 후속 연구 제안
        MethodsSection,       # 9. 분석 방법
        ReferencesSection,    # 10. 문헌 기반 해석
        AppendixSection,      # 11. 부록
    ]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[ReportConfig] = None
    ):
        super().__init__(input_dir, output_dir, config)

    def get_sections(self) -> List[Type[BaseSection]]:
        """Get section list based on data type."""
        if self.config.data_type == "singlecell":
            return self.SINGLECELL_SECTIONS
        else:
            return self.BULK_SECTIONS

    def generate_navigation(self, sections: List[BaseSection]) -> str:
        """Generate navigation bar HTML."""
        links = []
        for section in sections:
            if section.is_available() and section.section_id:
                title = section.get_title()
                # Shorten title for nav
                if len(title) > 15:
                    title = title[:12] + "..."
                links.append(f'<a href="#{section.section_id}">{title}</a>')

        return f'''
        <nav class="nav-bar">
            <div class="nav-container">
                <span class="nav-brand">BioInsight Report</span>
                <div class="nav-links">
                    {''.join(links)}
                </div>
            </div>
        </nav>
        '''

    def generate_footer(self) -> str:
        """Generate footer HTML."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f'''
        <footer class="report-footer">
            <div class="footer-disclaimer">
                <strong>면책조항:</strong> 본 보고서는 AI 지원 분석 파이프라인에 의해 생성되었습니다.
                모든 발견은 예비적이며, 임상 적용 전 실험적 검증이 필요합니다.
                예측 결과는 진단 목적으로 사용할 수 없습니다.
            </div>
            <p class="footer-credit">
                BioInsight AI RNA-seq Pipeline v{self.VERSION} | Generated: {timestamp}
            </p>
        </footer>
        '''

    def generate_html(self, data: ReportData) -> str:
        """Generate complete HTML report."""
        # Get section classes for this data type
        section_classes = self.get_sections()

        # Instantiate sections
        sections = [cls(self.config, data) for cls in section_classes]

        # Filter to available sections
        available_sections = [s for s in sections if s.is_available()]

        # Generate section HTML
        sections_html = []
        for section in available_sections:
            try:
                html = section.render()
                if html:
                    sections_html.append(html)
            except Exception as e:
                self.logger.warning(f"Error rendering section {section.section_id}: {e}")

        # Theme class
        theme_class = f"theme-{self.config.theme}" if self.config.theme != "light" else ""

        return f'''
<!DOCTYPE html>
<html lang="{self.config.language}" class="{theme_class}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.report_title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
{self.generate_css()}
    </style>
</head>
<body>
    {''.join(sections_html[:1])}

    {self.generate_navigation(available_sections)}

    <main class="main-content">
        {''.join(sections_html[1:])}
    </main>

    {self.generate_footer()}

    {self.generate_javascript(data)}
</body>
</html>
'''

    def generate(self) -> Path:
        """Generate the report and save to file."""
        self.logger.info(f"Generating {self.config.data_type} RNA-seq report...")

        # Load data
        data = self.load_data()
        self.logger.info(f"Loaded data: figures={len(data.figures)}, interactive={len(data.interactive_figures)}")

        # Generate HTML
        html_content = self.generate_html(data)

        # Save report
        report_path = self.output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Report saved: {report_path}")

        # Save report data as JSON for API consumption
        report_data = {
            "config": {
                "data_type": self.config.data_type,
                "cancer_type": self.config.cancer_type,
                "language": self.config.language,
            },
            "generated_at": datetime.now().isoformat(),
            "version": self.VERSION,
        }
        data_path = self.output_dir / "report_data.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        return report_path


def generate_report(
    input_dir: Path,
    output_dir: Path,
    data_type: str = "bulk",
    cancer_type: str = "unknown",
    report_title: Optional[str] = None,
    **kwargs
) -> Path:
    """
    Convenience function to generate a report.

    Args:
        input_dir: Directory containing analysis results
        output_dir: Directory to save the report
        data_type: "bulk" or "singlecell"
        cancer_type: Cancer type code (e.g., "BRCA", "LUAD")
        report_title: Custom report title
        **kwargs: Additional config options

    Returns:
        Path to generated report HTML
    """
    if report_title is None:
        data_type_label = "Single-cell" if data_type == "singlecell" else "Bulk"
        report_title = f"{data_type_label} RNA-seq Analysis Report"
        if cancer_type != "unknown":
            report_title = f"{cancer_type.upper()} {report_title}"

    config = ReportConfig(
        report_title=report_title,
        data_type=data_type,
        cancer_type=cancer_type,
        **kwargs
    )

    generator = UnifiedReportGenerator(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        config=config
    )

    return generator.generate()
