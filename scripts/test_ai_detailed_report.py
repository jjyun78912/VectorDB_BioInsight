#!/usr/bin/env python3
"""Test AI detailed analysis report generation."""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.agents.agent6_report import ReportAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Use existing accumulated data
    accumulated_dir = Path("/Users/admin/VectorDB_BioInsight/rnaseq_test_results/stad_full_report/run_20260124_205314/accumulated")
    output_dir = Path("/Users/admin/VectorDB_BioInsight/rnaseq_test_results/ai_detailed_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load report data
    report_data_path = accumulated_dir / "report_data.json"
    if not report_data_path.exists():
        logger.error(f"report_data.json not found at {report_data_path}")
        return

    with open(report_data_path, 'r') as f:
        report_data = json.load(f)

    logger.info(f"Loaded report data with {len(report_data)} keys")

    # Configure for STAD cancer type
    config = {
        'cancer_type': 'STAD',
        'output_language': 'ko',
        'report_title': 'AI 상세 분석 테스트 리포트'
    }

    # Initialize agent with input_dir, output_dir, config
    agent = ReportAgent(
        input_dir=accumulated_dir,
        output_dir=output_dir,
        config=config
    )

    # Run report generation (agent loads data from input_dir)
    logger.info("Starting report generation with detailed AI analysis...")
    result = agent.run()

    if result.get('status') == 'success':
        logger.info(f"✅ Report generated successfully!")
        logger.info(f"   Report path: {result.get('report_path')}")

        # Check if Extended Abstract was generated
        abstract_path = output_dir / "abstract_extended.json"
        if abstract_path.exists():
            with open(abstract_path, 'r') as f:
                abstract = json.load(f)
            abstract_text = abstract.get('abstract_extended', '')
            logger.info(f"   Extended Abstract length: {len(abstract_text)} characters")

        print(f"\n✅ 테스트 완료! 리포트 확인: {result.get('report_path')}")
    else:
        logger.error(f"❌ Report generation failed: {result.get('error')}")

if __name__ == "__main__":
    main()
