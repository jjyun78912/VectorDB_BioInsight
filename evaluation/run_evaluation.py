#!/usr/bin/env python3
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sc_pipeline_evaluator import SingleCellPipelineEvaluator

# ★ 본인 에이전트 경로에 맞게 수정
from rnaseq_pipeline.agents.singlecell.agent1_qc import SingleCellQCAgent
from rnaseq_pipeline.agents.singlecell.agent2_cluster import SingleCellClusterAgent
from rnaseq_pipeline.agents.singlecell.agent3_pathway import SingleCellPathwayAgent
from rnaseq_pipeline.agents.singlecell.agent4_trajectory import SingleCellTrajectoryAgent
from rnaseq_pipeline.agents.singlecell.agent5_cnv_ml import SingleCellCNVMLAgent
from rnaseq_pipeline.agents.singlecell.agent6_report import SingleCellReportAgent

evaluator = SingleCellPipelineEvaluator(
    agent_classes={
        1: SingleCellQCAgent,
        2: SingleCellClusterAgent,
        3: SingleCellPathwayAgent,
        4: SingleCellTrajectoryAgent,
        5: SingleCellCNVMLAgent,
        6: SingleCellReportAgent
    },
    output_dir=Path("./evaluation_results")
)

report = evaluator.run_full_evaluation()
print(f"\n결과: {report.overall_score:.1f}/100 (등급: {report.grade})")
