#!/usr/bin/env python3
"""
Single-Cell 6-Agent Pipeline 평가 실행 스크립트

Usage:
    python run_sc_evaluation.py --agents-dir /path/to/agents --output-dir ./results
"""

import argparse
import sys
import importlib.util
from pathlib import Path


def load_agent_module(module_path: Path):
    """에이전트 모듈 동적 로드"""
    spec = importlib.util.spec_from_file_location("agent_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_agent_classes(agents_dir: Path) -> dict:
    """에이전트 디렉토리에서 에이전트 클래스 찾기"""
    agent_classes = {}
    
    # Expected file patterns
    agent_patterns = {
        1: ["agent1", "qc", "preprocessing"],
        2: ["agent2", "cluster", "celltype"],
        3: ["agent3", "pathway", "database"],
        4: ["agent4", "trajectory", "dynamics"],
        5: ["agent5", "cnv", "ml_prediction"],
        6: ["agent6", "report", "visualization"],
    }
    
    for agent_num, patterns in agent_patterns.items():
        for py_file in agents_dir.glob("**/*.py"):
            file_lower = py_file.stem.lower()
            if any(p in file_lower for p in patterns):
                try:
                    module = load_agent_module(py_file)
                    
                    # Find agent class
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (isinstance(obj, type) and 
                            hasattr(obj, 'run') and 
                            hasattr(obj, 'validate_inputs') and
                            'Agent' in name):
                            agent_classes[agent_num] = obj
                            print(f"  Found Agent {agent_num}: {name} in {py_file.name}")
                            break
                except Exception as e:
                    print(f"  Warning: Could not load {py_file}: {e}")
                
                if agent_num in agent_classes:
                    break
    
    return agent_classes


def main():
    parser = argparse.ArgumentParser(
        description="Single-Cell 6-Agent Pipeline Evaluation"
    )
    parser.add_argument(
        "--agents-dir",
        type=str,
        required=True,
        help="Directory containing agent Python files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sc_evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated list of agent numbers to evaluate (default: all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file for agents (optional)"
    )
    
    args = parser.parse_args()
    
    agents_dir = Path(args.agents_dir)
    output_dir = Path(args.output_dir)
    
    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}")
        return 1
    
    print(f"\n🔍 Searching for agents in: {agents_dir}")
    agent_classes = find_agent_classes(agents_dir)
    
    if not agent_classes:
        print("Error: No agent classes found!")
        return 1
    
    # Filter agents if specified
    selected_agents = [int(x) for x in args.agents.split(",")]
    agent_classes = {k: v for k, v in agent_classes.items() if k in selected_agents}
    
    print(f"\n📋 Found {len(agent_classes)} agents to evaluate")
    
    # Load config if provided
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
    
    # Import evaluator
    from sc_pipeline_evaluator import SingleCellPipelineEvaluator
    
    # Run evaluation
    evaluator = SingleCellPipelineEvaluator(
        agent_classes=agent_classes,
        output_dir=output_dir
    )
    
    report = evaluator.run_full_evaluation(config=config)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Overall Score: {report.overall_score:.1f}/100")
    print(f"   Grade: {report.grade}")
    
    return 0 if report.overall_score >= 70 else 1


if __name__ == "__main__":
    sys.exit(main())
