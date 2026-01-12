#!/usr/bin/env python3
"""
Pan-Cancer 모델 검증 스크립트
============================

1. 모델 로드 테스트
2. 실제 TCGA 데이터로 예측
3. Unknown/OOD 탐지 검증
4. UnifiedPredictor 통합 테스트
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from rnaseq_pipeline.ml import (
    PanCancerClassifier,
    UnifiedPredictor,
)

def test_model_load():
    """1. 모델 로드 테스트"""
    print("\n" + "="*60)
    print("  [1] 모델 로드 테스트")
    print("="*60)
    
    model_dir = "models/rnaseq/pancancer"
    
    try:
        classifier = PanCancerClassifier(model_dir)
        classifier.load()
        
        info = classifier.get_model_info()
        print(f"  ✅ 모델 로드 성공")
        print(f"     - 클래스 수: {info['n_classes']}")
        print(f"     - 암종: {', '.join(info['cancer_types'][:5])}...")
        print(f"     - 앙상블 모델: {info['ensemble_models']}")
        print(f"     - 사용 유전자 수: {info['n_genes']}")
        return classifier
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
        return None


def test_prediction_with_tcga(classifier):
    """2. 실제 TCGA 데이터로 예측 테스트"""
    print("\n" + "="*60)
    print("  [2] TCGA 데이터 예측 테스트")
    print("="*60)
    
    # 테스트용 데이터 로드 (일부 샘플만)
    data_path = Path("data/tcga/pancancer/pancancer_counts.csv")
    meta_path = Path("data/tcga/pancancer/pancancer_metadata.csv")
    
    if not data_path.exists():
        print("  ⚠️ 테스트 데이터 없음, 스킵")
        return
    
    print("  데이터 로드 중...")
    counts = pd.read_csv(data_path, index_col=0)
    metadata = pd.read_csv(meta_path)
    
    # 암종별 2개씩 샘플 선택
    test_samples = []
    for cancer in metadata['cancer_type'].unique()[:5]:  # 5개 암종만
        samples = metadata[metadata['cancer_type'] == cancer]['barcode'].head(2).tolist()
        test_samples.extend(samples)
    
    test_counts = counts[test_samples]
    print(f"  테스트 샘플: {len(test_samples)}개")
    
    # 예측
    print("  예측 수행 중...")
    results = classifier.predict(test_counts, test_samples)
    
    # 결과 출력
    print(f"\n  {'샘플 ID':<20} {'예측 암종':<10} {'신뢰도':>8} {'레벨':<10}")
    print("  " + "-"*55)
    
    correct = 0
    for r in results:
        actual = metadata[metadata['barcode'] == r.sample_id]['cancer_type'].values[0]
        match = "✅" if r.predicted_cancer == actual else "❌"
        if r.predicted_cancer == actual:
            correct += 1
        print(f"  {r.sample_id[:18]:<20} {r.predicted_cancer:<10} {r.confidence:>7.1%} {r.confidence_level:<10} {match}")
    
    print(f"\n  정확도: {correct}/{len(results)} ({correct/len(results):.1%})")
    return results


def test_unknown_detection(classifier):
    """3. Unknown/OOD 탐지 테스트"""
    print("\n" + "="*60)
    print("  [3] Unknown/OOD 탐지 테스트")
    print("="*60)
    
    # 랜덤 노이즈 데이터 생성 (OOD 시뮬레이션)
    np.random.seed(42)
    n_genes = len(classifier.preprocessor.selected_genes)
    
    # 케이스 1: 완전 랜덤 노이즈
    random_data = pd.DataFrame(
        np.random.poisson(10, (n_genes, 2)),
        index=classifier.preprocessor.selected_genes,
        columns=['random_1', 'random_2']
    )
    
    # 케이스 2: 0에 가까운 발현 (빈 샘플)
    zero_data = pd.DataFrame(
        np.random.poisson(0.1, (n_genes, 1)),
        index=classifier.preprocessor.selected_genes,
        columns=['near_zero']
    )
    
    # 케이스 3: 극단적 발현
    extreme_data = pd.DataFrame(
        np.random.poisson(10000, (n_genes, 1)),
        index=classifier.preprocessor.selected_genes,
        columns=['extreme']
    )
    
    test_data = pd.concat([random_data, zero_data, extreme_data], axis=1)
    
    print("  OOD 테스트 데이터:")
    print("    - random_1, random_2: 랜덤 노이즈")
    print("    - near_zero: 거의 0 발현")
    print("    - extreme: 극단적 고발현")
    
    results = classifier.predict(test_data)
    
    print(f"\n  {'샘플':<15} {'예측':<10} {'신뢰도':>8} {'Unknown?':<10}")
    print("  " + "-"*50)
    
    for r in results:
        unknown_str = "⚠️ YES" if r.is_unknown else "NO"
        print(f"  {r.sample_id:<15} {r.predicted_cancer:<10} {r.confidence:>7.1%} {unknown_str:<10}")
        
    # Unknown 탐지 성공 여부
    unknown_count = sum(1 for r in results if r.is_unknown or r.confidence < 0.5)
    print(f"\n  낮은 신뢰도/Unknown 탐지: {unknown_count}/{len(results)}")
    
    if unknown_count >= 2:
        print("  ✅ OOD 탐지 기능 정상 작동")
    else:
        print("  ⚠️ OOD 탐지 임계값 조정 필요할 수 있음")


def test_unified_predictor():
    """4. UnifiedPredictor 통합 테스트"""
    print("\n" + "="*60)
    print("  [4] UnifiedPredictor 통합 테스트")
    print("="*60)
    
    try:
        predictor = UnifiedPredictor()
        predictor.load(load_binary=True)
        
        models = predictor.get_available_models()
        print("  사용 가능한 모델:")
        print(f"    - Pan-Cancer: {'✅' if models['pancancer']['available'] else '❌'}")
        for cancer, info in models['binary_models'].items():
            status = '✅' if info['available'] else '❌'
            print(f"    - {cancer}: {status}")
        
        # 테스트 데이터
        data_path = Path("data/tcga/pancancer/pancancer_counts.csv")
        if data_path.exists():
            counts = pd.read_csv(data_path, index_col=0)
            
            # BRCA 샘플 1개 테스트 (binary model 있음)
            meta = pd.read_csv("data/tcga/pancancer/pancancer_metadata.csv")
            brca_sample = meta[meta['cancer_type'] == 'BRCA']['barcode'].iloc[0]
            
            test_counts = counts[[brca_sample]]
            
            print(f"\n  통합 예측 테스트 (샘플: {brca_sample[:20]}...)")
            results = predictor.predict(test_counts, include_binary_analysis=True)
            
            r = results[0]
            print(f"\n  예측 결과:")
            print(f"    - 암종: {r.predicted_cancer} ({r.predicted_cancer_korean})")
            print(f"    - 신뢰도: {r.confidence:.1%} [{r.confidence_level}]")
            print(f"    - 앙상블 일치도: {r.ensemble_agreement:.1%}")
            
            if r.detailed_analysis:
                print(f"    - Binary 분석: {r.detailed_analysis}")
            
            if r.top_genes:
                print(f"    - Top 유전자: {[g['gene'] for g in r.top_genes[:3]]}")
                
            print("\n  ✅ UnifiedPredictor 정상 작동")
        else:
            print("  ⚠️ 테스트 데이터 없음")
            
    except Exception as e:
        print(f"  ❌ UnifiedPredictor 오류: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*60)
    print("  Pan-Cancer 모델 검증")
    print("="*60)
    
    # 1. 모델 로드
    classifier = test_model_load()
    if not classifier:
        return
    
    # 2. TCGA 데이터 예측
    test_prediction_with_tcga(classifier)
    
    # 3. Unknown 탐지
    test_unknown_detection(classifier)
    
    # 4. 통합 테스트
    test_unified_predictor()
    
    print("\n" + "="*60)
    print("  검증 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
