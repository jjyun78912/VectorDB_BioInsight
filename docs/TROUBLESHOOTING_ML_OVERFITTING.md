# ML Model Overfitting Troubleshooting

## Overview

이 문서는 BioInsight AI의 ML 모델(CatBoost Pan-Cancer Classifier)에서 발견된 과적합(overfitting) 가능성과 그 원인, 해결 방안을 다룹니다.

---

## 문제 상황

### 의심스러운 높은 성능 지표

| Metric | 값 | 일반적 기대치 |
|--------|-----|---------------|
| **AUC (Binary)** | 0.998 | 0.85-0.92 |
| **Accuracy (17-class)** | 97.2% | 80-90% |
| **F1 Macro** | 97.15% | 75-88% |
| **Synthetic Validation** | 100% | N/A (비현실적) |

이러한 높은 성능은 **Data Leakage** 또는 **과적합**을 강하게 시사합니다.

---

## 원인 분석

### 1. Data Leakage (v1.0 Preprocessor)

**문제**: v1.0 전처리기에서 Train/Test Split **이전에** Feature Selection(HVG 선택)을 수행

```python
# v1.0 (문제 있는 코드)
class PreprocessorV1:
    def preprocess(self, data):
        # 1. HVG Selection (전체 데이터로!)
        hvg = self._select_hvg(data)  # ❌ WRONG: Test set 정보가 누출됨

        # 2. Split (이미 누출된 후)
        train, test = train_test_split(data[hvg])
```

**결과**: Test set의 분포가 Feature Selection에 영향을 미침 → 모델이 Test set을 "미리 본" 효과

### 2. v2.0 Preprocessor (수정됨)

```python
# v2.0 (수정된 코드)
class PreprocessorV2:
    def preprocess(self, data):
        # 1. Split FIRST
        train, test = train_test_split(data)

        # 2. HVG Selection on TRAIN only
        hvg = self._select_hvg(train)  # ✅ CORRECT

        # 3. Apply to both
        train_hvg = train[hvg]
        test_hvg = test[hvg]  # Test는 fitting에 사용 안 됨
```

**그러나**: 현재 저장된 모델(`models/rnaseq/pancancer/`)은 v1.0으로 학습된 것으로 추정됨

### 3. Synthetic Validation의 문제

`robust_validation/validation_report.json`:
```json
{
  "accuracy": 1.0,
  "samples_tested": 85,
  "note": "Synthetic perturbation test"
}
```

**문제점**:
- Synthetic data가 원본과 너무 유사하게 생성됨
- 실제 외부 데이터(다른 연구실, 다른 플랫폼)와는 분포가 다름
- 100% 정확도는 명백한 과적합 신호

### 4. External Validation 부재

| 검증 유형 | 수행 여부 | 상태 |
|-----------|----------|------|
| Internal CV (5-fold) | ✅ | 완료 |
| Hold-out Test | ✅ | 완료 (누출 의심) |
| Synthetic Perturbation | ✅ | 비현실적 |
| **External Dataset (GEO, ArrayExpress)** | ❌ | **미수행** |
| **Independent Cohort** | ❌ | **미수행** |

---

## 현실적 성능 기대치

### 암종 분류 관련 문헌 참조

| 연구 | Dataset | Classes | Best AUC/Accuracy | 방법 |
|------|---------|---------|-------------------|------|
| Cancer Genome Atlas (2018) | TCGA Pan-Cancer | 33종 | ~85% | Random Forest |
| Hoadley et al. (2014) | TCGA | 12종 | ~90% | Cluster-based |
| Chen et al. (2020) | GEO+TCGA | 10종 | ~88% | Deep Learning |

**결론**: 17종 분류에서 **85-90% 정확도**가 현실적

### 혼동 가능 암종 쌍

생물학적으로 유사한 암종은 RNA-seq만으로 구분이 어려움:

| 쌍 | 공통점 | 예상 혼동률 |
|----|--------|-------------|
| HNSC ↔ LUSC | 편평상피세포암 (SCC) | 5-15% |
| LUAD ↔ PAAD | 선암 (Adenocarcinoma) | 3-8% |
| COAD ↔ STAD | 소화기 기원 | 5-10% |
| SKCM ↔ HNSC | 점막 흑색종 | 2-5% |

---

## 해결 방안

### 즉시 조치 (Priority: High)

1. **v2.0 Preprocessor로 재학습**
   ```bash
   python rnaseq_pipeline/ml/trainer.py \
     --preprocessor v2 \
     --output models/rnaseq/pancancer_v2/
   ```

2. **External Validation 수행**
   - GEO에서 동일 암종 데이터 다운로드
   - ArrayExpress에서 독립 코호트 확보
   - 학습에 사용되지 않은 데이터로만 검증

3. **성능 지표 재평가**
   - Bootstrap Confidence Interval 계산
   - Per-cancer-type 성능 분석
   - Confusion matrix 상세 분석

### 중기 조치 (Priority: Medium)

1. **Cross-platform Validation**
   - RNA-seq (Illumina) → Microarray 데이터로 검증
   - 다른 시퀀싱 플랫폼 (NovaSeq vs HiSeq)

2. **Prospective Validation**
   - 새로 생성되는 데이터로 지속적 검증
   - Time-based split (학습: ~2022, 검증: 2023+)

3. **Uncertainty Quantification**
   - Prediction confidence 출력
   - "확실하지 않음" 결과 허용
   - Confusable pair 경고 시스템

### 장기 조치 (Priority: Low)

1. **Multi-modal Integration**
   - RNA-seq + Methylation
   - RNA-seq + Copy Number
   - 더 robust한 분류 가능

2. **Ensemble with External Models**
   - 공개된 검증된 모델과 앙상블
   - Consensus prediction

---

## 사용자 주의사항

### 리포트에 표시되는 경고문

```
⚠️ ML 예측 결과 주의사항:
- 이 예측은 연구 참고용이며, 임상 진단이 아닙니다
- 모델 성능은 내부 검증 기준이며, 외부 데이터에서 다를 수 있습니다
- 최종 판단은 반드시 전문 의료진과 함께 해주세요
```

### Confidence Gap 해석

```python
# predictor.py에서의 불확실성 처리
if confidence_gap < 0.15:  # Top1 - Top2 확률 차이
    warning = "⚠️ 예측 불확실: 2개 이상의 암종이 유사한 확률을 보입니다"
```

### 혼동 가능 쌍 경고

```python
# 생물학적으로 혼동 가능한 경우 추가 경고
if is_confusable_pair(predicted, second_best):
    warning += f"\n⚠️ {predicted}와 {second_best}는 조직학적으로 유사합니다"
```

---

## 결론

현재 ML 모델의 높은 성능 지표(AUC 0.998, Accuracy 97.2%)는 **Data Leakage로 인한 과적합 가능성이 높습니다**.

### 신뢰할 수 있는 사용 조건:
1. 예측 결과를 "참고용"으로만 활용
2. Confidence gap과 혼동 가능 쌍 경고 확인
3. 반드시 실험적 검증 권장 (qPCR, IHC 등)

### 향후 개선 필요:
1. v2.0 preprocessor로 재학습
2. External validation (GEO/ArrayExpress)
3. 현실적 성능 기대치(85-90%) 반영

---

*Last Updated: 2026-01-27*
