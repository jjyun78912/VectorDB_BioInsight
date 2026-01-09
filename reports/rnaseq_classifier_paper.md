# Machine Learning-Based Classification of Breast and Pancreatic Cancer Using TCGA RNA-seq Data

## Abstract

**Background:** Accurate classification of tumor vs. normal tissue samples using transcriptomic data is fundamental for cancer diagnosis and biomarker discovery. We developed CatBoost-based classifiers for breast and pancreatic cancer using The Cancer Genome Atlas (TCGA) RNA-seq data, with SHAP (SHapley Additive exPlanations) for model interpretability.

**Methods:** RNA-seq count data was obtained from TCGA-BRCA (n=299) and TCGA-PAAD (n=183) via the GDC API. After preprocessing (CPM normalization, log2 transformation, ANOVA feature selection to top 3,000 genes), CatBoost classifiers were trained with Optuna hyperparameter optimization (20 trials). Model performance was evaluated using stratified 5-fold cross-validation and held-out test sets.

**Results:** The breast cancer classifier achieved perfect test performance (AUC=1.00, Accuracy=100%) with CV AUC of 0.993. The pancreatic cancer classifier showed high CV AUC (0.964) but limited test AUC (0.528) due to severe class imbalance (only 4 normal samples). SHAP analysis identified key discriminative genes for each cancer type.

**Conclusions:** Adequate representation of normal samples is critical for robust classifier development. The breast cancer model demonstrates clinical utility, while alternative strategies (GTEx integration, subtype comparison) are recommended for cancers with limited normal tissue availability.

---

## 1. Introduction

Cancer classification using gene expression profiles has become a cornerstone of precision oncology. RNA sequencing (RNA-seq) provides comprehensive transcriptomic snapshots that can distinguish malignant from normal tissues with high accuracy. The Cancer Genome Atlas (TCGA) provides the largest publicly available collection of matched tumor and normal RNA-seq data across 33 cancer types.

Gradient boosting methods, particularly CatBoost, have emerged as powerful tools for tabular data classification, offering native handling of high-dimensional features and built-in regularization. Combined with SHAP (SHapley Additive exPlanations), these models provide both accurate predictions and biological interpretability.

In this study, we developed and validated CatBoost classifiers for:
1. **Breast Invasive Carcinoma (BRCA)** - the most common cancer in women worldwide
2. **Pancreatic Adenocarcinoma (PAAD)** - one of the most lethal malignancies with poor prognosis

We systematically evaluated the impact of class imbalance on model performance and identified top discriminative genes using SHAP importance scores.

---

## 2. Materials and Methods

### 2.1 Data Acquisition

RNA-seq data was downloaded from the Genomic Data Commons (GDC) API using the following criteria:

| Parameter | Value |
|-----------|-------|
| Data Type | Gene Expression Quantification |
| Workflow | STAR - Counts |
| Access | Open |
| File Format | TSV (gzipped) |

**Table 1. Dataset Characteristics**

| Cancer Type | TCGA Project | Total Samples | Tumor | Normal | Normal % |
|-------------|--------------|---------------|-------|--------|----------|
| Breast | TCGA-BRCA | 299 | 272 | 27 | 9.0% |
| Pancreatic | TCGA-PAAD | 183 | 179 | 4 | 2.2% |

Sample type classification was based on TCGA barcode conventions:
- Codes 01-09: Tumor samples (Primary, Recurrent, Metastatic)
- Code 11: Solid Tissue Normal (adjacent normal tissue from cancer patients)

### 2.2 Data Preprocessing

The preprocessing pipeline consisted of:

1. **Low-expression filtering**: Genes with <10 counts in <20% of samples were removed
2. **Normalization**: Counts Per Million (CPM) normalization
3. **Transformation**: log2(CPM + 1) transformation for variance stabilization
4. **Feature Selection**: ANOVA F-test to select top 3,000 genes with highest tumor-normal differential expression
5. **Train-Test Split**: 80-20 stratified split

### 2.3 Model Training

**CatBoost Classifier** was selected for its:
- Robust handling of high-dimensional data
- Native categorical feature support
- Ordered boosting to reduce overfitting
- GPU acceleration capability

**Hyperparameter Optimization** was performed using Optuna with TPE (Tree-structured Parzen Estimator) sampler:

| Hyperparameter | Search Range |
|----------------|--------------|
| iterations | 100 - 1000 |
| depth | 3 - 10 |
| learning_rate | 0.01 - 0.3 |
| l2_leaf_reg | 1 - 10 |
| bagging_temperature | 0 - 1 |
| random_strength | 1e-9 - 10 |
| border_count | 32 - 255 |
| min_data_in_leaf | 1 - 100 |

Optimization objective: Maximize 5-fold cross-validation ROC-AUC.

### 2.4 Model Evaluation

Models were evaluated using:
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **5-Fold Stratified Cross-Validation**: For robust performance estimation

### 2.5 Model Interpretability

**SHAP TreeExplainer** was applied to quantify feature contributions:
- Global importance: Mean absolute SHAP values across all samples
- Local explanations: Per-sample feature contributions
- Summary plots: Distribution of SHAP values per feature

---

## 3. Results

### 3.1 Model Performance

**Table 2. Classification Performance Comparison**

| Metric | Breast Cancer (BRCA) | Pancreatic Cancer (PAAD) |
|--------|---------------------|-------------------------|
| Training Samples | 239 | 146 |
| Test Samples | 60 | 37 |
| Selected Genes | 3,000 | 3,000 |
| **CV ROC-AUC** | **0.9931 (±0.01)** | 0.9643 (±0.07) |
| **Test ROC-AUC** | **1.0000** | 0.5278 |
| **Test Accuracy** | **100%** | 97.30% |
| Test F1-Score | 1.0000 | 0.9863 |

The breast cancer classifier achieved near-perfect performance across all metrics. The pancreatic cancer classifier showed high CV performance but degraded test AUC due to having only 1 normal sample in the test set.

### 3.2 Optimal Hyperparameters

**Table 3. Best Hyperparameters from Optuna Optimization**

| Parameter | BRCA | PAAD |
|-----------|------|------|
| iterations | 515 | 158 |
| depth | 4 | 10 |
| learning_rate | 0.071 | 0.267 |
| l2_leaf_reg | 1.12 | 1.71 |
| bagging_temperature | 0.44 | 0.30 |
| min_data_in_leaf | 80 | 45 |

The BRCA model converged to a shallower tree (depth=4) with more iterations, suggesting smoother decision boundaries. The PAAD model required deeper trees, possibly to memorize the limited normal samples.

### 3.3 Top Discriminative Genes (SHAP Analysis)

**Table 4. Top 10 Genes by Mean |SHAP| - Breast Cancer**

| Rank | Ensembl ID | Mean |SHAP| |
|------|------------|-------------|
| 1 | ENSG00000124212.6 | 0.5003 |
| 2 | ENSG00000160753.16 | 0.3015 |
| 3 | ENSG00000168497.5 | 0.2953 |
| 4 | ENSG00000165028.12 | 0.2224 |
| 5 | ENSG00000161649.13 | 0.2127 |
| 6 | ENSG00000168079.17 | 0.1939 |
| 7 | ENSG00000148541.13 | 0.1156 |
| 8 | ENSG00000123500.10 | 0.1144 |
| 9 | ENSG00000235904.3 | 0.0984 |
| 10 | ENSG00000277954.1 | 0.0976 |

**Table 5. Top 10 Genes by Mean |SHAP| - Pancreatic Cancer**

| Rank | Ensembl ID | Mean |SHAP| |
|------|------------|-------------|
| 1 | ENSG00000253522.6 | 0.1305 |
| 2 | ENSG00000169884.14 | 0.1062 |
| 3 | ENSG00000250144.1 | 0.0734 |
| 4 | ENSG00000286747.1 | 0.0697 |
| 5 | ENSG00000265206.5 | 0.0624 |
| 6 | ENSG00000089356.18 | 0.0598 |
| 7 | ENSG00000216866.5 | 0.0565 |
| 8 | ENSG00000136014.12 | 0.0523 |
| 9 | ENSG00000250986.1 | 0.0515 |
| 10 | ENSG00000166428.14 | 0.0497 |

### 3.4 Impact of Class Imbalance

The stark performance difference between BRCA (AUC=1.0) and PAAD (AUC=0.53) classifiers illustrates the critical importance of balanced class representation:

**Figure 1. Class Distribution Comparison**

```
BRCA:  Tumor ████████████████████████████ 272 (91%)
       Normal ███ 27 (9%)

PAAD:  Tumor ████████████████████████████████ 179 (98%)
       Normal █ 4 (2%)
```

With only 4 normal samples in PAAD, the model cannot learn robust normal-tissue patterns, leading to:
- High training/CV performance (overfitting to few normals)
- Poor generalization (test set had only 1 normal sample)
- Unreliable feature importance rankings

---

## 4. Discussion

### 4.1 Clinical Implications

The breast cancer classifier demonstrates clinical utility for:
- **Quality Control**: Validating tissue sample identity
- **Biomarker Discovery**: SHAP-identified genes as candidate markers
- **Research Applications**: Batch effect detection, sample contamination screening

The pancreatic cancer classifier, despite its limitations, achieved 97.3% accuracy by correctly classifying all tumor samples. However, its inability to reliably identify normal tissue limits clinical applicability.

### 4.2 Limitations

1. **TCGA Normal Samples**: "Normal" samples are adjacent tissues from cancer patients, not healthy individuals. They may harbor field effects or pre-malignant changes.

2. **Class Imbalance**: Severe imbalance in PAAD (2.2% normal) prevented robust classifier development.

3. **Single Data Source**: Models trained on TCGA data may not generalize to other cohorts due to batch effects.

4. **Gene-Level Analysis**: Isoform-level or pathway-level features were not considered.

### 4.3 Recommendations for Low-Normal Cancers

For cancer types with insufficient normal samples, we recommend:

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **GTEx Integration** | Use healthy tissue from GTEx | True normals | Batch effects |
| **Subtype Comparison** | Compare molecular subtypes | No normals needed | Different question |
| **Pan-Cancer Normal** | Pool normals across cancers | More samples | Tissue heterogeneity |
| **Unsupervised Learning** | Clustering, dimensionality reduction | No labels needed | No direct classification |

### 4.4 Future Directions

1. **Multi-omic Integration**: Combine RNA-seq with methylation, mutation, and protein data
2. **Deep Learning**: Explore transformer architectures for gene expression
3. **Transfer Learning**: Pre-train on large cohorts, fine-tune for rare cancers
4. **Prospective Validation**: Test models on independent clinical cohorts

---

## 5. Conclusions

We developed interpretable machine learning classifiers for breast and pancreatic cancer using TCGA RNA-seq data. The breast cancer classifier achieved perfect test performance (AUC=1.0), demonstrating the feasibility of ML-based tissue classification when adequate normal samples are available. The pancreatic cancer classifier highlighted the critical challenge of class imbalance in clinical ML applications.

SHAP analysis provided biologically interpretable feature importance rankings, offering candidate genes for further validation. Future work should focus on integrating external normal tissue data (e.g., GTEx) and validating models on independent cohorts.

---

## 6. Methods Availability

All code is available at: `VectorDB_BioInsight/rnaseq_pipeline/ml/`

**Key Files:**
- `tcga_downloader.py` - TCGA data acquisition
- `preprocessor.py` - Data preprocessing pipeline
- `trainer.py` - CatBoost training with Optuna
- `explainer.py` - SHAP analysis module
- `predictor.py` - Inference API

**Trained Models:**
- `models/rnaseq/breast/` - BRCA classifier
- `models/rnaseq/pancreatic/` - PAAD classifier

---

## References

1. The Cancer Genome Atlas Research Network. Comprehensive molecular portraits of human breast tumours. Nature. 2012;490(7418):61-70.

2. Raphael BJ, et al. Integrated Genomic Characterization of Pancreatic Ductal Adenocarcinoma. Cancer Cell. 2017;32(2):185-203.

3. Prokhorenkova L, et al. CatBoost: unbiased boosting with categorical features. NeurIPS. 2018.

4. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS. 2017.

5. GTEx Consortium. The Genotype-Tissue Expression (GTEx) project. Nat Genet. 2013;45(6):580-585.

---

## Supplementary Information

### S1. SHAP Summary Plots

SHAP summary plots are available at:
- `models/rnaseq/breast/plots/shap_summary.png`
- `models/rnaseq/pancreatic/plots/shap_summary.png`

### S2. Complete Gene Lists

Full SHAP importance rankings (top 50 genes) are available at:
- `models/rnaseq/breast/shap_importance.csv`
- `models/rnaseq/pancreatic/shap_importance.csv`

### S3. Model Metadata

Detailed training metadata including all hyperparameters:
- `models/rnaseq/breast/model_metadata.json`
- `models/rnaseq/pancreatic/model_metadata.json`

---

*Report generated: 2026-01-09*
*BioInsight AI - RNA-seq Classification Pipeline v1.0*
