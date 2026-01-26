# BioInsight AI

AI-powered integrated research platform for Bio & Healthcare researchers.

> **"Accelerate discoveries while keeping researchers in control"**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Features
- **Paper RAG Search**: Semantic search across biomedical literature using PubMedBERT embeddings
- **Real-time PubMed Search**: Live search with Korean↔English translation
- **Knowledge Graph**: 3D interactive visualization of paper/gene/disease relationships
- **Daily Briefing**: Automated news digest from FDA, ClinicalTrials, bioRxiv

### RNA-seq Analysis Pipeline
- **6-Agent Bulk RNA-seq Pipeline**: DEG → Network → Pathway → Validation → Visualization → Report
- **Single-cell RNA-seq**: QC → Clustering → Cell Type Annotation → Report
- **ML Cancer Classification**: Pan-cancer classifier (17 cancer types, AUC 0.988)
- **RAG-based Interpretation**: Literature-backed gene interpretations with PMID citations

## Quick Start

### Prerequisites

- Python 3.11+
- R 4.3+ (for DESeq2)
- Node.js 18+ (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/bioinsight-ai.git
cd bioinsight-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-rnaseq.txt

# Install R packages (required for Bulk RNA-seq)
Rscript -e "install.packages('BiocManager'); BiocManager::install(c('DESeq2', 'apeglm'))"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
# Start backend server
uvicorn backend.app.main:app --reload --port 8000

# Start frontend (in another terminal)
cd frontend/react_app
npm install
npm run dev
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f backend
```

## RNA-seq Pipeline Usage

### Bulk RNA-seq Analysis

```python
from rnaseq_pipeline.orchestrator import RNAseqPipeline

# Initialize pipeline
pipeline = RNAseqPipeline(
    input_dir='data/my_experiment',
    output_dir='results/my_experiment',
    config={'cancer_type': 'breast_cancer'},
    pipeline_type='bulk'
)

# Run full analysis
result = pipeline.run()

# View report
print(f"Report: {result['run_dir']}/agent6_report/report.html")
```

### Input Files Required

```
data/my_experiment/
├── count_matrix.csv    # Genes (rows) x Samples (columns)
├── metadata.csv        # sample_id, condition, batch
└── config.json         # contrast, cancer_type, cutoffs
```

### Example `metadata.csv`
```csv
sample_id,condition,batch
SAMPLE_01,tumor,batch1
SAMPLE_02,tumor,batch1
SAMPLE_03,normal,batch1
SAMPLE_04,normal,batch1
```

### Example `config.json`
```json
{
  "contrast": ["tumor", "normal"],
  "cancer_type": "breast_cancer",
  "padj_cutoff": 0.05,
  "log2fc_cutoff": 1.0
}
```

## Project Structure

```
bioinsight-ai/
├── backend/app/           # FastAPI backend
│   ├── api/routes/        # API endpoints
│   └── core/              # Core services
├── frontend/react_app/    # React frontend
├── rnaseq_pipeline/       # RNA-seq analysis
│   ├── agents/            # 6-Agent pipeline
│   ├── ml/                # ML models
│   └── rag/               # RAG interpretation
├── models/rnaseq/         # Pre-trained models
├── tests/                 # Test suite
├── Dockerfile             # Container config
└── docker-compose.yml     # Multi-service setup
```

## API Documentation

Once running, access the API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rnaseq/upload` | POST | Upload RNA-seq data |
| `/api/rnaseq/start/{job_id}` | POST | Start pipeline |
| `/api/rnaseq/progress/{job_id}` | GET | SSE progress stream |
| `/api/paper/search` | GET | Search papers |
| `/api/chat/ask` | POST | RAG Q&A |

## Environment Variables

```bash
# LLM APIs (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional
NCBI_API_KEY=...  # For PubMed access
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rnaseq_pipeline --cov-report=html
```

## Performance Benchmarks

| Pipeline | Dataset | Time | Output |
|----------|---------|------|--------|
| Bulk RNA-seq (6-Agent) | TCGA BRCA 50 samples | ~18 min | 9.4MB HTML report |
| ML Prediction | Single sample | < 1 sec | Cancer type + confidence |
| RAG Interpretation | 20 genes | ~2 min | Literature-backed analysis |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BioInsight AI in your research, please cite:

```bibtex
@software{bioinsight_ai,
  title = {BioInsight AI: AI-powered Research Platform for Bioinformatics},
  year = {2026},
  url = {https://github.com/your-org/bioinsight-ai}
}
```

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/your-org/bioinsight-ai/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/bioinsight-ai/discussions)
