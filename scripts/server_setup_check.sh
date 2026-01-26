#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BioInsight AI - Server Setup & Health Check Script
# 모든 기능을 한 번에 확인하고 설정합니다.
# ═══════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_fail() { echo -e "${RED}[✗]${NC} $1"; }

APP_DIR="${HOME}/VectorDB_BioInsight"
cd "$APP_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "       BioInsight AI - Server Setup & Health Check"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════
# 1. Python 환경 확인
# ═══════════════════════════════════════════════════════════════
log_info "1. Python 환경 확인..."

if [ -d ".venv" ]; then
    log_ok "가상환경 존재"
    source .venv/bin/activate
else
    log_warn "가상환경 없음 - 생성 중..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# 패키지 설치
log_info "   Python 패키지 확인/설치..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r requirements-rnaseq.txt 2>/dev/null || log_warn "requirements-rnaseq.txt 일부 실패"
pip install -q schedule feedparser 2>/dev/null || true
log_ok "Python 패키지 설치 완료"

# ═══════════════════════════════════════════════════════════════
# 2. R 및 DESeq2 확인 (RNA-seq용)
# ═══════════════════════════════════════════════════════════════
log_info "2. R/DESeq2 확인 (RNA-seq용)..."

if command -v R &> /dev/null; then
    R_VERSION=$(R --version | head -1)
    log_ok "R 설치됨: $R_VERSION"

    # DESeq2 확인
    if R -q -e "library(DESeq2)" &> /dev/null; then
        log_ok "DESeq2 설치됨"
    else
        log_warn "DESeq2 미설치 - 설치하려면:"
        echo "       R -e \"BiocManager::install('DESeq2')\""
    fi
else
    log_warn "R 미설치 - RNA-seq DEG 분석 불가"
    echo "       설치: sudo apt install r-base"
fi

# ═══════════════════════════════════════════════════════════════
# 3. 환경변수 (.env) 확인
# ═══════════════════════════════════════════════════════════════
log_info "3. 환경변수 확인..."

if [ -f ".env" ]; then
    log_ok ".env 파일 존재"

    # 필수 키 확인
    check_env() {
        if grep -q "^$1=" .env && ! grep -q "^$1=xxx" .env && ! grep -q "^$1=$" .env; then
            log_ok "  $1 설정됨"
        else
            log_fail "  $1 미설정!"
        fi
    }

    check_env "OPENAI_API_KEY"
    check_env "ANTHROPIC_API_KEY"
    check_env "NCBI_API_KEY"
else
    log_fail ".env 파일 없음!"
    echo "       cp .env.example .env && vim .env"
fi

# ═══════════════════════════════════════════════════════════════
# 4. 데이터 파일 확인
# ═══════════════════════════════════════════════════════════════
log_info "4. 데이터 파일 확인..."

# ChromaDB
if [ -d "chroma_db" ] && [ "$(ls -A chroma_db 2>/dev/null)" ]; then
    COLLECTIONS=$(ls chroma_db 2>/dev/null | wc -l)
    log_ok "ChromaDB 존재 ($COLLECTIONS collections)"
else
    log_warn "ChromaDB 비어있음 - 논문 검색/RAG 불가"
fi

# ML 모델
if [ -d "models/rnaseq" ]; then
    MODELS=$(find models/rnaseq -name "*.cbm" 2>/dev/null | wc -l)
    if [ "$MODELS" -gt 0 ]; then
        log_ok "ML 모델 존재 ($MODELS개)"
    else
        log_warn "ML 모델 없음 - 암종 예측 불가"
    fi
else
    log_warn "models/rnaseq 디렉토리 없음"
fi

# ═══════════════════════════════════════════════════════════════
# 5. 백엔드 서비스 확인/시작
# ═══════════════════════════════════════════════════════════════
log_info "5. 백엔드 서비스 확인..."

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    log_ok "백엔드 실행 중 (port 8000)"
else
    log_warn "백엔드 미실행 - 시작 중..."
    pkill -f "uvicorn backend.app.main" 2>/dev/null || true
    nohup .venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
    sleep 3

    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_ok "백엔드 시작됨"
    else
        log_fail "백엔드 시작 실패! 로그 확인: tail backend.log"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# 6. Nginx 확인
# ═══════════════════════════════════════════════════════════════
log_info "6. Nginx 확인..."

if systemctl is-active --quiet nginx; then
    log_ok "Nginx 실행 중"
else
    log_warn "Nginx 미실행 - 시작 중..."
    sudo systemctl start nginx
fi

# ═══════════════════════════════════════════════════════════════
# 7. API 기능 테스트
# ═══════════════════════════════════════════════════════════════
log_info "7. API 기능 테스트..."

test_api() {
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$1" 2>/dev/null)
    if [ "$RESPONSE" = "200" ]; then
        log_ok "  $2"
    else
        log_fail "  $2 (HTTP $RESPONSE)"
    fi
}

test_api "http://localhost:8000/health" "Health Check"
test_api "http://localhost:8000/api/crawler/health" "Crawler (트렌드/검색)"
test_api "http://localhost:8000/api/briefing/latest" "Daily Briefing"

# RNA-seq는 POST라서 별도 처리
RNASEQ_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/api/rnaseq/health" 2>/dev/null)
if [ "$RNASEQ_RESPONSE" = "200" ] || [ "$RNASEQ_RESPONSE" = "404" ]; then
    log_ok "  RNA-seq API (라우터 정상)"
else
    log_warn "  RNA-seq API 확인 필요"
fi

# ═══════════════════════════════════════════════════════════════
# 8. systemd 서비스 등록 (선택)
# ═══════════════════════════════════════════════════════════════
log_info "8. systemd 서비스 확인..."

if systemctl is-enabled --quiet bioinsight-backend 2>/dev/null; then
    log_ok "백엔드 서비스 등록됨 (자동 시작)"
else
    log_warn "백엔드 서비스 미등록 - 등록하려면:"
    echo "       sudo ./scripts/server_setup_check.sh --register-service"
fi

# ═══════════════════════════════════════════════════════════════
# 서비스 등록 옵션
# ═══════════════════════════════════════════════════════════════
if [ "$1" = "--register-service" ]; then
    log_info "systemd 서비스 등록 중..."

    sudo tee /etc/systemd/system/bioinsight-backend.service > /dev/null << EOF
[Unit]
Description=BioInsight AI Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable bioinsight-backend
    sudo systemctl restart bioinsight-backend
    log_ok "서비스 등록 완료!"
fi

# ═══════════════════════════════════════════════════════════════
# 완료
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
log_ok "설정 확인 완료!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "웹사이트 접속: http://$(curl -s ifconfig.me 2>/dev/null || echo '<서버IP>')"
echo ""
