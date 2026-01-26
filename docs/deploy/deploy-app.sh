#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BioInsight AI - Application Deployment Script
# ═══════════════════════════════════════════════════════════════
#
# 사용법:
#   chmod +x deploy-app.sh
#   ./deploy-app.sh
#
# 사전 조건: setup-server.sh 실행 완료
# ═══════════════════════════════════════════════════════════════

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════

APP_DIR="/opt/bioinsight"
REPO_URL="https://github.com/jjyun78912/VectorDB_BioInsight.git"
DOMAIN=""  # 나중에 설정

# pyenv 로드
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# ═══════════════════════════════════════════════════════════════
# Step 1: 코드 클론
# ═══════════════════════════════════════════════════════════════

log_info "Step 1: 코드 클론"

cd $APP_DIR

if [ -d "VectorDB_BioInsight" ]; then
    log_info "기존 코드 업데이트"
    cd VectorDB_BioInsight
    git pull origin main
else
    log_info "새로 클론"
    git clone $REPO_URL
    cd VectorDB_BioInsight
fi

log_success "코드 준비 완료"

# ═══════════════════════════════════════════════════════════════
# Step 2: Python 가상환경 설정
# ═══════════════════════════════════════════════════════════════

log_info "Step 2: Python 가상환경 설정"

# 가상환경 생성
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate

# pip 업그레이드
pip install --upgrade pip wheel setuptools

# 패키지 설치
log_info "Python 패키지 설치 중... (시간이 걸릴 수 있습니다)"
pip install -r requirements.txt
pip install -r requirements-rnaseq.txt

log_success "Python 환경 설정 완료"

# ═══════════════════════════════════════════════════════════════
# Step 3: Frontend 빌드
# ═══════════════════════════════════════════════════════════════

log_info "Step 3: Frontend 빌드"

cd frontend/react_app

npm install
npm run build

cd ../..

log_success "Frontend 빌드 완료"

# ═══════════════════════════════════════════════════════════════
# Step 4: 환경 변수 파일 생성
# ═══════════════════════════════════════════════════════════════

log_info "Step 4: 환경 변수 파일 확인"

if [ ! -f ".env" ]; then
    log_warning ".env 파일이 없습니다. 템플릿 생성 중..."
    cat > .env << 'EOF'
# ═══════════════════════════════════════════════════════════════
# BioInsight AI - Environment Variables
# ═══════════════════════════════════════════════════════════════

# LLM API Keys (필수)
OPENAI_API_KEY=sk-proj-xxx
ANTHROPIC_API_KEY=sk-ant-xxx

# Google API (권장)
GOOGLE_API_KEY=AIzaSy...

# NCBI/PubMed (권장)
NCBI_API_KEY=xxx

# Supabase (선택)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=xxx

# 앱 설정
CHROMA_PERSIST_DIR=/opt/bioinsight/VectorDB_BioInsight/chroma_db
LOG_LEVEL=INFO
EOF
    log_warning "⚠️  .env 파일에 API 키를 설정하세요!"
    log_warning "    vim $APP_DIR/VectorDB_BioInsight/.env"
else
    log_success ".env 파일 존재함"
fi

# ═══════════════════════════════════════════════════════════════
# Step 5: systemd 서비스 생성
# ═══════════════════════════════════════════════════════════════

log_info "Step 5: systemd 서비스 설정"

# Backend 서비스
sudo tee /etc/systemd/system/bioinsight-backend.service > /dev/null << EOF
[Unit]
Description=BioInsight AI Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR/VectorDB_BioInsight
Environment="PATH=$APP_DIR/VectorDB_BioInsight/.venv/bin"
EnvironmentFile=$APP_DIR/VectorDB_BioInsight/.env
ExecStart=$APP_DIR/VectorDB_BioInsight/.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bioinsight-backend

log_success "systemd 서비스 설정 완료"

# ═══════════════════════════════════════════════════════════════
# Step 6: Nginx 설정
# ═══════════════════════════════════════════════════════════════

log_info "Step 6: Nginx 설정"

sudo tee /etc/nginx/sites-available/bioinsight > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;  # 모든 도메인 또는 IP

    # Frontend (정적 파일)
    location / {
        root /opt/bioinsight/VectorDB_BioInsight/frontend/react_app/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }

    # SSE (Server-Sent Events) for RNA-seq progress
    location /api/rnaseq/progress/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
        proxy_read_timeout 86400s;
    }

    # 파일 업로드 크기 제한
    client_max_body_size 100M;
}
EOF

# 사이트 활성화
sudo ln -sf /etc/nginx/sites-available/bioinsight /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Nginx 설정 테스트
sudo nginx -t

sudo systemctl reload nginx

log_success "Nginx 설정 완료"

# ═══════════════════════════════════════════════════════════════
# 완료
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
log_success "앱 배포 완료!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "다음 단계:"
echo ""
echo "1. API 키 설정:"
echo "   vim $APP_DIR/VectorDB_BioInsight/.env"
echo ""
echo "2. 데이터 파일 업로드 (로컬에서):"
echo "   scp -r chroma_db/ user@server:$APP_DIR/VectorDB_BioInsight/"
echo "   scp -r models/ user@server:$APP_DIR/VectorDB_BioInsight/"
echo ""
echo "3. 백엔드 서비스 시작:"
echo "   sudo systemctl start bioinsight-backend"
echo "   sudo systemctl status bioinsight-backend"
echo ""
echo "4. 로그 확인:"
echo "   sudo journalctl -u bioinsight-backend -f"
echo ""
echo "5. 접속 테스트:"
echo "   curl http://localhost/health"
echo "   또는 브라우저에서 http://<서버IP>"
echo ""
