#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BioInsight AI - GCP Debian Server Setup Script
# ═══════════════════════════════════════════════════════════════
#
# 사용법:
#   chmod +x setup-server.sh
#   ./setup-server.sh
#
# 테스트 환경: Debian 12 (Bookworm) on GCP
# 재실행 가능: 이미 설치된 것은 스킵됨
# ═══════════════════════════════════════════════════════════════

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
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1 - 이미 설치됨"; }

# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════

PYTHON_VERSION="3.11.9"
NODE_VERSION="20"
APP_DIR="/opt/bioinsight"

# ═══════════════════════════════════════════════════════════════
# Step 0: 기존 잘못된 설정 정리
# ═══════════════════════════════════════════════════════════════

log_info "Step 0: 기존 설정 정리"

# 문제 있는 CRAN 저장소 제거
if [ -f /etc/apt/sources.list.d/cran.list ]; then
    log_info "문제 있는 CRAN 저장소 제거"
    sudo rm -f /etc/apt/sources.list.d/cran.list
    sudo rm -f /etc/apt/sources.list.d/cran*.list
    sudo rm -f /etc/apt/trusted.gpg.d/cran*
fi

# sources.list에서 CRAN 제거
if grep -q "cloud.r-project" /etc/apt/sources.list 2>/dev/null; then
    log_info "sources.list에서 CRAN 제거"
    sudo sed -i '/cloud.r-project/d' /etc/apt/sources.list
fi

log_success "정리 완료"

# ═══════════════════════════════════════════════════════════════
# Step 1: 시스템 패키지 업데이트
# ═══════════════════════════════════════════════════════════════

log_info "Step 1: 시스템 패키지 업데이트"
sudo apt-get update
sudo apt-get upgrade -y
log_success "시스템 업데이트 완료"

# ═══════════════════════════════════════════════════════════════
# Step 2: 필수 시스템 패키지 설치
# ═══════════════════════════════════════════════════════════════

log_info "Step 2: 필수 시스템 패키지 설치"
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    libhdf5-dev \
    libcurl4-openssl-dev

log_success "시스템 패키지 설치 완료"

# ═══════════════════════════════════════════════════════════════
# Step 3: Python 3.11 설치 (pyenv)
# ═══════════════════════════════════════════════════════════════

log_info "Step 3: Python 3.11 설치 (pyenv)"

# pyenv 설치
if [ ! -d "$HOME/.pyenv" ]; then
    log_info "pyenv 설치 중..."
    curl https://pyenv.run | bash
else
    log_skip "pyenv"
fi

# pyenv 환경 설정
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# .bashrc에 추가
if ! grep -q "pyenv" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
fi

# Python 설치 (이미 있으면 스킵)
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    log_info "Python $PYTHON_VERSION 설치 중... (5-10분 소요)"
    pyenv install -s $PYTHON_VERSION
else
    log_skip "Python $PYTHON_VERSION"
fi

pyenv global $PYTHON_VERSION
log_success "Python 설정 완료: $(python --version)"

# ═══════════════════════════════════════════════════════════════
# Step 4: Node.js 20 설치 (NodeSource)
# ═══════════════════════════════════════════════════════════════

log_info "Step 4: Node.js $NODE_VERSION 설치"

if ! command -v node &> /dev/null; then
    log_info "Node.js 설치 중..."

    # NodeSource 저장소 추가
    sudo mkdir -p /etc/apt/keyrings

    # 기존 키 있으면 삭제 후 재생성
    sudo rm -f /etc/apt/keyrings/nodesource.gpg
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_VERSION.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list

    sudo apt-get update
    sudo apt-get install -y nodejs
else
    log_skip "Node.js $(node --version)"
fi

log_success "Node.js 설정 완료: $(node --version), npm $(npm --version)"

# ═══════════════════════════════════════════════════════════════
# Step 5: R 설치 (Debian 기본 저장소 - 안정적)
# ═══════════════════════════════════════════════════════════════

log_info "Step 5: R 설치 (Debian 기본 저장소)"

if ! command -v R &> /dev/null; then
    log_info "R 설치 중..."

    # Debian 기본 저장소에서 R 설치 (4.2.x, DESeq2에 충분)
    sudo apt-get install -y r-base r-base-dev
else
    log_skip "R $(R --version 2>&1 | head -1 | awk '{print $3}')"
fi

log_success "R 설정 완료"

# ═══════════════════════════════════════════════════════════════
# Step 6: R 패키지 설치 (DESeq2) - 선택사항
# ═══════════════════════════════════════════════════════════════

log_info "Step 6: R 패키지 확인 (DESeq2)"

# DESeq2 설치 여부 확인
if Rscript -e 'if(!requireNamespace("DESeq2", quietly=TRUE)) quit(status=1)' 2>/dev/null; then
    log_skip "DESeq2"
else
    log_warning "DESeq2 미설치 - 나중에 설치하려면:"
    echo "    sudo Rscript -e 'BiocManager::install(\"DESeq2\")'"
    echo ""

    # 자동 설치 시도 (실패해도 계속 진행)
    log_info "DESeq2 설치 시도 중... (10-20분 소요, 실패해도 계속 진행)"
    sudo Rscript -e '
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager", repos="https://cloud.r-project.org")
    tryCatch({
        BiocManager::install(c("DESeq2", "apeglm"), ask = FALSE, update = FALSE)
    }, error = function(e) {
        message("DESeq2 설치 실패 - 나중에 수동 설치 필요")
    })
    ' || log_warning "DESeq2 설치 실패 - 나중에 수동 설치 필요"
fi

# ═══════════════════════════════════════════════════════════════
# Step 7: Nginx 설치
# ═══════════════════════════════════════════════════════════════

log_info "Step 7: Nginx 설치"

if ! command -v nginx &> /dev/null; then
    sudo apt-get install -y nginx
    sudo systemctl enable nginx
    sudo systemctl start nginx
else
    log_skip "Nginx"
fi

log_success "Nginx 설정 완료"

# ═══════════════════════════════════════════════════════════════
# Step 8: 애플리케이션 디렉토리 생성
# ═══════════════════════════════════════════════════════════════

log_info "Step 8: 애플리케이션 디렉토리 생성"

if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p $APP_DIR
    sudo chown $USER:$USER $APP_DIR
    log_success "디렉토리 생성: $APP_DIR"
else
    log_skip "$APP_DIR 디렉토리"
fi

# ═══════════════════════════════════════════════════════════════
# 완료 메시지
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
log_success "서버 기본 환경 설정 완료!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "설치된 소프트웨어:"
echo "  - Python: $(python --version 2>&1)"
echo "  - Node.js: $(node --version 2>&1)"
echo "  - npm: $(npm --version 2>&1)"
echo "  - R: $(R --version 2>&1 | head -1)"
echo "  - Nginx: $(nginx -v 2>&1)"
echo ""
echo "다음 단계:"
echo "  1. 터미널 재시작 또는: source ~/.bashrc"
echo "  2. 앱 코드 배포: ./deploy-app.sh"
echo ""
