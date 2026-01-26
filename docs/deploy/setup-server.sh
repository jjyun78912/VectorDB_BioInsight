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
# ═══════════════════════════════════════════════════════════════

set -e  # 오류 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════

PYTHON_VERSION="3.11.9"
NODE_VERSION="20"
APP_DIR="/opt/bioinsight"
APP_USER="bioinsight"

# ═══════════════════════════════════════════════════════════════
# Step 1: 시스템 패키지 업데이트
# ═══════════════════════════════════════════════════════════════

log_info "Step 1: 시스템 패키지 업데이트"
sudo apt-get update
sudo apt-get upgrade -y

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
    curl https://pyenv.run | bash
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

# Python 설치
pyenv install -s $PYTHON_VERSION
pyenv global $PYTHON_VERSION

log_success "Python $PYTHON_VERSION 설치 완료"
python --version

# ═══════════════════════════════════════════════════════════════
# Step 4: Node.js 20 설치 (NodeSource)
# ═══════════════════════════════════════════════════════════════

log_info "Step 4: Node.js $NODE_VERSION 설치"

# NodeSource 저장소 추가
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_VERSION.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list

sudo apt-get update
sudo apt-get install -y nodejs

log_success "Node.js 설치 완료"
node --version
npm --version

# ═══════════════════════════════════════════════════════════════
# Step 5: R 4.x 설치
# ═══════════════════════════════════════════════════════════════

log_info "Step 5: R 설치"

# R CRAN 저장소 추가
sudo apt-get install -y dirmngr
wget -qO- https://cloud.r-project.org/bin/linux/debian/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_debian_key.asc

# Debian 버전에 따라 저장소 추가
DEBIAN_VERSION=$(lsb_release -cs)
echo "deb http://cloud.r-project.org/bin/linux/debian ${DEBIAN_VERSION}-cran40/" | sudo tee /etc/apt/sources.list.d/cran.list

sudo apt-get update
sudo apt-get install -y r-base r-base-dev

log_success "R 설치 완료"
R --version | head -1

# ═══════════════════════════════════════════════════════════════
# Step 6: R 패키지 설치 (DESeq2)
# ═══════════════════════════════════════════════════════════════

log_info "Step 6: R 패키지 설치 (DESeq2) - 시간이 걸릴 수 있습니다"

# Bioconductor 및 DESeq2 설치
sudo Rscript -e '
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos="https://cloud.r-project.org")
BiocManager::install(c("DESeq2", "apeglm"), ask = FALSE, update = FALSE)
'

log_success "R 패키지 설치 완료"

# ═══════════════════════════════════════════════════════════════
# Step 7: Nginx 설치
# ═══════════════════════════════════════════════════════════════

log_info "Step 7: Nginx 설치"

sudo apt-get install -y nginx

sudo systemctl enable nginx
sudo systemctl start nginx

log_success "Nginx 설치 완료"

# ═══════════════════════════════════════════════════════════════
# Step 8: 애플리케이션 디렉토리 생성
# ═══════════════════════════════════════════════════════════════

log_info "Step 8: 애플리케이션 디렉토리 생성"

# 앱 디렉토리 생성
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

log_success "디렉토리 생성 완료: $APP_DIR"

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
echo "  - Node.js: $(node --version)"
echo "  - npm: $(npm --version)"
echo "  - R: $(R --version 2>&1 | head -1)"
echo "  - Nginx: $(nginx -v 2>&1)"
echo ""
echo "다음 단계:"
echo "  1. 터미널 재시작 또는: source ~/.bashrc"
echo "  2. 앱 코드 배포: ./deploy-app.sh"
echo ""
