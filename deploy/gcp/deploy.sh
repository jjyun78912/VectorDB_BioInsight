#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BioInsight AI - GCP Cloud Run 배포 스크립트
# ═══════════════════════════════════════════════════════════════
#
# 사용법:
#   ./deploy.sh setup       # 최초 1회: 프로젝트 설정
#   ./deploy.sh build       # Docker 이미지 빌드 & 푸시
#   ./deploy.sh deploy      # Cloud Run 배포 (공개)
#   ./deploy.sh deploy-private  # Cloud Run 배포 (비공개 - 인증 필요)
#   ./deploy.sh all         # 전체 실행 (build + deploy)
#   ./deploy.sh add-user    # 접근 권한 사용자 추가
#   ./deploy.sh logs        # 로그 확인
#
# ═══════════════════════════════════════════════════════════════

set -e

# ─────────────────────────────────────────────────────────────────
# 설정 (수정 필요)
# ─────────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-asia-northeast3}"  # 서울 리전
BUCKET_NAME="${GCS_BUCKET_NAME:-bioinsight-data}"

# 서비스 이름
BACKEND_SERVICE="bioinsight-backend"
FRONTEND_SERVICE="bioinsight-frontend"

# 접근 모드: "public" 또는 "private"
ACCESS_MODE="${ACCESS_MODE:-public}"

# 허용할 사용자 목록 (private 모드에서 사용, 쉼표로 구분)
# 예: "user1@gmail.com,user2@gmail.com"
ALLOWED_USERS="${ALLOWED_USERS:-}"

# Artifact Registry
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/bioinsight"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ─────────────────────────────────────────────────────────────────
# 사전 체크
# ─────────────────────────────────────────────────────────────────
check_prerequisites() {
    log_info "사전 요구사항 확인 중..."

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI가 설치되지 않았습니다"
        echo "설치: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되지 않았습니다"
        exit 1
    fi

    # 로그인 확인
    if ! gcloud auth print-identity-token &> /dev/null; then
        log_warn "GCP 로그인 필요"
        gcloud auth login
    fi

    log_info "사전 요구사항 확인 완료 ✓"
}

# ─────────────────────────────────────────────────────────────────
# 최초 설정
# ─────────────────────────────────────────────────────────────────
setup() {
    log_info "=== GCP 프로젝트 초기 설정 ==="

    # 프로젝트 설정
    gcloud config set project $PROJECT_ID

    # API 활성화
    log_info "필요한 API 활성화 중..."
    gcloud services enable \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com \
        secretmanager.googleapis.com \
        storage.googleapis.com

    # Artifact Registry 저장소 생성
    log_info "Artifact Registry 저장소 생성..."
    gcloud artifacts repositories create bioinsight \
        --repository-format=docker \
        --location=$REGION \
        --description="BioInsight AI Docker images" \
        2>/dev/null || log_warn "저장소가 이미 존재합니다"

    # Docker 인증 설정
    log_info "Docker 인증 설정..."
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

    # Cloud Storage 버킷 생성
    log_info "Cloud Storage 버킷 생성..."
    gsutil mb -l $REGION gs://$BUCKET_NAME 2>/dev/null || log_warn "버킷이 이미 존재합니다"

    # Secret Manager에 API 키 등록
    setup_secrets

    log_info "=== 초기 설정 완료 ✓ ==="
}

# ─────────────────────────────────────────────────────────────────
# Secret Manager 설정
# ─────────────────────────────────────────────────────────────────
setup_secrets() {
    log_info "Secret Manager 설정..."

    # .env 파일에서 시크릿 읽기
    if [ -f "../../.env" ]; then
        while IFS='=' read -r key value; do
            # 주석, 빈 줄 스킵
            [[ $key =~ ^#.*$ ]] && continue
            [[ -z $key ]] && continue

            # API 키만 등록
            if [[ $key == *"API_KEY"* ]] || [[ $key == *"_KEY"* ]]; then
                # 기존 시크릿 삭제 (있으면)
                gcloud secrets delete $key --quiet 2>/dev/null || true

                # 새 시크릿 생성
                echo -n "$value" | gcloud secrets create $key \
                    --data-file=- \
                    --replication-policy="automatic" \
                    2>/dev/null && log_info "  ✓ $key" || log_warn "  ⚠ $key 이미 존재"
            fi
        done < "../../.env"
    else
        log_warn ".env 파일을 찾을 수 없습니다. 수동으로 시크릿을 설정하세요."
    fi
}

# ─────────────────────────────────────────────────────────────────
# 데이터 업로드
# ─────────────────────────────────────────────────────────────────
upload_data() {
    log_info "=== Cloud Storage에 데이터 업로드 ==="

    # 모델 업로드
    log_info "ML 모델 업로드 중..."
    gsutil -m rsync -r ../../models/rnaseq/pancancer/ gs://$BUCKET_NAME/models/rnaseq/pancancer/

    # ChromaDB 업로드
    log_info "ChromaDB 업로드 중..."
    gsutil -m rsync -r ../../chroma_db/ gs://$BUCKET_NAME/chroma_db/

    log_info "=== 데이터 업로드 완료 ✓ ==="
}

# ─────────────────────────────────────────────────────────────────
# Docker 이미지 빌드 & 푸시
# ─────────────────────────────────────────────────────────────────
build() {
    log_info "=== Docker 이미지 빌드 ==="

    cd ../..  # 프로젝트 루트로 이동

    # Backend 빌드
    log_info "Backend 이미지 빌드 중... (약 5-10분 소요)"
    docker build \
        -f deploy/gcp/Dockerfile.backend \
        -t ${REGISTRY}/backend:latest \
        -t ${REGISTRY}/backend:$(date +%Y%m%d-%H%M%S) \
        .

    # Frontend 빌드
    log_info "Frontend 이미지 빌드 중..."

    # Backend URL 가져오기 (이미 배포된 경우)
    BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE \
        --region=$REGION \
        --format='value(status.url)' 2>/dev/null || echo "http://localhost:8080")

    docker build \
        -f deploy/gcp/Dockerfile.frontend \
        --build-arg VITE_API_URL=$BACKEND_URL \
        -t ${REGISTRY}/frontend:latest \
        -t ${REGISTRY}/frontend:$(date +%Y%m%d-%H%M%S) \
        .

    # 이미지 푸시
    log_info "이미지를 Artifact Registry에 푸시 중..."
    docker push ${REGISTRY}/backend:latest
    docker push ${REGISTRY}/frontend:latest

    cd deploy/gcp
    log_info "=== 빌드 완료 ✓ ==="
}

# ─────────────────────────────────────────────────────────────────
# Cloud Run 배포 (공개)
# ─────────────────────────────────────────────────────────────────
deploy() {
    deploy_internal "public"
}

# ─────────────────────────────────────────────────────────────────
# Cloud Run 배포 (비공개 - Google 계정 인증 필요)
# ─────────────────────────────────────────────────────────────────
deploy_private() {
    deploy_internal "private"
}

# ─────────────────────────────────────────────────────────────────
# Cloud Run 배포 (내부 함수)
# ─────────────────────────────────────────────────────────────────
deploy_internal() {
    local mode="${1:-public}"

    if [ "$mode" == "private" ]; then
        log_info "=== Cloud Run 배포 (🔒 비공개 모드) ==="
        AUTH_FLAG="--no-allow-unauthenticated"
    else
        log_info "=== Cloud Run 배포 (🌐 공개 모드) ==="
        AUTH_FLAG="--allow-unauthenticated"
    fi

    # Backend 배포 (항상 비공개 - Frontend에서만 접근)
    log_info "Backend 배포 중..."
    gcloud run deploy $BACKEND_SERVICE \
        --image=${REGISTRY}/backend:latest \
        --region=$REGION \
        --platform=managed \
        --memory=8Gi \
        --cpu=4 \
        --timeout=3600 \
        --concurrency=10 \
        --min-instances=0 \
        --max-instances=5 \
        --set-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME,GCP_PROJECT_ID=$PROJECT_ID" \
        --set-secrets="OPENAI_API_KEY=OPENAI_API_KEY:latest,ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,GOOGLE_API_KEY=GOOGLE_API_KEY:latest,NCBI_API_KEY=NCBI_API_KEY:latest" \
        --allow-unauthenticated  # Backend는 Frontend에서 호출하므로 공개

    # Backend URL 가져오기
    BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE \
        --region=$REGION \
        --format='value(status.url)')

    log_info "Backend URL: $BACKEND_URL"

    # Frontend 재빌드 (Backend URL 포함)
    log_info "Frontend 재빌드 (Backend URL 포함)..."
    cd ../..
    docker build \
        -f deploy/gcp/Dockerfile.frontend \
        --build-arg VITE_API_URL=$BACKEND_URL \
        -t ${REGISTRY}/frontend:latest \
        .
    docker push ${REGISTRY}/frontend:latest
    cd deploy/gcp

    # Frontend 배포
    log_info "Frontend 배포 중..."
    gcloud run deploy $FRONTEND_SERVICE \
        --image=${REGISTRY}/frontend:latest \
        --region=$REGION \
        --platform=managed \
        --memory=512Mi \
        --cpu=1 \
        --timeout=60 \
        --concurrency=100 \
        --min-instances=0 \
        --max-instances=10 \
        $AUTH_FLAG

    # Frontend URL 가져오기
    FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE \
        --region=$REGION \
        --format='value(status.url)')

    echo ""
    log_info "═══════════════════════════════════════════════════════════"
    if [ "$mode" == "private" ]; then
        log_info "  🔒 배포 완료 (비공개 모드)"
        log_info "═══════════════════════════════════════════════════════════"
        echo ""
        echo "  Frontend: $FRONTEND_URL"
        echo "  Backend:  $BACKEND_URL"
        echo ""
        log_warn "  ⚠️  접근하려면 Google 계정 인증이 필요합니다!"
        echo ""
        echo "  사용자 추가 방법:"
        echo "    ./deploy.sh add-user user@gmail.com"
        echo ""
    else
        log_info "  🚀 배포 완료 (공개 모드)"
        log_info "═══════════════════════════════════════════════════════════"
        echo ""
        echo "  Frontend: $FRONTEND_URL"
        echo "  Backend:  $BACKEND_URL"
        echo "  API Docs: $BACKEND_URL/docs"
        echo ""
    fi
}

# ─────────────────────────────────────────────────────────────────
# 사용자 접근 권한 추가
# ─────────────────────────────────────────────────────────────────
add_user() {
    local email="$1"

    if [ -z "$email" ]; then
        log_error "이메일 주소를 입력하세요"
        echo "사용법: ./deploy.sh add-user user@gmail.com"
        exit 1
    fi

    log_info "사용자 추가 중: $email"

    # Frontend 접근 권한 부여
    gcloud run services add-iam-policy-binding $FRONTEND_SERVICE \
        --region=$REGION \
        --member="user:$email" \
        --role="roles/run.invoker"

    log_info "✓ $email 에게 접근 권한이 부여되었습니다"
    echo ""
    echo "이제 $email 계정으로 Google 로그인하면 사이트에 접속할 수 있습니다."
}

# ─────────────────────────────────────────────────────────────────
# 사용자 접근 권한 제거
# ─────────────────────────────────────────────────────────────────
remove_user() {
    local email="$1"

    if [ -z "$email" ]; then
        log_error "이메일 주소를 입력하세요"
        echo "사용법: ./deploy.sh remove-user user@gmail.com"
        exit 1
    fi

    log_info "사용자 제거 중: $email"

    gcloud run services remove-iam-policy-binding $FRONTEND_SERVICE \
        --region=$REGION \
        --member="user:$email" \
        --role="roles/run.invoker"

    log_info "✓ $email 의 접근 권한이 제거되었습니다"
}

# ─────────────────────────────────────────────────────────────────
# 허용된 사용자 목록 확인
# ─────────────────────────────────────────────────────────────────
list_users() {
    log_info "=== 허용된 사용자 목록 ==="
    echo ""

    gcloud run services get-iam-policy $FRONTEND_SERVICE \
        --region=$REGION \
        --format="table(bindings.members)" \
        2>/dev/null | grep "user:" || echo "허용된 사용자가 없습니다"

    echo ""
}

# ─────────────────────────────────────────────────────────────────
# 공개/비공개 모드 전환
# ─────────────────────────────────────────────────────────────────
make_public() {
    log_info "Frontend를 공개 모드로 전환 중..."

    gcloud run services add-iam-policy-binding $FRONTEND_SERVICE \
        --region=$REGION \
        --member="allUsers" \
        --role="roles/run.invoker"

    log_info "✓ 이제 누구나 접속할 수 있습니다"
}

make_private() {
    log_info "Frontend를 비공개 모드로 전환 중..."

    gcloud run services remove-iam-policy-binding $FRONTEND_SERVICE \
        --region=$REGION \
        --member="allUsers" \
        --role="roles/run.invoker" \
        2>/dev/null || true

    log_info "✓ 이제 허용된 사용자만 접속할 수 있습니다"
    echo ""
    echo "사용자 추가: ./deploy.sh add-user user@gmail.com"
}

# ─────────────────────────────────────────────────────────────────
# 로그 확인
# ─────────────────────────────────────────────────────────────────
logs() {
    log_info "Backend 로그 (최근 50줄):"
    gcloud run services logs read $BACKEND_SERVICE --region=$REGION --limit=50
}

# ─────────────────────────────────────────────────────────────────
# 상태 확인
# ─────────────────────────────────────────────────────────────────
status() {
    log_info "=== 서비스 상태 ==="

    echo ""
    echo "Backend:"
    gcloud run services describe $BACKEND_SERVICE --region=$REGION \
        --format='table(status.url, status.conditions[0].status, spec.template.spec.containers[0].resources)'

    echo ""
    echo "Frontend:"
    gcloud run services describe $FRONTEND_SERVICE --region=$REGION \
        --format='table(status.url, status.conditions[0].status)'
}

# ─────────────────────────────────────────────────────────────────
# 삭제
# ─────────────────────────────────────────────────────────────────
destroy() {
    log_warn "모든 리소스를 삭제합니다. 계속하시겠습니까? (y/N)"
    read -r confirm
    if [[ $confirm != "y" && $confirm != "Y" ]]; then
        log_info "취소됨"
        exit 0
    fi

    gcloud run services delete $BACKEND_SERVICE --region=$REGION --quiet || true
    gcloud run services delete $FRONTEND_SERVICE --region=$REGION --quiet || true

    log_info "서비스 삭제 완료"
    log_warn "버킷($BUCKET_NAME)과 시크릿은 수동으로 삭제하세요"
}

# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
case "${1:-help}" in
    setup)
        check_prerequisites
        setup
        ;;
    upload-data)
        upload_data
        ;;
    build)
        check_prerequisites
        build
        ;;
    deploy)
        check_prerequisites
        deploy
        ;;
    deploy-private)
        check_prerequisites
        deploy_private
        ;;
    all)
        check_prerequisites
        build
        deploy
        ;;
    all-private)
        check_prerequisites
        build
        deploy_private
        ;;
    add-user)
        check_prerequisites
        add_user "$2"
        ;;
    remove-user)
        check_prerequisites
        remove_user "$2"
        ;;
    list-users)
        check_prerequisites
        list_users
        ;;
    make-public)
        check_prerequisites
        make_public
        ;;
    make-private)
        check_prerequisites
        make_private
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    destroy)
        destroy
        ;;
    *)
        echo "BioInsight AI - GCP 배포 스크립트"
        echo ""
        echo "사용법: $0 <command>"
        echo ""
        echo "배포 Commands:"
        echo "  setup          최초 1회 GCP 프로젝트 설정"
        echo "  upload-data    모델/데이터를 Cloud Storage에 업로드"
        echo "  build          Docker 이미지 빌드 & 푸시"
        echo "  deploy         Cloud Run에 배포 (🌐 공개)"
        echo "  deploy-private Cloud Run에 배포 (🔒 비공개)"
        echo "  all            build + deploy (공개)"
        echo "  all-private    build + deploy (비공개)"
        echo ""
        echo "접근 제어 Commands:"
        echo "  add-user <email>     사용자 접근 권한 추가"
        echo "  remove-user <email>  사용자 접근 권한 제거"
        echo "  list-users           허용된 사용자 목록"
        echo "  make-public          공개 모드로 전환"
        echo "  make-private         비공개 모드로 전환"
        echo ""
        echo "관리 Commands:"
        echo "  logs           Backend 로그 확인"
        echo "  status         서비스 상태 확인"
        echo "  destroy        모든 리소스 삭제"
        echo ""
        echo "예시:"
        echo "  ./deploy.sh all-private              # 비공개로 배포"
        echo "  ./deploy.sh add-user me@gmail.com    # 사용자 추가"
        echo "  ./deploy.sh add-user friend@gmail.com"
        ;;
esac
