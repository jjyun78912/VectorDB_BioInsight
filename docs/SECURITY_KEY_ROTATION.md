# 🔐 API 키 재발급 안내 (Security Key Rotation Guide)

## ⚠️ 긴급 조치 필요

기존 `.env` 파일에 포함된 API 키가 노출되었을 수 있습니다. **즉시 모든 API 키를 재발급**해주세요.

---

## 📋 재발급이 필요한 API 키 목록

### 1. OpenAI API Key (필수)
- **재발급 위치**: https://platform.openai.com/api-keys
- **조치 사항**:
  1. 기존 키 삭제 (Revoke)
  2. 새 키 생성
  3. `.env` 파일의 `OPENAI_API_KEY` 업데이트

### 2. Anthropic API Key (필수)
- **재발급 위치**: https://console.anthropic.com/account/keys
- **조치 사항**:
  1. 기존 키 삭제
  2. 새 키 생성
  3. `.env` 파일의 `ANTHROPIC_API_KEY` 업데이트

### 3. Google Gemini API Key
- **재발급 위치**: https://aistudio.google.com/app/apikey
- **조치 사항**:
  1. API 키 목록에서 기존 키 삭제
  2. 새 키 생성
  3. `.env` 파일의 `GOOGLE_API_KEY` 업데이트

### 4. NCBI API Key
- **재발급 위치**: https://www.ncbi.nlm.nih.gov/account/settings/
- **조치 사항**:
  1. Settings > API Key Management
  2. 새 키 생성 (기존 키 자동 무효화)
  3. `.env` 파일의 `NCBI_API_KEY` 업데이트

### 5. Perplexity API Key (선택)
- **재발급 위치**: https://www.perplexity.ai/settings/api
- **조치 사항**: 새 키 생성 후 `PERPLEXITY_API_KEY` 업데이트

### 6. Groq API Key (선택)
- **재발급 위치**: https://console.groq.com/keys
- **조치 사항**: 새 키 생성 후 `GROQ_API_KEY` 업데이트

---

## 🛡️ 보안 모범 사례

### .env 파일 관리
```bash
# .env 파일은 절대 git에 커밋하지 마세요
# .gitignore에 이미 포함되어 있습니다

# 설정 방법
cp .env.example .env
# 텍스트 에디터로 .env 파일을 열고 실제 키 입력
```

### 키 백업 (안전한 방법)
- **추천**: 1Password, Bitwarden 등 패스워드 매니저 사용
- **금지**: 이메일, 메신저, 클라우드 노트에 저장

### 정기적인 키 교체
- 3-6개월마다 API 키 교체 권장
- 의심스러운 활동 감지 시 즉시 교체

---

## 📊 비용 모니터링

키 노출 시 무단 사용으로 인한 비용이 발생할 수 있습니다.
각 서비스의 사용량과 비용을 확인하세요:

| 서비스 | 대시보드 URL |
|--------|-------------|
| OpenAI | https://platform.openai.com/usage |
| Anthropic | https://console.anthropic.com/settings/billing |
| Google AI | https://aistudio.google.com/app/prompts |
| Groq | https://console.groq.com/usage |

---

## ✅ 완료 체크리스트

- [ ] OpenAI API 키 재발급 완료
- [ ] Anthropic API 키 재발급 완료
- [ ] Google Gemini API 키 재발급 완료
- [ ] NCBI API 키 재발급 완료
- [ ] Perplexity API 키 재발급 완료 (사용 시)
- [ ] Groq API 키 재발급 완료 (사용 시)
- [ ] `.env` 파일 업데이트 완료
- [ ] 애플리케이션 정상 작동 확인
- [ ] 각 서비스 비용 대시보드 확인

---

## 🔧 문제 해결

### 키 업데이트 후 앱이 작동하지 않을 때
```bash
# 백엔드 재시작
cd backend && uvicorn app.main:app --reload

# 또는 Docker 사용 시
docker-compose down && docker-compose up -d
```

### 환경변수가 적용되지 않을 때
```bash
# 환경변수 확인
cat .env | grep -v "^#" | grep -v "^$"

# Python에서 확인
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT SET')[:10])"
```

---

*Last Updated: 2026-01-26*
