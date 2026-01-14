"""
LLM 초기화 헬퍼 - Vertex AI 또는 Google AI Studio 자동 선택.
"""
import os
import logging

logger = logging.getLogger(__name__)

# 기본 모델
DEFAULT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash-001")


def get_langchain_llm(model: str = None, temperature: float = 0.3):
    """
    LangChain LLM 인스턴스를 반환합니다.

    Vertex AI 설정이 있으면 ChatVertexAI를 사용하고,
    없으면 ChatGoogleGenerativeAI(API 키)를 사용합니다.

    Args:
        model: 모델 이름 (기본: gemini-2.0-flash-001)
        temperature: 생성 온도 (기본: 0.3)

    Returns:
        LangChain ChatModel 인스턴스
    """
    model = model or DEFAULT_MODEL

    # Vertex AI 사용 여부 확인
    use_vertex = os.getenv("PAPER_EXPLAINER_USE_VERTEX", "false").lower() == "true"
    credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("VERTEX_CREDENTIALS_FILE")

    if use_vertex and credentials_file and os.path.exists(credentials_file):
        try:
            from langchain_google_vertexai import ChatVertexAI

            project_id = os.getenv("VERTEX_PROJECT_ID", "gen-lang-client-0509379035")
            location = os.getenv("VERTEX_LOCATION", "us-central1")

            llm = ChatVertexAI(
                model=model,
                project=project_id,
                location=location,
                temperature=temperature,
            )
            logger.info(f"Vertex AI LLM 초기화 완료: {model}")
            return llm

        except Exception as e:
            logger.warning(f"Vertex AI 초기화 실패, API 키로 fallback: {e}")

    # Google AI Studio (API 키) 사용
    from backend.app.core.config import GOOGLE_API_KEY

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다")

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
    logger.info(f"Google AI Studio LLM 초기화 완료: {model}")
    return llm
