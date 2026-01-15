"""
LLM 초기화 헬퍼 - OpenAI, Vertex AI, Google AI Studio 자동 선택.

우선순위:
1. OpenAI (OPENAI_API_KEY 설정 시)
2. Vertex AI (PAPER_EXPLAINER_USE_VERTEX=true 시)
3. Google AI Studio (GOOGLE_API_KEY 설정 시)
"""
import os
import logging

logger = logging.getLogger(__name__)

# 기본 모델
DEFAULT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash-001")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_langchain_llm(model: str = None, temperature: float = 0.3):
    """
    LangChain LLM 인스턴스를 반환합니다.

    우선순위:
    1. OpenAI (OPENAI_API_KEY 설정 시, 가장 안정적)
    2. Vertex AI (PAPER_EXPLAINER_USE_VERTEX=true 시)
    3. Google AI Studio (GOOGLE_API_KEY 설정 시)

    Args:
        model: 모델 이름 (자동 선택)
        temperature: 생성 온도 (기본: 0.3)

    Returns:
        LangChain ChatModel 인스턴스
    """
    # 1. OpenAI 사용 (가장 우선)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI

            openai_model = model if model and model.startswith("gpt") else DEFAULT_OPENAI_MODEL
            llm = ChatOpenAI(
                model=openai_model,
                api_key=openai_api_key,
                temperature=temperature,
            )
            logger.info(f"OpenAI LLM 초기화 완료: {openai_model}")
            return llm
        except ImportError:
            logger.warning("langchain-openai 미설치, 다른 LLM으로 fallback")
        except Exception as e:
            logger.warning(f"OpenAI 초기화 실패: {e}")

    # 2. Vertex AI 사용 여부 확인
    use_vertex = os.getenv("PAPER_EXPLAINER_USE_VERTEX", "false").lower() == "true"
    credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("VERTEX_CREDENTIALS_FILE")

    if use_vertex and credentials_file and os.path.exists(credentials_file):
        try:
            from langchain_google_vertexai import ChatVertexAI

            project_id = os.getenv("VERTEX_PROJECT_ID", "gen-lang-client-0509379035")
            location = os.getenv("VERTEX_LOCATION", "us-central1")
            vertex_model = model or DEFAULT_MODEL

            llm = ChatVertexAI(
                model=vertex_model,
                project=project_id,
                location=location,
                temperature=temperature,
            )
            logger.info(f"Vertex AI LLM 초기화 완료: {vertex_model}")
            return llm

        except Exception as e:
            logger.warning(f"Vertex AI 초기화 실패, API 키로 fallback: {e}")

    # 3. Google AI Studio (API 키) 사용
    from backend.app.core.config import GOOGLE_API_KEY

    if not GOOGLE_API_KEY:
        raise ValueError("LLM API 키가 설정되지 않았습니다. OPENAI_API_KEY 또는 GOOGLE_API_KEY를 설정하세요.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    google_model = model or DEFAULT_MODEL

    llm = ChatGoogleGenerativeAI(
        model=google_model,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
    logger.info(f"Google AI Studio LLM 초기화 완료: {google_model}")
    return llm
