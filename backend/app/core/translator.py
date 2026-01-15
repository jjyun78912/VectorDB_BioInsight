"""
Translation Service with OpenAI/Claude fallback.

Features:
- Korean <-> English translation
- Language detection
- Batch translation for search results

Priority:
1. OpenAI (OPENAI_API_KEY 설정 시)
2. Claude/Anthropic (ANTHROPIC_API_KEY 설정 시)
"""
import os
import re
from typing import Optional


class TranslationService:
    """Translate text between Korean and English using OpenAI or Claude."""

    def __init__(self):
        self.client = None
        self.provider = None
        self.model = None

        # 1. OpenAI 우선 시도
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_key)
                self.provider = "openai"
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                print(f"TranslationService: OpenAI 사용 ({self.model})")
                return
            except ImportError:
                print("TranslationService: openai 패키지 미설치")
            except Exception as e:
                print(f"TranslationService: OpenAI 초기화 실패: {e}")

        # 2. Claude/Anthropic fallback
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=anthropic_key)
                self.provider = "anthropic"
                self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
                print(f"TranslationService: Anthropic 사용 ({self.model})")
                return
            except ImportError:
                print("TranslationService: anthropic 패키지 미설치")
            except Exception as e:
                print(f"TranslationService: Anthropic 초기화 실패: {e}")

        raise ValueError("OPENAI_API_KEY 또는 ANTHROPIC_API_KEY가 필요합니다")

    def _call_llm(self, system_prompt: str, user_text: str) -> str:
        """Call LLM API with system and user prompts."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}
                    ],
                    max_tokens=4096,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_text}
                    ]
                )
                return response.content[0].text.strip()

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            print(f"LLM API error ({self.provider}): {e}")
            raise

    def detect_language(self, text: str) -> str:
        """
        Detect if text is Korean or English.

        Returns: 'ko', 'en', or 'other'
        """
        # Quick heuristic check first
        korean_chars = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]', text))
        total_chars = len(re.findall(r'\w', text))

        if total_chars == 0:
            return 'other'

        korean_ratio = korean_chars / total_chars

        if korean_ratio > 0.3:
            return 'ko'
        elif korean_ratio < 0.05:
            return 'en'

        # Use LLM for ambiguous cases
        try:
            system = "Detect the language of the given text. Reply with only 'ko' for Korean, 'en' for English, or 'other' for other languages. No explanation."
            result = self._call_llm(system, text)
            lang = result.lower()
            if lang in ['ko', 'en', 'other']:
                return lang
            return 'en'  # Default to English
        except Exception:
            return 'en'  # Default to English on any error

    def translate_to_english(self, text: str) -> str:
        """Translate Korean text to English."""
        try:
            system = """You are a professional medical/scientific translator.
Translate the Korean text to English.
- Keep scientific terms accurate
- Preserve medical terminology
- Output ONLY the translation, nothing else"""
            return self._call_llm(system, text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_to_korean(self, text: str) -> str:
        """Translate English text to Korean."""
        try:
            system = """You are a professional medical/scientific translator.
Translate the English text to Korean.
- Keep scientific terms accurate (can keep English terms in parentheses if commonly used)
- Preserve medical terminology
- Output ONLY the translation, nothing else"""
            return self._call_llm(system, text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_search_query(self, query: str) -> tuple[str, bool]:
        """
        Translate search query if Korean.

        Returns: (translated_query, was_translated)
        """
        lang = self.detect_language(query)

        if lang == 'ko':
            translated = self.translate_to_english(query)
            return translated, True

        return query, False

    def translate_paper_result(self, paper: dict, to_korean: bool = True) -> dict:
        """
        Translate paper metadata to Korean.

        Args:
            paper: Paper dict with title, abstract, etc.
            to_korean: If True, translate to Korean

        Returns:
            Paper dict with translated fields
        """
        if not to_korean:
            return paper

        translated = paper.copy()

        # Translate title
        if paper.get('title'):
            translated['title_ko'] = self.translate_to_korean(paper['title'])

        # Translate abstract (if present and not too long)
        if paper.get('abstract') and len(paper['abstract']) < 5000:
            translated['abstract_ko'] = self.translate_to_korean(paper['abstract'])

        return translated


# Singleton instance
_translator: Optional[TranslationService] = None


def get_translator() -> TranslationService:
    """Get or create translator instance."""
    global _translator
    if _translator is None:
        _translator = TranslationService()
    return _translator
