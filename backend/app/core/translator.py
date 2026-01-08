"""
Translation Service using Gemini.

Features:
- Korean <-> English translation
- Language detection
- Batch translation for search results
"""
import re
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from .config import GOOGLE_API_KEY, GEMINI_MODEL


class TranslationService:
    """Translate text between Korean and English using Gemini."""

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set")

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,  # Low temperature for consistent translations
        )

        self._build_prompts()

    def _build_prompts(self):
        """Build translation prompts."""
        self.detect_prompt = ChatPromptTemplate.from_messages([
            ("system", "Detect the language of the given text. Reply with only 'ko' for Korean, 'en' for English, or 'other' for other languages. No explanation."),
            ("human", "{text}")
        ])

        self.ko_to_en_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional medical/scientific translator.
Translate the Korean text to English.
- Keep scientific terms accurate
- Preserve medical terminology
- Output ONLY the translation, nothing else"""),
            ("human", "{text}")
        ])

        self.en_to_ko_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional medical/scientific translator.
Translate the English text to Korean.
- Keep scientific terms accurate (can keep English terms in parentheses if commonly used)
- Preserve medical terminology
- Output ONLY the translation, nothing else"""),
            ("human", "{text}")
        ])

        self.detect_chain = self.detect_prompt | self.llm
        self.ko_to_en_chain = self.ko_to_en_prompt | self.llm
        self.en_to_ko_chain = self.en_to_ko_prompt | self.llm

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
            result = self.detect_chain.invoke({"text": text})
            lang = result.content.strip().lower()
            if lang in ['ko', 'en', 'other']:
                return lang
            return 'en'  # Default to English
        except Exception as e:
            return 'en'  # Default to English on any error

    def translate_to_english(self, text: str) -> str:
        """Translate Korean text to English."""
        try:
            result = self.ko_to_en_chain.invoke({"text": text})
            return result.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_to_korean(self, text: str) -> str:
        """Translate English text to Korean."""
        try:
            result = self.en_to_ko_chain.invoke({"text": text})
            return result.content.strip()
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
        if paper.get('abstract') and len(paper['abstract']) < 3000:
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
