"""
Translation Service using Claude API.

Features:
- Korean <-> English translation
- Language detection
- Batch translation for search results
"""
import re
from typing import Optional
import anthropic

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL


class TranslationService:
    """Translate text between Korean and English using Claude."""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL or "claude-sonnet-4-20250514"

    def _call_claude(self, system_prompt: str, user_text: str) -> str:
        """Call Claude API with system and user prompts."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_text}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Claude API error: {e}")
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
            result = self._call_claude(system, text)
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
            return self._call_claude(system, text)
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
            return self._call_claude(system, text)
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
