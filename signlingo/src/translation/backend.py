from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class TranslationBackend(ABC):
    """Abstract interface for gloss → target-language text generation."""

    @abstractmethod
    def translate(
        self,
        gloss: str,
        target_lang: str,
        context: Optional[dict] = None,
    ) -> Tuple[str, str]:
        """
        Convert an ASL gloss string to target language.
        Returns (native_script, roman_transliteration). Roman may be empty.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can run (dependencies reachable, config valid)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier."""
        raise NotImplementedError
