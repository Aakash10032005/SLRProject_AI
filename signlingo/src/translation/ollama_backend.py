from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from .backend import TranslationBackend
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaBackend(TranslationBackend):
    """Optional local LLM via Ollama HTTP API (same protocol as before)."""

    def __init__(
        self,
        base_url: str = 'http://localhost:11434',
        model: str = 'llama3.2:3b-instruct-q4_K_M',
        timeout: int = 10,
        language_prompts: Optional[Dict[str, Any]] = None,
        availability_ttl_s: float = 15.0,
    ):
        self.language_prompts = language_prompts or {}
        self._client = OllamaClient(base_url=base_url, model=model, timeout=timeout)
        self._ttl = max(1.0, float(availability_ttl_s))
        self._cached_available: Optional[bool] = None
        self._last_check_mono: float = 0.0

    def _probe_availability(self) -> bool:
        ok = self._client.check_connection()
        self._cached_available = ok
        self._last_check_mono = time.monotonic()
        return ok

    def is_available(self) -> bool:
        now = time.monotonic()
        if (
            self._cached_available is None
            or (now - self._last_check_mono) >= self._ttl
        ):
            return self._probe_availability()
        return bool(self._cached_available)

    def refresh_availability(self) -> bool:
        """Force a new health check (e.g. after starting ``ollama serve``)."""
        self._cached_available = None
        self._last_check_mono = 0.0
        return self._probe_availability()

    def translate(
        self,
        gloss: str,
        target_lang: str,
        context: Optional[dict] = None,
    ) -> Tuple[str, str]:
        if not self.is_available():
            self.refresh_availability()
            if not self.is_available():
                return gloss, ''

        lang_config = self.language_prompts.get(target_lang, {})
        system_prompt = lang_config.get('system_prompt', '')
        if not system_prompt:
            logger.warning('No system prompt for %s; using generic instruction', target_lang)
            system_prompt = f'Translate the following ASL gloss to {target_lang}.'

        example_input = lang_config.get('example_input', '')
        example_native = lang_config.get('example_native', '')
        example_roman = lang_config.get('example_roman', '')
        if example_input and example_native:
            system_prompt += (
                f'\n\nExample:\nInput: {example_input}\n'
                f'Output line 1: {example_native}\n'
                f'Output line 2: {example_roman}'
            )

        native, roman = self._client.translate(gloss, system_prompt)
        if native == 'Translation unavailable':
            self._cached_available = False
            self._last_check_mono = time.monotonic()
            return gloss, ''
        return native, roman

    @property
    def backend_name(self) -> str:
        return 'ollama'
