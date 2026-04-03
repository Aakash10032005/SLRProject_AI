from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from .backend import TranslationBackend
from .rule_backend import RuleBasedBackend

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    native_text: str
    roman_text: str
    language: str
    confidence: float


def create_translation_backend(
    config: Dict[str, Any],
    language_prompts: Dict[str, Any],
) -> TranslationBackend:
    """
    Factory: build the configured translation backend.
    ``translation.backend``: ``rule_based`` | ``ollama``
    """
    trans = config.get('translation', {})
    backend_type = (trans.get('backend') or 'rule_based').strip().lower()
    prompts_path = trans.get('prompts_path', 'config/language_prompts.yaml')

    if backend_type == 'rule_based':
        return RuleBasedBackend(prompts_path=prompts_path, language_prompts=language_prompts)

    if backend_type == 'ollama':
        try:
            from .ollama_backend import OllamaBackend
        except ImportError:
            logger.warning('Ollama backend import failed; using rule_based')
            return RuleBasedBackend(prompts_path=prompts_path, language_prompts=language_prompts)

        o_cfg = trans.get('ollama', {})
        base_url = (
            o_cfg.get('base_url')
            or trans.get('ollama_base_url')
            or 'http://localhost:11434'
        )
        model = (
            o_cfg.get('model')
            or trans.get('ollama_model')
            or 'llama3.2:3b-instruct-q4_K_M'
        )
        timeout = int(o_cfg.get('timeout', trans.get('ollama_timeout', 10)))
        ttl = float(o_cfg.get('availability_ttl_s', 15.0))
        return OllamaBackend(
            base_url=base_url,
            model=model,
            timeout=timeout,
            language_prompts=language_prompts,
            availability_ttl_s=ttl,
        )

    raise ValueError(f'Unknown translation backend: {backend_type}')


class CGME:
    """
    Contextual Grammatical Morphing Engine.
    Delegates gloss→language generation to a pluggable TranslationBackend.
    """

    def __init__(self, config: Dict[str, Any], language_prompts: Dict[str, Any]):
        self.language_prompts = language_prompts
        self.backend = create_translation_backend(config, language_prompts)

    def translate(self, asl_gloss: str, target_language: str) -> TranslationResult:
        """Translate ASL gloss to the target language."""
        if not self.backend.is_available():
            logger.warning(
                "Translation backend '%s' unavailable; using raw gloss",
                self.backend.backend_name,
            )
            return TranslationResult(
                native_text=asl_gloss,
                roman_text='',
                language=target_language,
                confidence=0.0,
            )

        try:
            native, roman = self.backend.translate(asl_gloss, target_language, context=None)
        except Exception as e:
            logger.error('Translation error (%s): %s', self.backend.backend_name, e)
            return TranslationResult(
                native_text=asl_gloss,
                roman_text='',
                language=target_language,
                confidence=0.0,
            )

        if not native.strip():
            native = asl_gloss

        return TranslationResult(
            native_text=native.strip(),
            roman_text=roman.strip(),
            language=target_language,
            confidence=1.0,
        )
