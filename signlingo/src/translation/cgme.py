import logging
from dataclasses import dataclass
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    native_text: str
    roman_text: str
    language: str
    confidence: float


class CGME:
    """
    Contextual Grammatical Morphing Engine.
    Wraps OllamaClient with per-language system prompts for
    grammatically correct Indian language translation.
    """

    def __init__(self, ollama_client: OllamaClient, language_prompts: dict):
        self.client = ollama_client
        self.language_prompts = language_prompts

    def translate(self, asl_gloss: str, target_language: str) -> TranslationResult:
        """Translate ASL gloss to target Indian language."""
        lang_config = self.language_prompts.get(target_language, {})
        system_prompt = lang_config.get('system_prompt', '')

        if not system_prompt:
            logger.warning(f"No system prompt found for language: {target_language}")
            system_prompt = f"Translate the following ASL gloss to {target_language}."

        # Append few-shot example to system prompt
        example_input = lang_config.get('example_input', '')
        example_native = lang_config.get('example_native', '')
        example_roman = lang_config.get('example_roman', '')
        if example_input and example_native:
            system_prompt += (
                f"\n\nExample:\nInput: {example_input}\n"
                f"Output line 1: {example_native}\n"
                f"Output line 2: {example_roman}"
            )

        native, roman = self.client.translate(asl_gloss, system_prompt)

        # Confidence: 1.0 if translation succeeded, 0.0 if unavailable
        confidence = 0.0 if native == 'Translation unavailable' else 1.0

        return TranslationResult(
            native_text=native,
            roman_text=roman,
            language=target_language,
            confidence=confidence
        )
