import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)


def test_ollama_connection():
    from src.translation.ollama_client import OllamaClient
    client = OllamaClient()
    result = client.check_connection()
    logging.info(f"Ollama connection: {'OK' if result else 'Not available (expected if Ollama not running)'}")
    assert isinstance(result, bool)
    print(f"OllamaClient.check_connection() -> {result} PASSED")


def test_sentence_composer():
    from src.translation.sentence_composer import SentenceComposer
    composer = SentenceComposer()

    composer.add_sign('hello')
    composer.add_sign('world')
    composer.add_sign('today')

    gloss = composer.get_gloss()
    assert gloss == 'HELLO WORLD TODAY', f"Got: {gloss}"
    assert composer.is_sentence_ready(pause_detected=True)
    assert not composer.is_sentence_ready(pause_detected=False)

    composer.clear()
    assert composer.get_gloss() == ''
    assert not composer.is_sentence_ready(pause_detected=True)
    print("SentenceComposer tests PASSED")


def test_tts_engine_init():
    from src.output.tts_engine import TTSEngine
    tts = TTSEngine()
    # Just verify it initializes without crashing
    assert tts is not None
    print("TTSEngine init PASSED")


if __name__ == '__main__':
    test_ollama_connection()
    test_sentence_composer()
    test_tts_engine_init()
    print("\nAll translation tests passed.")
