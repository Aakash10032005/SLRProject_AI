import requests
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class OllamaClient:
    """HTTP client for Ollama local LLM inference."""

    def __init__(self, base_url: str = 'http://localhost:11434',
                 model: str = 'llama3.2:3b-instruct-q4_K_M',
                 timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

    def translate(self, asl_gloss: str, system_prompt: str) -> Tuple[str, str]:
        """
        Translate ASL gloss to target language via Ollama.
        Returns (native_text, roman_transliteration).
        Line 1 of response = native script, Line 2 = roman.
        """
        payload = {
            "model": self.model,
            "prompt": f"Translate this ASL gloss to the target language:\n{asl_gloss}",
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 100}
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get('response', '').strip()

            lines = [l.strip() for l in response_text.split('\n') if l.strip()]
            native = lines[0] if len(lines) > 0 else asl_gloss
            roman = lines[1] if len(lines) > 1 else ''
            return native, roman

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return ('Translation unavailable', '')
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is 'ollama serve' running?")
            return ('Translation unavailable', '')
        except Exception as e:
            logger.error(f"Ollama translation error: {e}")
            return ('Translation unavailable', '')

    def check_connection(self) -> bool:
        """Ping Ollama server. Returns True if available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
