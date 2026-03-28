"""
Basic Roman transliteration helpers.
The primary transliteration is handled by Ollama (line 2 of response).
This module provides fallback utilities.
"""


LANGUAGE_SCRIPTS = {
    'hindi': 'Devanagari',
    'marathi': 'Devanagari',
    'tamil': 'Tamil',
    'telugu': 'Telugu',
    'malayalam': 'Malayalam',
}


def get_script_name(language: str) -> str:
    return LANGUAGE_SCRIPTS.get(language.lower(), 'Unknown')


def is_native_script(text: str) -> bool:
    """Check if text contains non-ASCII characters (i.e., native script)."""
    return any(ord(c) > 127 for c in text)
