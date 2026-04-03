from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .backend import TranslationBackend

_DEFAULT_LANGS = ('hindi', 'tamil', 'telugu', 'malayalam', 'marathi')


def _default_lexicon() -> Dict[str, Dict[str, str]]:
    """Small offline gloss → per-language string map (extend via YAML)."""
    return {
        'HELLO': {
            'hindi': 'नमस्ते',
            'tamil': 'வணக்கம்',
            'telugu': 'నమస్కారం',
            'malayalam': 'നമസ്കാരം',
            'marathi': 'नमस्कार',
        },
        'THANK': {
            'hindi': 'धन्यवाद',
            'tamil': 'நன்றி',
            'telugu': 'ధన్యవాదాలు',
            'malayalam': 'നന്ദി',
            'marathi': 'धन्यवाद',
        },
        'YOU': {
            'hindi': 'आप',
            'tamil': 'நீங்கள்',
            'telugu': 'మీరు',
            'malayalam': 'നിങ്ങൾ',
            'marathi': 'तुम्ही',
        },
        'PLEASE': {
            'hindi': 'कृपया',
            'tamil': 'தயவுசெய்து',
            'telugu': 'దయచేసి',
            'malayalam': 'ദയവായി',
            'marathi': 'कृपया',
        },
        'YES': {
            'hindi': 'हाँ',
            'tamil': 'ஆம்',
            'telugu': 'అవును',
            'malayalam': 'അതെ',
            'marathi': 'हो',
        },
        'NO': {
            'hindi': 'नहीं',
            'tamil': 'இல்லை',
            'telugu': 'కాదు',
            'malayalam': 'അല്ല',
            'marathi': 'नाही',
        },
        'HELP': {
            'hindi': 'मदद',
            'tamil': 'உதவி',
            'telugu': 'సహాయం',
            'malayalam': 'സഹായം',
            'marathi': 'मदत',
        },
        'WATER': {
            'hindi': 'पानी',
            'tamil': 'தண்ணீர்',
            'telugu': 'నీరు',
            'malayalam': 'വെള്ളം',
            'marathi': 'पाणी',
        },
    }


class RuleBasedBackend(TranslationBackend):
    """
    Template + lexicon gloss→text conversion. No external services.
    Reads optional ``templates`` and ``lexicon`` from language_prompts YAML.
    """

    def __init__(
        self,
        prompts_path: str = 'config/language_prompts.yaml',
        language_prompts: Optional[Dict[str, Any]] = None,
    ):
        self.prompts_path = Path(prompts_path)
        self._language_prompts = language_prompts or {}
        self.templates = self._load_templates()
        self.lexicon: Dict[str, Dict[str, str]] = _default_lexicon()
        self._merge_yaml_lexicon()

    def _load_templates(self) -> Dict[str, str]:
        data = self._load_yaml_file()
        raw = data.get('templates', {})
        out: dict[str, str] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, str):
                    out[str(k)] = v
        # Defaults for known Indian languages from app
        for lang in _DEFAULT_LANGS:
            out.setdefault(lang, '{gloss}')
        return out

    def _load_yaml_file(self) -> Dict[str, Any]:
        if not self.prompts_path.exists():
            return {}
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}

    def _merge_yaml_lexicon(self) -> None:
        data = self._load_yaml_file()
        extra = data.get('lexicon', {})
        if not isinstance(extra, dict):
            return
        for gloss_key, per_lang in extra.items():
            if not isinstance(per_lang, dict):
                continue
            k = str(gloss_key).strip().upper().replace(' ', '-')
            self.lexicon.setdefault(k, {})
            for lang, text in per_lang.items():
                if isinstance(text, str):
                    self.lexicon[k][str(lang).lower()] = text

    def translate(
        self,
        gloss: str,
        target_lang: str,
        context: Optional[dict] = None,
    ) -> Tuple[str, str]:
        if not gloss.strip():
            return '', ''

        target = target_lang.lower().strip()
        tokens = gloss.strip().upper().split()
        translated_tokens: list[str] = []

        for tok in tokens:
            clean = re.sub(r'[^A-Z0-9\-]', '', tok)
            if not clean:
                continue
            per = self.lexicon.get(clean)
            if per is not None and target in per:
                translated_tokens.append(per[target])
            else:
                translated_tokens.append(clean.replace('-', ' ').lower().capitalize())

        gloss_text = ' '.join(translated_tokens)
        template = self.templates.get(target, '{gloss}')
        try:
            native = template.format(
                gloss=gloss_text,
                sentence_case=gloss_text[:1].upper() + gloss_text[1:]
                if gloss_text
                else '',
            )
        except (KeyError, ValueError):
            native = gloss_text

        return native, ''

    def is_available(self) -> bool:
        if self.prompts_path.exists():
            return True
        return bool(self._language_prompts)

    @property
    def backend_name(self) -> str:
        return 'rule_based'
