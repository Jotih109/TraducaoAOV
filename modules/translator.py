# -*- coding: utf-8 -*-
import requests
import time
import re
from typing import Optional, Dict, Tuple
from collections import OrderedDict

try:
    from deep_translator import GoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    HAS_DEEP_TRANSLATOR = False


def format_subtitle_text(text: str) -> str:
    """
    Limpa e formata o texto para estilo profissional de legenda:
    - Remove espaços duplicados
    - Capitaliza o início da primeira palavra
    - Preserva pontuação relevante
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # Capitalizar primeira letra se for minúscula
    if text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class TranslationEngine:
    """
    Motor de tradução robusto de alta performance com fallback multi-serviço,
    normalização de parâmetros, medição de latência e cache LRU inteligente.
    Provedores:
    1. Google Translate Direct JSON API (ultra-rápido, sem limites de caracteres)
    2. Deep-Translator Google
    3. MyMemory Translated API
    4. Lingva Open Translate
    """
    def __init__(self, max_cache_size: int = 3000):
        self.max_cache_size = max_cache_size
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.deep_translators: Dict[Tuple[str, str], object] = {}
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })

    def translate(self, text: str, source: str = "pt", target: str = "en") -> str:
        translated, _ = self.translate_with_metrics(text, source, target)
        return translated

    def translate_with_metrics(self, text: str, source: str = "pt", target: str = "en") -> Tuple[str, float]:
        """Retorna (texto_traduzido, tempo_gasto_em_segundos)."""
        text = text.strip()
        if not text:
            return "", 0.0

        if source == target:
            return format_subtitle_text(text), 0.0

        start_time = time.time()
        cache_key = f"{source}:{target}:{text.lower()}"
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            latency = time.time() - start_time
            return self.cache[cache_key], latency

        # 1. Google Translate Direct
        result = self._translate_google_direct(text, source, target)
        
        # 2. Deep-Translator (se o primeiro falhar)
        if not result and HAS_DEEP_TRANSLATOR:
            result = self._translate_deep_translator(text, source, target)

        # 3. MyMemory (fallback alternativo)
        if not result:
            result = self._translate_mymemory(text, source, target)

        # 4. Lingva (fallback final)
        if not result:
            result = self._translate_lingva(text, source, target)

        if not result:
            result = text

        formatted = format_subtitle_text(result)
        self._add_to_cache(cache_key, formatted)
        latency = time.time() - start_time
        return formatted, latency

    def _add_to_cache(self, key: str, val: str):
        self.cache[key] = val
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)

    def _translate_google_direct(self, text: str, source: str, target: str) -> Optional[str]:
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": source,
                "tl": target,
                "dt": "t",
                "q": text
            }
            resp = self.session.get(url, params=params, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0 and data[0]:
                    translated = "".join([part[0] for part in data[0] if part and part[0]])
                    if translated:
                        return translated.strip()
        except Exception:
            pass
        return None

    def _translate_deep_translator(self, text: str, source: str, target: str) -> Optional[str]:
        try:
            pair = (source, target)
            if pair not in self.deep_translators:
                self.deep_translators[pair] = GoogleTranslator(source=source, target=target)
            translator = self.deep_translators[pair]
            res = translator.translate(text)
            if res:
                return res.strip()
        except Exception:
            pass
        return None

    def _translate_mymemory(self, text: str, source: str, target: str) -> Optional[str]:
        try:
            url = "https://api.mymemory.translated.net/get"
            params = {
                "q": text,
                "langpair": f"{source}|{target}"
            }
            resp = self.session.get(url, params=params, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                if "responseData" in data and "translatedText" in data["responseData"]:
                    translated = data["responseData"]["translatedText"]
                    if translated and not translated.startswith("MYMEMORY WARNING"):
                        return translated.strip()
        except Exception:
            pass
        return None

    def _translate_lingva(self, text: str, source: str, target: str) -> Optional[str]:
        try:
            import urllib.parse
            encoded_text = urllib.parse.quote(text)
            url = f"https://lingva.ml/api/v1/{source}/{target}/{encoded_text}"
            resp = self.session.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                if "translation" in data and data["translation"]:
                    return data["translation"].strip()
        except Exception:
            pass
        return None

# Instância global compartilhada
translator_engine = TranslationEngine()
