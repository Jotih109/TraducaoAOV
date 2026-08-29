# -*- coding: utf-8 -*-
import requests
import urllib.parse
import time
from typing import Optional, Dict

class TranslationEngine:
    """
    Motor de tradução robusto com fallback multi-serviço e cache local.
    Provedores:
    1. Google Translate Direct JSON API
    2. Deep-Translator Google
    3. MyMemory Translated API
    4. Lingva Open Translate
    """
    def __init__(self):
        self.cache: Dict[str, str] = {}
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def translate(self, text: str, source: str = "pt", target: str = "en") -> str:
        text = text.strip()
        if not text:
            return ""

        cache_key = f"{source}:{target}:{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 1. Tentar Google Translate Direct
        result = self._translate_google_direct(text, source, target)
        if result:
            self.cache[cache_key] = result
            return result

        # 2. Tentar Deep-Translator
        result = self._translate_deep_translator(text, source, target)
        if result:
            self.cache[cache_key] = result
            return result

        # 3. Tentar MyMemory
        result = self._translate_mymemory(text, source, target)
        if result:
            self.cache[cache_key] = result
            return result

        # 4. Tentar Lingva
        result = self._translate_lingva(text, source, target)
        if result:
            self.cache[cache_key] = result
            return result

        return text

    def _translate_google_direct(self, text: str, source: str, target: str) -> Optional[str]:
        try:
            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source}&tl={target}&dt=t&q={encoded_text}"
            resp = self.session.get(url, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                translated = "".join([part[0] for part in data[0] if part and part[0]])
                if translated:
                    return translated.strip()
        except Exception:
            pass
        return None

    def _translate_deep_translator(self, text: str, source: str, target: str) -> Optional[str]:
        try:
            from deep_translator import GoogleTranslator
            res = GoogleTranslator(source=source, target=target).translate(text)
            if res:
                return res.strip()
        except Exception:
            pass
        return None

    def _translate_mymemory(self, text: str, source: str, target: str) -> Optional[str]:
        try:
            encoded_text = urllib.parse.quote(text)
            url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair={source}|{target}"
            resp = self.session.get(url, timeout=4)
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
            encoded_text = urllib.parse.quote(text)
            url = f"https://lingva.ml/api/v1/{source}/{target}/{encoded_text}"
            resp = self.session.get(url, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if "translation" in data and data["translation"]:
                    return data["translation"].strip()
        except Exception:
            pass
        return None

# Instância global compartilhada
translator_engine = TranslationEngine()
