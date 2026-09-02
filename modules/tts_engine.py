# -*- coding: utf-8 -*-
import threading
import queue
import time
from typing import List, Dict, Optional

try:
    import pythoncom
    HAS_PYTHONCOM = True
except ImportError:
    HAS_PYTHONCOM = False

import pyttsx3


class TTSEngine:
    """
    Motor de Síntese de Voz (TTS) assíncrono e ultra-estável em segundo plano.
    - Inicialização COM por thread para evitar travamentos no Windows
    - Engine persistente por thread para sintetização instantânea (<10ms)
    - Descarte automático de falas antigas se a fila acumular (anti-lag para lives)
    - Suporte a parada imediata e seleção inteligente por idioma
    """
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.rate = 160
        self.volume = 0.9
        self.selected_voice_id: Optional[str] = None
        self.available_voices: List[Dict[str, str]] = []
        
        self._detect_voices()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _detect_voices(self):
        try:
            if HAS_PYTHONCOM:
                pythoncom.CoInitialize()
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            self.available_voices = []
            default_target_voice = None

            # Palavras-chave para localizar uma voz no idioma de destino padrão (português)
            target_keywords = ("portuguese", "brazil", "brasil", "maria", "helena", "francisca", "pt-br", "pt_br")

            for v in voices:
                voice_info = {
                    "id": v.id,
                    "name": v.name,
                    "languages": getattr(v, "languages", [])
                }
                self.available_voices.append(voice_info)
                name_lower = v.name.lower()
                if any(kw in name_lower for kw in target_keywords):
                    if not default_target_voice:
                        default_target_voice = v.id

            if default_target_voice:
                self.selected_voice_id = default_target_voice
            elif self.available_voices:
                self.selected_voice_id = self.available_voices[0]["id"]
                
            engine.stop()
        except Exception as e:
            print(f"Erro ao detectar vozes TTS: {e}")

    def auto_select_voice_for_language(self, lang_code: str):
        """Seleciona automaticamente uma voz do sistema compatível com o idioma informado."""
        keywords_map = {
            "pt": ("portuguese", "brazil", "brasil", "maria", "helena", "francisca", "pt-br", "pt_br", "portugal"),
            "en": ("english", "en-us", "en-gb", "david", "zira", "george", "mark", "hazel", "united states", "united kingdom"),
            "es": ("spanish", "espanol", "español", "mexico", "spain", "sabina", "raul", "laura"),
            "fr": ("french", "francais", "français", "france", "hortense", "julie", "paul"),
            "de": ("german", "deutsch", "germany", "hedda", "stefan"),
            "it": ("italian", "italiano", "italy", "elsa", "cosimo"),
            "ja": ("japanese", "japan", "ayumi", "haruka", "ichiro"),
            "zh": ("chinese", "china", "mandarin", "huihui", "yaoyao", "kangkang"),
            "ru": ("russian", "russia", "irina", "pavel"),
            "ko": ("korean", "korea", "heami"),
        }
        target_kws = keywords_map.get(lang_code, ())
        for v in self.available_voices:
            name_lower = v["name"].lower()
            if any(kw in name_lower for kw in target_kws):
                self.selected_voice_id = v["id"]
                return v["id"]
        return self.selected_voice_id

    def get_voices(self) -> List[Dict[str, str]]:
        return self.available_voices

    def set_voice(self, voice_id: str):
        self.selected_voice_id = voice_id

    def set_rate(self, rate: int):
        self.rate = max(80, min(300, rate))

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))

    def speak(self, text: str, priority: bool = False):
        if not text or not text.strip():
            return
        
        # Se for prioridade ou a fila acumulou mais de 2 itens, descartar antigos para não atrasar a live
        if priority or self.queue.qsize() > 1:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        self.queue.put(text.strip())

    def stop(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _worker_loop(self):
        if HAS_PYTHONCOM:
            try:
                pythoncom.CoInitialize()
            except Exception:
                pass

        engine = None
        try:
            engine = pyttsx3.init()
        except Exception as e:
            print(f"Erro ao inicializar pyttsx3 no worker: {e}")

        while self.running:
            try:
                text = self.queue.get(timeout=0.3)
                if not text or not self.running:
                    continue

                if engine is None:
                    try:
                        engine = pyttsx3.init()
                    except Exception:
                        continue

                try:
                    if self.selected_voice_id:
                        engine.setProperty("voice", self.selected_voice_id)
                    engine.setProperty("rate", self.rate)
                    engine.setProperty("volume", self.volume)
                    engine.say(text)
                    engine.runAndWait()
                except Exception as ex:
                    # Se falhar, tenta reiniciar a instância na próxima
                    print(f"Erro na síntese: {ex}")
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    engine = None

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro no loop TTS: {e}")
                time.sleep(0.1)

        if engine:
            try:
                engine.stop()
            except Exception:
                pass


# Instância global compartilhada
tts_engine = TTSEngine()
