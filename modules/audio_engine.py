# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
from typing import List, Dict, Optional
from PyQt5.QtCore import QObject, pyqtSignal

# Códigos de idioma para o reconhecedor de fala (BCP-47) e metadados de exibição
AVAILABLE_LANGUAGES = {
    "en": {"name": "Inglês", "flag": "🇺🇸", "speech_code": "en-US"},
    "pt": {"name": "Português (BR)", "flag": "🇧🇷", "speech_code": "pt-BR"},
    "es": {"name": "Espanhol", "flag": "🇪🇸", "speech_code": "es-ES"},
    "fr": {"name": "Francês", "flag": "🇫🇷", "speech_code": "fr-FR"},
    "de": {"name": "Alemão", "flag": "🇩🇪", "speech_code": "de-DE"},
    "it": {"name": "Italiano", "flag": "🇮🇹", "speech_code": "it-IT"},
    "ja": {"name": "Japonês", "flag": "🇯🇵", "speech_code": "ja-JP"},
    "zh": {"name": "Chinês", "flag": "🇨🇳", "speech_code": "zh-CN"},
    "ru": {"name": "Russo", "flag": "🇷🇺", "speech_code": "ru-RU"},
    "ko": {"name": "Coreano", "flag": "🇰🇷", "speech_code": "ko-KR"},
}

SPEECH_LANG_CODES = {k: v["speech_code"] for k, v in AVAILABLE_LANGUAGES.items()}
LANG_DISPLAY_NAMES = {k: v["name"] for k, v in AVAILABLE_LANGUAGES.items()}


class AudioEngine(QObject):
    """
    Motor de captura de áudio de alta precisão em tempo real.
    - Captura na taxa nativa do hardware (44.1kHz / 48kHz)
    - Downmix estéreo -> mono de alta fidelidade
    - Remoção de DC offset e Normalização Automática de Ganho (AGC)
    - VAD (Detecção de voz) ultra-rápido com latência reduzida
    - Idioma de origem/destino configurável e invertível em tempo real
    """
    level_changed = pyqtSignal(float, list)       # (RMS 0.0-1.0, waveform sample array)
    speech_started = pyqtSignal()                 # Fala detectada
    speech_ended = pyqtSignal()                   # Pausa na fala detectada
    transcription_ready = pyqtSignal(str)         # Texto original transcrito no idioma de origem
    translation_ready = pyqtSignal(str, str)      # (Texto original, Tradução no idioma de destino)
    status_changed = pyqtSignal(str)              # Status em tempo real
    error_occurred = pyqtSignal(str)              # Alerta de erro

    def __init__(self, settings_manager, translator, tts):
        super().__init__()
        self.settings = settings_manager
        self.translator = translator
        self.tts = tts
        self.recognizer = sr.Recognizer()

        # Idioma de origem (o que está sendo falado na live) e de destino (legenda traduzida)
        self.source_lang = self.settings.get("source_lang", "en")
        self.target_lang = self.settings.get("target_lang", "pt")

        # Parâmetros de áudio
        self.native_samplerate = 44100
        self.native_channels = 1
        self.block_size = 1024
        self.stream = None
        self.is_recording = False
        
        # VAD ultra-rápido e responsivo
        self.sensitivity = self.settings.get("mic_sensitivity", 50)
        self.silence_timeout = 0.45   # segundos de silêncio para disparar tradução imediata
        self.min_speech_duration = 0.35  # segundos mínimos de fala
        
        self.is_speaking = False
        self.speech_frames = []
        self.last_speech_time = 0
        self.speech_start_time = 0
        
        # Fila de reconhecimento assíncrono
        self.recognition_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.worker_thread.start()

    def get_input_devices(self) -> List[Dict]:
        """Retorna lista de microfones disponíveis com informações de taxa nativa."""
        devices = []
        try:
            dev_list = sd.query_devices()
            default_in = sd.default.device[0]
            for idx, dev in enumerate(dev_list):
                if dev["max_input_channels"] > 0:
                    devices.append({
                        "index": idx,
                        "name": dev["name"],
                        "channels": dev["max_input_channels"],
                        "default_samplerate": int(dev["default_samplerate"]),
                        "is_default": (idx == default_in)
                    })
        except Exception as e:
            print(f"Erro ao listar dispositivos: {e}")
        return devices

    def set_sensitivity(self, value: int):
        self.sensitivity = max(5, min(95, value))
        self.settings.set("mic_sensitivity", self.sensitivity)

    def set_languages(self, source_lang: str, target_lang: str):
        """Atualiza dinamicamente os idiomas de origem e destino."""
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.settings.set("source_lang", source_lang)
        self.settings.set("target_lang", target_lang)
        source_info = AVAILABLE_LANGUAGES.get(source_lang, {"name": source_lang})
        target_info = AVAILABLE_LANGUAGES.get(target_lang, {"name": target_lang})
        if self.is_recording:
            self.status_changed.emit(f"🎙️ Microfone ativo - captando em {source_info['name']} ➔ traduzindo para {target_info['name']}")

    def _get_threshold(self) -> float:
        # Sensibilidade ajustada com curva logarítmica para máxima precisão
        normalized = (100 - self.sensitivity) / 100.0
        return 0.003 + (normalized ** 2.2) * 0.08

    def start_listening(self, device_index: Optional[int] = None):
        if self.is_recording:
            return

        if device_index is None:
            device_index = self.settings.get("mic_device_index", None)

        try:
            # Obter taxa nativa e canais do dispositivo selecionado
            if device_index is not None:
                dev_info = sd.query_devices(device_index)
            else:
                dev_info = sd.query_devices(kind='input')
                device_index = dev_info['index'] if 'index' in dev_info else None

            self.native_samplerate = int(dev_info.get("default_samplerate", 44100))
            self.native_channels = min(2, max(1, dev_info.get("max_input_channels", 1)))
            self.block_size = int(self.native_samplerate * 0.05) # ~50ms por bloco

            self.stream = sd.InputStream(
                device=device_index,
                channels=self.native_channels,
                samplerate=self.native_samplerate,
                blocksize=self.block_size,
                dtype=np.int16,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_recording = True
            source_name = LANG_DISPLAY_NAMES.get(self.source_lang, self.source_lang)
            self.status_changed.emit(f"🎙️ Microfone ativo ({self.native_samplerate}Hz) - captando áudio em {source_name}...")
        except Exception as e:
            self.is_recording = False
            self.error_occurred.emit(f"Erro ao abrir microfone: {e}")

    def stop_listening(self):
        if not self.is_recording:
            return

        self.is_recording = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"Erro ao fechar stream: {e}")

        # Se houver fala acumulada ao pausar, processa
        if self.speech_frames and len(self.speech_frames) > 3:
            full_audio = np.concatenate(self.speech_frames)
            self.recognition_queue.put((full_audio, self.native_samplerate))
        
        self.speech_frames = []
        self.is_speaking = False
        self.status_changed.emit("⏸️ Microfone pausado")
        self.level_changed.emit(0.0, [0] * 32)

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.is_recording:
            return

        # Converter canais (estéreo -> mono de alta qualidade)
        if self.native_channels > 1 and indata.shape[1] > 1:
            audio_mono = (indata[:, 0].astype(np.int32) + indata[:, 1].astype(np.int32)) // 2
            audio_mono = audio_mono.astype(np.int16)
        else:
            audio_mono = indata[:, 0]

        # Calcular nível RMS normalizado
        float_data = audio_mono.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(float_data ** 2))
        
        # Amostras para o visualizador de onda
        step = max(1, len(float_data) // 32)
        waveform_sample = [float(x) for x in float_data[::step][:32]]
        self.level_changed.emit(float(rms), waveform_sample)

        threshold = self._get_threshold()
        current_time = time.time()

        if rms > threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
                self.speech_frames = []
                self.speech_started.emit()
                source_name = LANG_DISPLAY_NAMES.get(self.source_lang, self.source_lang)
                self.status_changed.emit(f"🎧 Detectando áudio em {source_name}...")
            
            self.speech_frames.append(audio_mono.copy())
            self.last_speech_time = current_time
        else:
            if self.is_speaking:
                self.speech_frames.append(audio_mono.copy())
                # Disparar quando o silêncio atingir o limite
                if current_time - self.last_speech_time >= self.silence_timeout:
                    self.is_speaking = False
                    self.speech_ended.emit()
                    
                    duration = current_time - self.speech_start_time
                    if duration >= self.min_speech_duration and len(self.speech_frames) > 0:
                        full_audio = np.concatenate(self.speech_frames)
                        self.recognition_queue.put((full_audio, self.native_samplerate))
                        self.status_changed.emit("⚡ Reconhecendo com alta precisão e traduzindo...")
                    else:
                        self.status_changed.emit("🎙️ Aguardando fala...")
                    
                    self.speech_frames = []

    def _preprocess_audio(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Pré-processamento de estúdio:
        1. Remoção de DC Offset (zera ruído de fundo contínuo)
        2. Normalização Automática de Ganho (AGC) para voz ficar nítida e clara
        """
        audio_float = raw_audio.astype(np.float32)
        
        # 1. Remover DC offset
        audio_float = audio_float - np.mean(audio_float)
        
        # 2. Normalização de Pico / Ganho Inteligente
        max_val = np.max(np.abs(audio_float))
        if max_val > 50:
            target_peak = 27500.0  # ~85% da escala de 16-bit
            gain = min(15.0, target_peak / max_val) # limite de ganho de 15x para não estourar ruído
            audio_float = audio_float * gain

        # Limitar para int16
        audio_clipped = np.clip(audio_float, -32768, 32767).astype(np.int16)
        return audio_clipped

    def _recognition_worker(self):
        """Worker em thread separada para transcrever e traduzir com máxima fidelidade."""
        while True:
            try:
                item = self.recognition_queue.get()
                if item is None:
                    break

                audio_np, sample_rate = item
                if len(audio_np) == 0:
                    continue

                # Pré-processar áudio para ganho ideal e nitidez
                clean_audio = self._preprocess_audio(audio_np)
                raw_pcm = clean_audio.tobytes()
                audio_data = sr.AudioData(raw_pcm, sample_rate, 2)

                # 1. Reconhecimento de fala no idioma de origem configurado (padrão: inglês da live)
                speech_code = SPEECH_LANG_CODES.get(self.source_lang, "en-US")
                text_original = None
                try:
                    text_original = self.recognizer.recognize_google(
                        audio_data,
                        language=speech_code,
                        show_all=False
                    )
                except sr.UnknownValueError:
                    # Voz muito baixa ou som incompreensível
                    self.status_changed.emit("🎙️ Microfone pronto - aproxime melhor o áudio se necessário...")
                    continue
                except sr.RequestError as e:
                    self.error_occurred.emit(f"Erro de conexão com o servidor de voz: {e}")
                    continue

                if not text_original or not text_original.strip():
                    continue

                # Emite a transcrição original imediatamente
                self.transcription_ready.emit(text_original.strip())

                # 2. Tradução ultra-rápida para o idioma de destino (padrão: português)
                translated_text = self.translator.translate(text_original.strip(), source=self.source_lang, target=self.target_lang)
                self.translation_ready.emit(text_original.strip(), translated_text)
                self.status_changed.emit("✅ Transcrito e traduzido com sucesso!")

                # 3. Pronúncia da tradução (TTS) se configurado
                if self.settings.get("auto_tts", False):
                    self.tts.speak(translated_text)

            except Exception as e:
                print(f"Erro no worker de reconhecimento: {e}")
            finally:
                time.sleep(0.02)
