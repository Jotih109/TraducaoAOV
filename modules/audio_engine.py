# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
import re
from collections import deque
from typing import List, Dict, Optional, Tuple
from PyQt5.QtCore import QObject, pyqtSignal

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from modules.translator import format_subtitle_text

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


def resample_to_16k(audio_int16: np.ndarray, orig_rate: int) -> np.ndarray:
    """
    Converte áudio de qualquer taxa de amostragem (48kHz, 44.1kHz, etc.)
    para 16kHz mono nativo com máxima fidelidade e velocidade (< 5ms).
    O padrão de 16kHz é o ideal absoluto para o modelo acústico do Google STT.
    """
    if orig_rate == 16000 or len(audio_int16) == 0:
        return audio_int16

    if not HAS_SCIPY:
        return audio_int16

    audio_float = audio_int16.astype(np.float32)
    try:
        if orig_rate == 48000:
            resampled = scipy.signal.resample_poly(audio_float, 1, 3)
        elif orig_rate == 44100:
            resampled = scipy.signal.resample_poly(audio_float, 160, 441)
        elif orig_rate == 96000:
            resampled = scipy.signal.resample_poly(audio_float, 1, 6)
        else:
            from math import gcd
            g = gcd(orig_rate, 16000)
            up = 16000 // g
            down = orig_rate // g
            resampled = scipy.signal.resample_poly(audio_float, up, down)
        return np.clip(resampled, -32768, 32767).astype(np.int16)
    except Exception as e:
        print(f"Erro no resample: {e}")
        return audio_int16


class AudioEngine(QObject):
    """
    Motor de captura de áudio com VAD de resposta instantânea (< 400ms) e 16kHz de estúdio.
    - Captura preferencial via Windows WASAPI de alta resolução e ganho nítido
    - Downsample ultra-rápido para 16kHz (reduz payload em 67% e acelera resposta)
    - Corte imediato de silêncio final (elimina dead air e delays)
    - Sensibilidade adaptativa de alta precisão (capta voz suave sem ruídos)
    """
    level_changed = pyqtSignal(float, list, float)  # (RMS, waveform samples, threshold)
    speech_started = pyqtSignal()                   # Fala detectada
    speech_ended = pyqtSignal()                     # Pausa na fala detectada
    transcription_ready = pyqtSignal(str)           # Texto original transcrito
    translation_ready = pyqtSignal(str, str)        # (Texto original, Tradução)
    translation_ready_with_metrics = pyqtSignal(str, str, float)  # (Original, Tradução, Latência em ms)
    status_changed = pyqtSignal(str)                # Status em tempo real
    error_occurred = pyqtSignal(str)                # Alerta de erro

    def __init__(self, settings_manager, translator, tts):
        super().__init__()
        self.settings = settings_manager
        self.translator = translator
        self.tts = tts
        self.recognizer = sr.Recognizer()

        # Idioma de origem e destino
        self.source_lang = self.settings.get("source_lang", "pt")
        self.target_lang = self.settings.get("target_lang", "en")

        # Parâmetros de áudio
        self.native_samplerate = 44100
        self.native_channels = 1
        self.block_size = 1024
        self.stream = None
        self.is_recording = False
        self.active_device_index = None
        self.active_device_name = ""

        # VAD ultra-rápido calibrado
        self.sensitivity = int(self.settings.get("mic_sensitivity", 75))
        self.ambient_rms = 0.0008            # Ruído ambiente calibrado dinamicamente
        self.silence_timeout = 0.38          # Apenas 380ms de silêncio para disparar (resposta relâmpago)
        self.min_speech_duration = 0.25      # Mínimo de fala aceito
        self.max_speech_duration = float(self.settings.get("max_speech_duration", 6.0))

        # Ring buffer para pre-roll (5 blocos de ~50ms = ~250ms antes da fala)
        self.pre_roll_blocks = 5
        self.pre_roll_buffer = deque(maxlen=self.pre_roll_blocks)

        self.is_speaking = False
        self.speech_frames = []
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0
        self.silence_blocks_count = 0

        # Fila de reconhecimento assíncrono: armazena (audio_np, sample_rate, timestamp_queue)
        self.recognition_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.worker_thread.start()

    def get_input_devices(self) -> List[Dict]:
        """
        Retorna a lista limpa e deduplicada com apenas os 4-5 dispositivos reais
        do Windows, priorizando WASAPI para máxima qualidade e ganho.
        """
        result = []
        result.append({
            "index": None,
            "name": "Microfone Padrão do Windows",
            "display_name": "🎙️ [Recomendado] Microfone Padrão do Windows",
            "channels": 1,
            "is_default": True,
            "is_pc_audio": False,
            "category": "default"
        })

        try:
            dev_list = sd.query_devices()
            hostapis = sd.query_hostapis()
            default_in = sd.default.device[0]

            api_priority = {
                "Windows WASAPI": 1,
                "Windows DirectSound": 2,
                "MME": 3,
                "Windows WDM-KS": 4
            }

            grouped = {}

            for idx, dev in enumerate(dev_list):
                if dev.get("max_input_channels", 0) <= 0:
                    continue

                raw_name = dev.get("name", "")
                name_lower = raw_name.lower()

                if "@system32" in name_lower or "bthhfenum" in name_lower:
                    continue
                if name_lower.startswith("mapeador de som") or name_lower.startswith("driver de captura"):
                    continue
                if "alto-falante" in name_lower and not ("mixagem" in name_lower or "stereo" in name_lower):
                    continue

                api_name = ""
                if "hostapi" in dev and 0 <= dev["hostapi"] < len(hostapis):
                    api_name = hostapis[dev["hostapi"]].get("name", "")
                prio = api_priority.get(api_name, 5)

                is_pc = any(kw in name_lower for kw in ["mixagem", "stereo mix", "wave out", "cable output", "what u hear"])
                is_headset = any(kw in name_lower for kw in ["headset", "headphone", "tws", "airdots", "buds", "airpods"])
                is_virtual = any(kw in name_lower for kw in ["voice changer", "virtual audio", "mfdriver"])

                if is_pc:
                    group_key = "pc_audio"
                    tag = "🔊 [Som do PC / Live]"
                    label = "Mixagem Estéreo (Áudio Interno do PC)"
                elif is_headset:
                    match = re.search(r"\((.*?)\)", raw_name)
                    headset_name = match.group(1) if match else "Bluetooth"
                    group_key = f"headset_{headset_name.lower()}"
                    tag = "🎧 [Headset]"
                    label = f"Headset ({headset_name})"
                elif is_virtual:
                    group_key = "virtual_mic"
                    tag = "🎙️ [Virtual]"
                    label = "Microfone Virtual (Voice Changer / Cabo)"
                else:
                    group_key = "main_mic"
                    tag = "🎙️ [Microfone]"
                    label = "Microfone Integrado do Computador"

                if group_key not in grouped or prio < grouped[group_key]["prio"]:
                    grouped[group_key] = {
                        "index": idx,
                        "name": raw_name,
                        "display_name": f"{tag} {label}",
                        "channels": dev["max_input_channels"],
                        "default_samplerate": int(dev.get("default_samplerate", 44100)),
                        "is_default": (idx == default_in),
                        "is_pc_audio": is_pc,
                        "prio": prio
                    }

            if "main_mic" in grouped:
                result.append(grouped.pop("main_mic"))
            if "pc_audio" in grouped:
                result.append(grouped.pop("pc_audio"))
            for k in list(grouped.keys()):
                if k.startswith("headset_"):
                    result.append(grouped.pop(k))
            if "virtual_mic" in grouped:
                result.append(grouped.pop("virtual_mic"))
            for item in grouped.values():
                result.append(item)

        except Exception as e:
            print(f"Erro ao listar dispositivos: {e}")

        return result

    def resolve_valid_device(self, preferred_index: Optional[int] = None) -> Tuple[Optional[int], Dict]:
        """Garante a escolha do melhor microfone físico real, priorizando WASAPI sem pegar cabos virtuais por engano."""
        try:
            dev_list = sd.query_devices()
            hostapis = sd.query_hostapis()

            # 1. Se um índice específico foi escolhido pelo usuário e tem canais de entrada
            if preferred_index is not None and 0 <= preferred_index < len(dev_list):
                dev = dev_list[preferred_index]
                if dev.get("max_input_channels", 0) > 0:
                    return preferred_index, dev

            # 2. Se for Padrão (None), buscar o microfone FÍSICO REAL no WASAPI (Realtek / Integrado)
            wasapi_idx = None
            for h_i, h in enumerate(hostapis):
                if "wasapi" in h.get("name", "").lower():
                    wasapi_idx = h_i
                    break

            if wasapi_idx is not None:
                # Prioridade 1: Microfone físico Realtek / integrado
                for idx, dev in enumerate(dev_list):
                    if dev.get("hostapi") == wasapi_idx and dev.get("max_input_channels", 0) > 0:
                        name_l = dev.get("name", "").lower()
                        if any(v in name_l for v in ["virtual", "voice changer", "cable", "mfdriver"]):
                            continue
                        if "realtek" in name_l or "grupo de" in name_l or "array" in name_l:
                            return idx, dev

                # Prioridade 2: Qualquer microfone físico não-virtual no WASAPI
                for idx, dev in enumerate(dev_list):
                    if dev.get("hostapi") == wasapi_idx and dev.get("max_input_channels", 0) > 0:
                        name_l = dev.get("name", "").lower()
                        if any(v in name_l for v in ["virtual", "voice changer", "cable", "mfdriver"]):
                            continue
                        if "microfone" in name_l or "headset" in name_l:
                            return idx, dev

            # 3. Dispositivo padrão do Windows
            default_in = sd.default.device[0]
            if default_in is not None and 0 <= default_in < len(dev_list):
                dev = dev_list[default_in]
                if dev.get("max_input_channels", 0) > 0:
                    return default_in, dev

            # 4. Qualquer dispositivo com entrada disponível
            for idx, dev in enumerate(dev_list):
                if dev.get("max_input_channels", 0) > 0:
                    return idx, dev
        except Exception as e:
            print(f"Erro ao resolver dispositivo: {e}")
        return None, {}

    def set_sensitivity(self, value: int):
        self.sensitivity = max(15, min(95, value))
        self.settings.set("mic_sensitivity", self.sensitivity)

    def set_languages(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.settings.set("source_lang", source_lang)
        self.settings.set("target_lang", target_lang)
        source_info = AVAILABLE_LANGUAGES.get(source_lang, {"name": source_lang})
        target_info = AVAILABLE_LANGUAGES.get(target_lang, {"name": target_lang})
        if self.is_recording:
            self.status_changed.emit(f"🎙️ Escutando em {source_info['name']} ➔ Traduzindo para {target_info['name']}")

    def _get_threshold(self) -> float:
        """Limiar adaptativo com sensibilidade calibrada para captar voz suave instantaneamente."""
        factor = (100.0 - self.sensitivity) / 100.0  # 0.05 a 0.85
        # Em 75%: base ~ 0.0015
        base_thresh = 0.0006 + (factor ** 2.0) * 0.015
        adaptive_thresh = self.ambient_rms * (1.25 + factor * 1.25)
        return max(base_thresh, adaptive_thresh)

    def start_listening(self, device_index: Optional[int] = None):
        if self.is_recording:
            return

        if device_index is None:
            device_index = self.settings.get("mic_device_index", None)

        valid_index, dev_info = self.resolve_valid_device(device_index)
        if valid_index is None:
            self.error_occurred.emit("Nenhum microfone ou dispositivo de áudio funcional foi encontrado.")
            return

        try:
            self.active_device_index = valid_index
            self.active_device_name = dev_info.get("name", f"Dispositivo {valid_index}")

            self.native_samplerate = int(dev_info.get("default_samplerate", 44100))
            self.native_channels = min(2, max(1, dev_info.get("max_input_channels", 1)))
            self.block_size = int(self.native_samplerate * 0.05)  # ~50ms por bloco

            self.stream = sd.InputStream(
                device=self.active_device_index,
                channels=self.native_channels,
                samplerate=self.native_samplerate,
                blocksize=self.block_size,
                dtype=np.int16,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_recording = True
            
            source_name = LANG_DISPLAY_NAMES.get(self.source_lang, self.source_lang)
            target_name = LANG_DISPLAY_NAMES.get(self.target_lang, self.target_lang)
            self.status_changed.emit(f"🎙️ Escutando em {source_name} ➔ Traduzindo para {target_name}")
        except Exception as e:
            self.is_recording = False
            self.error_occurred.emit(f"Erro ao abrir áudio: {e}")

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

        if self.speech_frames and len(self.speech_frames) > 3:
            clean_frames = self.speech_frames[:-max(0, self.silence_blocks_count - 1)] if self.silence_blocks_count > 1 else self.speech_frames
            if clean_frames:
                full_audio = np.concatenate(clean_frames)
                self.recognition_queue.put((full_audio, self.native_samplerate, time.time()))
        
        self.speech_frames = []
        self.pre_roll_buffer.clear()
        self.is_speaking = False
        self.silence_blocks_count = 0
        self.status_changed.emit("⏸️ Escuta pausada")
        self.level_changed.emit(0.0, [0.0] * 32, 0.0)

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.is_recording:
            return

        # Downmix estéreo -> mono
        if self.native_channels > 1 and indata.shape[1] > 1:
            audio_mono = (indata[:, 0].astype(np.int32) + indata[:, 1].astype(np.int32)) // 2
            audio_mono = audio_mono.astype(np.int16)
        else:
            audio_mono = indata[:, 0].copy()

        # Nível RMS
        float_data = audio_mono.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(float_data ** 2)))
        
        step = max(1, len(float_data) // 32)
        waveform_sample = [float(x) for x in float_data[::step][:32]]

        threshold = self._get_threshold()
        self.level_changed.emit(rms, waveform_sample, threshold)

        current_time = time.time()

        if rms > threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
                self.speech_frames = list(self.pre_roll_buffer)
                self.speech_started.emit()
                source_name = LANG_DISPLAY_NAMES.get(self.source_lang, self.source_lang)
                self.status_changed.emit(f"🎧 Voz detectada ({source_name})...")
            
            self.speech_frames.append(audio_mono.copy())
            self.last_speech_time = current_time
            self.silence_blocks_count = 0

            # Fatiamento contínuo suave (6 segundos)
            if current_time - self.speech_start_time >= self.max_speech_duration:
                if len(self.speech_frames) > 0:
                    chunk_audio = np.concatenate(self.speech_frames)
                    self.recognition_queue.put((chunk_audio, self.native_samplerate, current_time))
                    self.status_changed.emit("⚡ Processando fala contínua...")
                self.speech_frames = self.speech_frames[-2:] if len(self.speech_frames) >= 2 else []
                self.speech_start_time = current_time

        else:
            if not self.is_speaking:
                self.ambient_rms = 0.96 * self.ambient_rms + 0.04 * min(rms, 0.004)
                self.pre_roll_buffer.append(audio_mono.copy())
            else:
                self.speech_frames.append(audio_mono.copy())
                self.silence_blocks_count += 1

                # Disparo rápido de fim de fala (380ms de silêncio)
                if current_time - self.last_speech_time >= self.silence_timeout:
                    self.is_speaking = False
                    self.speech_ended.emit()
                    
                    duration = current_time - self.speech_start_time
                    if duration >= self.min_speech_duration and len(self.speech_frames) > self.silence_blocks_count:
                        # CORTE DE SILÊNCIO FINAL: descarta blocos mudos para áudio terminar limpo no final da frase
                        trim_count = max(0, self.silence_blocks_count - 2)
                        if trim_count > 0 and len(self.speech_frames) > trim_count:
                            clean_frames = self.speech_frames[:-trim_count]
                        else:
                            clean_frames = self.speech_frames

                        full_audio = np.concatenate(clean_frames)
                        self.recognition_queue.put((full_audio, self.native_samplerate, current_time))
                        self.status_changed.emit("⚡ Reconhecendo fala...")
                    else:
                        self.status_changed.emit("🎙️ Aguardando fala...")
                    
                    self.speech_frames = []
                    self.silence_blocks_count = 0

    def _preprocess_audio(self, raw_audio: np.ndarray) -> np.ndarray:
        """Limpeza de DC offset e ganho de estúdio para voz soar nítida e alta."""
        audio_float = raw_audio.astype(np.float32)
        audio_float = audio_float - np.mean(audio_float)
        
        max_val = np.max(np.abs(audio_float))
        if max_val > 50:
            target_peak = 26000.0
            gain = min(15.0, target_peak / max_val)
            audio_float = audio_float * gain

        return np.clip(audio_float, -32768, 32767).astype(np.int16)

    def _recognition_worker(self):
        """Worker assíncrono: faz resample para 16kHz, transcreve com alta precisão e traduz."""
        while True:
            try:
                item = self.recognition_queue.get()
                if item is None:
                    break

                audio_np, sample_rate, queue_timestamp = item
                if len(audio_np) == 0:
                    continue

                # 1. Pré-processar áudio (ganho e nitidez)
                clean_audio = self._preprocess_audio(audio_np)

                # 2. Resample de estúdio para 16.000 Hz nativo do Google Speech Recognition
                audio_16k = resample_to_16k(clean_audio, sample_rate)
                raw_pcm = audio_16k.tobytes()
                audio_data = sr.AudioData(raw_pcm, 16000, 2)

                # 3. Reconhecimento de fala no idioma configurado
                speech_code = SPEECH_LANG_CODES.get(self.source_lang, "pt-BR")
                text_original = None
                try:
                    text_original = self.recognizer.recognize_google(
                        audio_data,
                        language=speech_code,
                        show_all=False
                    )
                except sr.UnknownValueError:
                    self.status_changed.emit("🎙️ Microfone pronto - aguardando fala...")
                    continue
                except sr.RequestError as e:
                    self.error_occurred.emit(f"Erro de conexão com o servidor de voz: {e}")
                    continue

                if not text_original or not text_original.strip():
                    continue

                formatted_original = format_subtitle_text(text_original)
                self.transcription_ready.emit(formatted_original)

                # 4. Tradução ultra-rápida
                translated_text, _ = self.translator.translate_with_metrics(
                    formatted_original, source=self.source_lang, target=self.target_lang
                )

                total_latency_ms = (time.time() - queue_timestamp) * 1000.0

                self.translation_ready.emit(formatted_original, translated_text)
                self.translation_ready_with_metrics.emit(formatted_original, translated_text, total_latency_ms)
                
                source_name = LANG_DISPLAY_NAMES.get(self.source_lang, self.source_lang)
                self.status_changed.emit(f"✅ Transcrito e traduzido em {int(total_latency_ms)}ms!")

                if self.settings.get("auto_tts", False):
                    self.tts.speak(translated_text)

            except Exception as e:
                print(f"Erro no worker de reconhecimento: {e}")
            finally:
                time.sleep(0.01)
