# -*- coding: utf-8 -*-
"""Captura de áudio WASAPI loopback nativa, sem Stereo Mix nem cabos virtuais.

Responsabilidade única: entregar um fluxo contínuo de frames de **512 amostras
float32 mono a 16 kHz** (32 ms) vindos do dispositivo de *saída* padrão do
Windows, com carimbo temporal e tolerância a trocas de dispositivo a quente.

Notas de implementação relevantes
---------------------------------

**O WASAPI não nos dá 16 kHz.** Em shared mode o PortAudio abre a stream no
*mix format* do endpoint (tipicamente 48 kHz estéreo). Pedir 16 kHz mono ao
driver falha ou é silenciosamente convertido com qualidade imprevisível. Por
isso a conversão é feita aqui, com :class:`PolyphaseStreamResampler`.

**O resampler tem de ter estado.** Chamar ``scipy.signal.resample_poly`` chunk a
chunk é o erro comum: cada chamada assume zeros fora do bloco, o que injecta uma
descontinuidade a cada 32 ms. Isso degrada o VAD (transientes falsos) e o
log-mel do Whisper. A classe abaixo mantém histórico de entrada e a fase do
decimador entre chamadas, produzindo bit-a-bit o mesmo resultado que um
``resample_poly`` sobre a stream inteira.

**O PortAudio fotografa a lista de dispositivos no ``Pa_Initialize``.** Não há
como ver um novo dispositivo padrão sem terminar e reinicializar a biblioteca —
daí o ciclo de reabertura em :meth:`LoopbackCapture._run` criar sempre uma
instância nova de ``PyAudio``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Final, Iterator, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import signal

try:  # pragma: no cover - depende do sistema
    import pyaudiowpatch as pyaudio

    HAS_PYAUDIO: Final[bool] = True
except ImportError:  # pragma: no cover
    pyaudio = None  # type: ignore[assignment]
    HAS_PYAUDIO = False

from pipeline.config import FRAME_SAMPLES, TARGET_RATE, AudioConfig

__all__ = [
    "AudioDeviceError",
    "AudioRingBuffer",
    "CaptureStats",
    "DeviceInfo",
    "FrameSink",
    "HAS_PYAUDIO",
    "LoopbackCapture",
    "PolyphaseStreamResampler",
    "list_loopback_devices",
]

LOG = logging.getLogger(__name__)

_INT16_SCALE: Final[np.float32] = np.float32(1.0 / 32768.0)
_MAX_BACKOFF: Final[float] = 5.0
_EMPTY: Final[NDArray[np.float32]] = np.zeros(0, dtype=np.float32)


class AudioDeviceError(RuntimeError):
    """Nenhum endpoint de loopback WASAPI utilizável foi encontrado."""


class FrameSink(Protocol):
    """Destino de cada frame de 512 amostras.

    Chamado a partir da thread de captura. Deve retornar em muito menos de 32 ms
    ou a stream acumula atraso — trabalho pesado pertence a outra thread.
    """

    def __call__(self, frame: NDArray[np.float32], captured_at: float) -> None:
        """Recebe ``frame`` (512 amostras float32 em [-1, 1]).

        Args:
            frame: Vista só-de-leitura sobre o buffer interno. Copie antes de
                guardar; o conteúdo é reutilizado no frame seguinte.
            captured_at: ``time.monotonic()`` aproximado do último sample.
        """


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Endpoint de loopback resolvido."""

    index: int
    name: str
    sample_rate: int
    channels: int

    @property
    def key(self) -> tuple[str, int, int]:
        """Identidade estável do endpoint, imune a renumeração de índices."""
        return (self.name, self.sample_rate, self.channels)


@dataclass(slots=True)
class CaptureStats:
    """Contadores de diagnóstico da captura."""

    frames_emitted: int = 0
    reads: int = 0
    overflows: int = 0
    restarts: int = 0
    device_changes: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def seconds_captured(self) -> float:
        return self.frames_emitted * FRAME_SAMPLES / TARGET_RATE


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class PolyphaseStreamResampler:
    """Reamostrador racional polifásico com estado, para uso em streaming.

    Converte ``src_rate`` para ``dst_rate`` através de um filtro FIR único
    (janela de Kaiser), interpolando por ``up`` e decimando por ``down``, onde
    ``up/down = dst_rate/src_rate`` na forma irredutível.

    A continuidade entre chamadas é garantida guardando as últimas amostras de
    entrada que ainda influenciam o suporte do filtro e alinhando o início do
    buffer a um múltiplo de ``down``, de modo a que o índice do primeiro sample
    de saída do bloco seja sempre inteiro. A saída é ancorada no centro do FIR
    (atraso de grupo ``half_len``), ficando alinhada no tempo com a entrada
    exactamente como em ``resample_poly``. Não há descontinuidades nas
    fronteiras dos blocos: o resultado é idêntico, amostra a amostra, ao de um
    ``resample_poly`` aplicado à stream inteira de uma só vez.

    O único custo é um atraso intrínseco de ``half_len / down`` amostras de
    saída (0,6 ms a 48 kHz), que é a latência inevitável do filtro.

    Args:
        src_rate: Taxa de entrada, em Hz.
        dst_rate: Taxa de saída, em Hz.
        taps_per_phase: Coeficientes por fase polifásica. 20 iguala a qualidade
            por omissão do ``scipy.signal.resample_poly``.
        beta: Parâmetro da janela de Kaiser.

    Example:
        >>> r = PolyphaseStreamResampler(48000, 16000)
        >>> first = r.process(np.zeros(960, dtype=np.float32))   # 20 ms
        >>> second = r.process(np.zeros(960, dtype=np.float32))
        >>> first.size, second.size   # 10 amostras retidas pelo atraso de grupo
        (310, 320)
    """

    __slots__ = (
        "_buffer",
        "_buffer_start",
        "_down",
        "_half_len",
        "_passthrough",
        "_taps",
        "_total_in",
        "_total_out",
        "_up",
        "dst_rate",
        "src_rate",
    )

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        taps_per_phase: int = 20,
        beta: float = 5.0,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"taxas inválidas: {src_rate} -> {dst_rate}")
        self.src_rate = src_rate
        self.dst_rate = dst_rate

        divisor = math.gcd(src_rate, dst_rate)
        self._up = dst_rate // divisor
        self._down = src_rate // divisor
        self._passthrough = self._up == 1 and self._down == 1

        self._buffer: NDArray[np.float32] = _EMPTY
        self._buffer_start = 0
        self._total_in = 0
        self._total_out = 0

        if self._passthrough:
            self._taps = _EMPTY
            self._half_len = 0
            return

        max_rate = max(self._up, self._down)
        # half_len é arredondado para cima até um múltiplo de ``down``. Isso faz
        # com que o atraso de grupo seja um número inteiro de amostras de saída,
        # o que por sua vez mantém a grelha do ``upfirdn`` alinhada com a grelha
        # global sempre que o buffer começa num múltiplo de ``down``.
        self._half_len = (
            -(-(taps_per_phase // 2) * max_rate // self._down) * self._down
        )
        # Corte na Nyquist mais restritiva das duas taxas, normalizado à Nyquist
        # do domínio sobreamostrado (up * src_rate).
        taps = signal.firwin(
            2 * self._half_len + 1, 1.0 / max_rate, window=("kaiser", beta)
        ) * self._up
        self._taps = taps.astype(np.float32, copy=False)

    @property
    def ratio(self) -> float:
        """Amostras de saída por amostra de entrada."""
        return self._up / self._down

    def reset(self) -> None:
        """Esquece o histórico. Usar apenas ao trocar de dispositivo."""
        self._buffer = _EMPTY
        self._buffer_start = 0
        self._total_in = 0
        self._total_out = 0

    def process(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """Reamostra um bloco, dando continuidade ao bloco anterior.

        Args:
            samples: Bloco mono float32.

        Returns:
            Bloco reamostrado. O comprimento varia entre chamadas (a razão
            raramente é inteira); pode ser vazio se ainda não houver saída.
        """
        if samples.size == 0:
            return _EMPTY
        if self._passthrough:
            return samples

        self._buffer = (
            samples.copy() if self._buffer.size == 0
            else np.concatenate((self._buffer, samples))
        )
        self._total_in += samples.size

        # A saída n lê a convolução no índice sobreamostrado n*down + half_len,
        # logo precisa de entrada até floor((n*down + half_len) / up). Daqui sai
        # o número de saídas já produzíveis com self._total_in entradas.
        end = max(
            0,
            -(-(self._total_in * self._up - self._half_len) // self._down),
        )
        if end <= self._total_out:
            return _EMPTY

        # upfirdn devolve a convolução do buffer local decimada por ``down``.
        # Como _buffer_start e _half_len são ambos múltiplos de ``down``, a
        # saída global n cai exactamente no índice local n - base.
        local = signal.upfirdn(self._taps, self._buffer, self._up, self._down)
        base = (self._buffer_start * self._up - self._half_len) // self._down
        out = local[self._total_out - base : end - base]
        self._total_out = end

        # Descarta o histórico que a próxima saída já não pode alcançar,
        # preservando o alinhamento a múltiplos de ``down``.
        next_tap = self._total_out * self._down + self._half_len
        need_from = max(0, -(-(next_tap - self._taps.size + 1) // self._up))
        keep_from = need_from - (need_from % self._down)
        if keep_from > self._buffer_start:
            self._buffer = self._buffer[keep_from - self._buffer_start :]
            self._buffer_start = keep_from

        return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class AudioRingBuffer:
    """Buffer circular de capacidade fixa para áudio mono float32.

    Alocado uma única vez; as escritas nunca alocam. Usado para o pre-roll do
    VAD (200 ms de áudio anterior ao início da fala), que evita cortar a
    primeira sílaba de cada elocução.

    Não é thread-safe: destina-se a um único produtor e um único consumidor na
    mesma thread (a thread de VAD).
    """

    __slots__ = ("_data", "_filled", "_write")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacidade tem de ser positiva")
        self._data: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._filled = 0

    def __len__(self) -> int:
        return self._filled

    @property
    def capacity(self) -> int:
        return self._data.size

    def clear(self) -> None:
        self._write = 0
        self._filled = 0

    def write(self, samples: NDArray[np.float32]) -> None:
        """Escreve um bloco, sobrepondo as amostras mais antigas se necessário."""
        count = samples.size
        capacity = self._data.size
        if count == 0:
            return
        if count >= capacity:
            self._data[:] = samples[-capacity:]
            self._write = 0
            self._filled = capacity
            return

        end = self._write + count
        if end <= capacity:
            self._data[self._write : end] = samples
        else:
            split = capacity - self._write
            self._data[self._write :] = samples[:split]
            self._data[: count - split] = samples[split:]
        self._write = end % capacity
        self._filled = min(capacity, self._filled + count)

    def read_last(self, count: int | None = None) -> NDArray[np.float32]:
        """Devolve uma cópia contígua das últimas ``count`` amostras."""
        available = self._filled
        wanted = available if count is None else min(count, available)
        if wanted <= 0:
            return _EMPTY
        capacity = self._data.size
        start = (self._write - wanted) % capacity
        end = start + wanted
        if end <= capacity:
            return self._data[start:end].copy()
        return np.concatenate((self._data[start:], self._data[: end - capacity]))


# ---------------------------------------------------------------------------
# Resolução de dispositivos
# ---------------------------------------------------------------------------


def _require_pyaudio() -> None:
    if not HAS_PYAUDIO:
        raise AudioDeviceError(
            "PyAudioWPatch não está instalado. Execute: pip install PyAudioWPatch"
        )


def _resolve_default_loopback(handle: "pyaudio.PyAudio") -> DeviceInfo:
    """Encontra o endpoint de loopback do dispositivo de saída padrão."""
    try:
        wasapi = handle.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:  # host API ausente (Windows muito antigo / Wine)
        raise AudioDeviceError("WASAPI indisponível neste sistema") from exc

    index = int(wasapi.get("defaultOutputDevice", -1))
    if index < 0:
        raise AudioDeviceError("o Windows não reporta dispositivo de saída padrão")

    speakers = handle.get_device_info_by_index(index)
    if not speakers.get("isLoopbackDevice", False):
        # O endpoint de render tem um gémeo de loopback com o mesmo nome.
        for candidate in handle.get_loopback_device_info_generator():
            if str(speakers["name"]) in str(candidate["name"]):
                speakers = candidate
                break
        else:
            raise AudioDeviceError(
                f"sem loopback para o dispositivo padrão {speakers['name']!r}"
            )

    return DeviceInfo(
        index=int(speakers["index"]),
        name=str(speakers["name"]),
        sample_rate=int(speakers["defaultSampleRate"]),
        channels=max(1, int(speakers["maxInputChannels"])),
    )


def _resolve_explicit(handle: "pyaudio.PyAudio", index: int) -> DeviceInfo:
    info = handle.get_device_info_by_index(index)
    if int(info.get("maxInputChannels", 0)) <= 0:
        raise AudioDeviceError(f"dispositivo {index} não tem canais de entrada")
    return DeviceInfo(
        index=index,
        name=str(info["name"]),
        sample_rate=int(info["defaultSampleRate"]),
        channels=max(1, int(info["maxInputChannels"])),
    )


def list_loopback_devices() -> list[DeviceInfo]:
    """Enumera todos os endpoints de loopback WASAPI disponíveis."""
    _require_pyaudio()
    handle = pyaudio.PyAudio()
    try:
        return [
            DeviceInfo(
                index=int(dev["index"]),
                name=str(dev["name"]),
                sample_rate=int(dev["defaultSampleRate"]),
                channels=max(1, int(dev["maxInputChannels"])),
            )
            for dev in handle.get_loopback_device_info_generator()
        ]
    finally:
        handle.terminate()


# ---------------------------------------------------------------------------
# Captura
# ---------------------------------------------------------------------------


class LoopbackCapture:
    """Thread de captura WASAPI loopback com reconexão automática.

    Usa leitura bloqueante em vez do modo callback do PortAudio: o callback
    corre numa thread nativa que teria de adquirir o GIL, e qualquer pausa do
    interpretador nesse ponto provoca *glitches* na stream. Com leitura
    bloqueante o pior caso é um overrun, que o PortAudio resolve descartando
    áudio antigo sem partir a stream.

    Args:
        sink: Destino de cada frame de 512 amostras.
        config: Parâmetros de captura.
        on_status: Notificação de eventos relevantes (troca de dispositivo,
            reconexão). Chamado da thread de captura.

    Example:
        >>> capture = LoopbackCapture(lambda frame, t: None)  # doctest: +SKIP
        >>> capture.start()                                    # doctest: +SKIP
        >>> capture.stop()                                     # doctest: +SKIP
    """

    def __init__(
        self,
        sink: FrameSink,
        config: AudioConfig | None = None,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._sink = sink
        self._config = config or AudioConfig()
        self._on_status = on_status or (lambda _: None)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device: DeviceInfo | None = None
        self._device_lock = threading.Lock()
        self.stats = CaptureStats()

        # Estado de montagem de frames de 512 amostras.
        self._pending: list[NDArray[np.float32]] = []
        self._pending_size = 0
        self._resampler: PolyphaseStreamResampler | None = None

    # -- ciclo de vida ---------------------------------------------------

    @property
    def device(self) -> DeviceInfo | None:
        """Dispositivo actualmente aberto, ou ``None`` se parado."""
        with self._device_lock:
            return self._device

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Arranca a thread de captura. Idempotente."""
        if self.running:
            return
        _require_pyaudio()
        self._stop.clear()
        self.stats = CaptureStats()
        self._thread = threading.Thread(
            target=self._run, name="wasapi-loopback", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Sinaliza a paragem e espera pela thread."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self._thread = None

    def __enter__(self) -> LoopbackCapture:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- ciclo interno ---------------------------------------------------

    def _run(self) -> None:
        backoff = self._config.restart_backoff_seconds
        while not self._stop.is_set():
            handle = None
            stream = None
            try:
                handle = pyaudio.PyAudio()
                device = (
                    _resolve_explicit(handle, self._config.device_index)
                    if self._config.device_index is not None
                    else _resolve_default_loopback(handle)
                )
                stream = handle.open(
                    format=pyaudio.paInt16,
                    channels=device.channels,
                    rate=device.sample_rate,
                    input=True,
                    input_device_index=device.index,
                    frames_per_buffer=self._block_frames(device.sample_rate),
                )
                self._on_device_opened(device)
                backoff = self._config.restart_backoff_seconds
                self._pump(stream, device)
            except AudioDeviceError as exc:
                LOG.warning("captura indisponível: %s", exc)
                self._on_status(f"Áudio indisponível: {exc}")
            except OSError as exc:
                LOG.warning("stream de áudio falhou: %s", exc)
                self._on_status("Stream de áudio perdida; a reconectar...")
            except Exception:  # pragma: no cover - rede de segurança
                LOG.exception("erro inesperado na captura")
            finally:
                _close_quietly(stream, handle)
                with self._device_lock:
                    self._device = None

            if self._stop.is_set():
                break
            self.stats.restarts += 1
            self._stop.wait(backoff)
            backoff = min(_MAX_BACKOFF, backoff * 2.0)

    def _block_frames(self, sample_rate: int) -> int:
        return max(64, int(round(sample_rate * self._config.read_ms / 1000.0)))

    def _on_device_opened(self, device: DeviceInfo) -> None:
        with self._device_lock:
            previous = self._device
            self._device = device
        if previous is not None and previous.key != device.key:
            self.stats.device_changes += 1
        LOG.info(
            "loopback aberto: %s @ %d Hz, %d canal(is)",
            device.name,
            device.sample_rate,
            device.channels,
        )
        self._on_status(f"Áudio: {device.name}")
        self._resampler = PolyphaseStreamResampler(
            device.sample_rate,
            TARGET_RATE,
            taps_per_phase=self._config.resampler_taps_per_phase,
        )
        self._pending.clear()
        self._pending_size = 0

    def _pump(self, stream: "pyaudio.Stream", device: DeviceInfo) -> None:
        """Ciclo de leitura de uma stream aberta. Retorna para forçar reabertura."""
        block = self._block_frames(device.sample_rate)
        channels = device.channels
        assert self._resampler is not None

        silence_since: float | None = None
        probe_interval = self._config.silence_probe_seconds

        while not self._stop.is_set():
            raw = stream.read(block, exception_on_overflow=False)
            now = time.monotonic()
            self.stats.reads += 1

            mono = _to_mono_float32(raw, channels)
            if mono.size == 0:
                continue

            peak = float(np.max(np.abs(mono)))
            if peak < self._config.silence_floor:
                if silence_since is None:
                    silence_since = now
                elif now - silence_since >= probe_interval:
                    # Silêncio digital prolongado: ou não toca nada, ou o
                    # dispositivo padrão mudou. Reabrir é a única forma de saber
                    # (o PortAudio só relê a lista de dispositivos no init).
                    if self._default_device_changed(device):
                        LOG.info("dispositivo de saída padrão mudou; a reabrir")
                        self._on_status("Dispositivo de saída alterado")
                        return
                    silence_since = now
                    probe_interval = min(60.0, probe_interval * 2.0)
            else:
                silence_since = None
                probe_interval = self._config.silence_probe_seconds

            for frame in self._framer(self._resampler.process(mono)):
                self.stats.frames_emitted += 1
                self._sink(frame, now)

    def _framer(
        self, samples: NDArray[np.float32]
    ) -> Iterator[NDArray[np.float32]]:
        """Agrupa um fluxo de comprimento variável em frames de 512 amostras."""
        if samples.size:
            self._pending.append(samples)
            self._pending_size += samples.size
        if self._pending_size < FRAME_SAMPLES:
            return

        merged = (
            self._pending[0] if len(self._pending) == 1
            else np.concatenate(self._pending)
        )
        offset = 0
        limit = merged.size - FRAME_SAMPLES
        while offset <= limit:
            yield merged[offset : offset + FRAME_SAMPLES]
            offset += FRAME_SAMPLES

        leftover = merged[offset:]
        self._pending = [leftover.copy()] if leftover.size else []
        self._pending_size = leftover.size

    def _default_device_changed(self, current: DeviceInfo) -> bool:
        """Re-sonda o Windows com uma instância nova de PortAudio."""
        if self._config.device_index is not None:
            return False
        handle = None
        try:
            handle = pyaudio.PyAudio()
            return _resolve_default_loopback(handle).key != current.key
        except (AudioDeviceError, OSError):
            return True  # não conseguimos resolver: reabrir e deixar o ciclo tratar
        finally:
            if handle is not None:
                handle.terminate()


def _to_mono_float32(raw: bytes, channels: int) -> NDArray[np.float32]:
    """Converte PCM 16-bit intercalado para mono float32 normalizado.

    Duas alocações no total: ``frombuffer`` é uma vista sem cópia sobre os bytes
    do PortAudio, a redução de canais produz o array float32 final e a
    normalização é feita no lugar.
    """
    if len(raw) < 2:
        return _EMPTY
    pcm = np.frombuffer(raw, dtype=np.int16, count=len(raw) // 2)
    if channels > 1:
        usable = pcm.size - (pcm.size % channels)
        if usable == 0:
            return _EMPTY
        mono = pcm[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)
    else:
        mono = pcm.astype(np.float32)
    mono *= _INT16_SCALE
    return mono


def _close_quietly(
    stream: "pyaudio.Stream | None", handle: "pyaudio.PyAudio | None"
) -> None:
    if stream is not None:
        try:
            stream.stop_stream()
            stream.close()
        except (OSError, AttributeError):
            LOG.debug("falha ao fechar a stream", exc_info=True)
    if handle is not None:
        try:
            handle.terminate()
        except Exception:  # pragma: no cover
            LOG.debug("falha ao terminar o PortAudio", exc_info=True)
