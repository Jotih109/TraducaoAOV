# -*- coding: utf-8 -*-
"""Detecção neural de voz (Silero VAD sobre ONNX Runtime) e segmentação.

Dois objectos, com responsabilidades separadas de propósito:

:class:`SileroVad`
    Envelope puro do modelo. Recebe 512 amostras, devolve a probabilidade de
    voz. Sem estado de aplicação, sem política — só inferência.

:class:`SpeechSegmenter`
    A máquina de estados finita que transforma esse fluxo de probabilidades em
    eventos de elocução, com ring buffer de pre-roll e histerese::

        LISTENING ──p > start──▶ SPEAKING ──p < end──▶ TRAILING ──300 ms──▶ EOS
             ▲                      ▲                     │            │
             └──────────────────────┴───── p > start ──────┘            │
             └───────────────────────────────────────────────────────────┘

    O estado TRAILING continua a encaminhar áudio para o STT: os 300 ms de
    tolerância podem conter consoantes surdas de baixa energia, e cortá-las
    decapitaria o fim da frase. Se a voz regressar durante o TRAILING, volta-se
    a SPEAKING sem emitir EOS — é isso que evita partir uma frase a meio numa
    pausa para respirar.

O custo por frame é de cerca de 0,2 ms numa única thread de CPU, ou seja menos
de 1 % de um núcleo para um fluxo contínuo de frames de 32 ms.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - depende do sistema
    import onnxruntime as ort

    HAS_ONNXRUNTIME: Final[bool] = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]
    HAS_ONNXRUNTIME = False

from pipeline.audio_capture import AudioRingBuffer
from pipeline.config import FRAME_SAMPLES, MODELS_DIR, TARGET_RATE, VadConfig
from pipeline.protocol import VadPhase

__all__ = [
    "HAS_ONNXRUNTIME",
    "SileroVad",
    "SegmenterStats",
    "SpeechSegmenter",
    "VadEvent",
    "VadEventKind",
    "VadModelError",
    "ensure_silero_model",
    "frame_rms",
]

LOG = logging.getLogger(__name__)

_MODEL_FILENAME: Final[str] = "silero_vad.onnx"
_MODEL_URLS: Final[tuple[str, ...]] = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.onnx",
)
_MIN_MODEL_BYTES: Final[int] = 500_000
_ENV_OVERRIDE: Final[str] = "SILERO_VAD_ONNX"


class VadModelError(RuntimeError):
    """O modelo Silero VAD não pôde ser carregado."""


class VadEventKind(str, Enum):
    """Tipos de evento emitidos pelo segmentador."""

    SPEECH_START = "speech_start"
    """Início de elocução. ``samples`` traz o pre-roll seguido do frame actual."""

    SPEECH_AUDIO = "speech_audio"
    """Continuação da elocução (um frame)."""

    SPEECH_END = "speech_end"
    """Fim de fala confirmado (EOS). ``samples`` é ``None``."""

    SPEECH_DISCARDED = "speech_discarded"
    """A elocução foi curta demais; o consumidor deve descartar o que acumulou."""


@dataclass(frozen=True, slots=True)
class VadEvent:
    """Evento de segmentação.

    Attributes:
        kind: Tipo do evento.
        samples: Áudio associado, ou ``None`` para eventos de fim.
        captured_at: ``time.monotonic()`` do último sample do frame que gerou
            o evento. Propagado até à UI para medir latência ponta-a-ponta.
        probability: Probabilidade de voz do frame.
        duration: Duração total da elocução (fala mais o trailing), preenchida
            em eventos de fim.
        voiced_duration: Só a parte com voz, excluindo o trailing. É este valor
            que decide se a elocução é curta demais — usar ``duration`` faria
            com que ``min_speech_ms`` abaixo de ``trailing_ms`` nunca filtrasse
            nada.
    """

    kind: VadEventKind
    samples: NDArray[np.float32] | None
    captured_at: float
    probability: float
    duration: float = 0.0
    voiced_duration: float = 0.0


@dataclass(slots=True)
class SegmenterStats:
    """Contadores de diagnóstico da segmentação."""

    frames: int = 0
    utterances: int = 0
    discarded: int = 0
    speech_frames: int = 0

    @property
    def speech_ratio(self) -> float:
        return self.speech_frames / self.frames if self.frames else 0.0


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def ensure_silero_model(
    explicit: str | os.PathLike[str] | None = None,
    *,
    models_dir: Path = MODELS_DIR,
    timeout: float = 30.0,
    allow_download: bool = True,
) -> Path:
    """Localiza o ``silero_vad.onnx``, descarregando-o se necessário.

    A ordem de procura é: caminho explícito, variável de ambiente
    ``SILERO_VAD_ONNX``, cache local em ``models/`` e, por fim, download.

    Args:
        explicit: Caminho indicado na configuração.
        models_dir: Directório de cache.
        timeout: Tempo limite do download, em segundos.
        allow_download: Se ``False``, falha em vez de aceder à rede.

    Returns:
        Caminho para o ficheiro do modelo.

    Raises:
        VadModelError: Se o modelo não existir e não puder ser obtido.
    """
    for candidate in (explicit, os.environ.get(_ENV_OVERRIDE)):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path
            raise VadModelError(f"modelo Silero indicado não existe: {path}")

    cached = models_dir / _MODEL_FILENAME
    if cached.is_file() and cached.stat().st_size >= _MIN_MODEL_BYTES:
        return cached

    if not allow_download:
        raise VadModelError(
            f"{cached} em falta. Coloque lá o silero_vad.onnx ou defina "
            f"{_ENV_OVERRIDE}."
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in _MODEL_URLS:
        LOG.info("a descarregar o modelo Silero VAD de %s", url)
        try:
            _download(url, cached, timeout=timeout)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        LOG.info("modelo guardado em %s (%d bytes)", cached, cached.stat().st_size)
        return cached

    raise VadModelError(
        "não foi possível obter o silero_vad.onnx. Descarregue-o manualmente "
        f"para {cached}. Tentativas: " + "; ".join(errors)
    )


def _download(url: str, destination: Path, *, timeout: float) -> None:
    """Descarrega para um ficheiro temporário e só depois substitui o destino."""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        if getattr(response, "status", 200) != 200:
            raise OSError(f"HTTP {response.status}")
        handle, tmp_name = tempfile.mkstemp(dir=str(destination.parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(handle, "wb") as sink:
                shutil.copyfileobj(response, sink)
            if tmp.stat().st_size < _MIN_MODEL_BYTES:
                raise OSError(f"resposta demasiado pequena ({tmp.stat().st_size} bytes)")
            tmp.replace(destination)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


class SileroVad:
    """Envelope do Silero VAD em ONNX Runtime, compatível com as versões 4 e 5.

    As duas versões têm assinaturas de grafo diferentes — a v5 usa um único
    tensor ``state`` de forma ``(2, 1, 128)``, a v4 usa ``h`` e ``c`` separados
    com ``(2, 1, 64)``. A versão é deduzida dos nomes das entradas, pelo que o
    mesmo código serve ambas sem configuração.

    Args:
        model_path: Caminho do ``.onnx``. ``None`` resolve automaticamente.
        num_threads: Threads do ONNX Runtime. 1 é o correcto: o modelo é
            minúsculo e o paralelismo só acrescentaria latência de sincronização.
        provider: ``"cpu"`` ou ``"cuda"``.

    Raises:
        VadModelError: Se o ONNX Runtime faltar ou a sessão não abrir.
    """

    __slots__ = ("_input_name", "_names", "_session", "_sr", "_state", "_version")

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        num_threads: int = 1,
        provider: str = "cpu",
        allow_download: bool = True,
    ) -> None:
        if not HAS_ONNXRUNTIME:
            raise VadModelError(
                "onnxruntime não está instalado. Execute: pip install onnxruntime"
            )
        path = ensure_silero_model(model_path, allow_download=allow_download)

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, num_threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        try:
            self._session = ort.InferenceSession(
                str(path), sess_options=options, providers=providers
            )
        except Exception as exc:  # ORT lança tipos próprios não exportados
            raise VadModelError(f"falha ao abrir {path}: {exc}") from exc

        self._names = {entry.name for entry in self._session.get_inputs()}
        self._input_name = "input" if "input" in self._names else next(
            iter(entry.name for entry in self._session.get_inputs())
        )
        self._version = 5 if "state" in self._names else 4
        self._sr = np.array(TARGET_RATE, dtype=np.int64)
        self._state: dict[str, NDArray[np.float32]] = {}
        self.reset()
        LOG.info(
            "Silero VAD v%d carregado (%s, %s)",
            self._version,
            path.name,
            self._session.get_providers()[0],
        )

    @property
    def version(self) -> int:
        """4 ou 5, conforme o grafo carregado."""
        return self._version

    def reset(self) -> None:
        """Zera o estado recorrente. Chamar entre elocuções independentes."""
        if self._version == 5:
            self._state = {"state": np.zeros((2, 1, 128), dtype=np.float32)}
        else:
            self._state = {
                "h": np.zeros((2, 1, 64), dtype=np.float32),
                "c": np.zeros((2, 1, 64), dtype=np.float32),
            }

    def __call__(self, frame: NDArray[np.float32]) -> float:
        """Devolve a probabilidade de o frame conter voz.

        Args:
            frame: Exactamente 512 amostras float32 a 16 kHz, em [-1, 1].

        Returns:
            Probabilidade em [0, 1].

        Raises:
            ValueError: Se o frame não tiver 512 amostras.
        """
        if frame.size != FRAME_SAMPLES:
            raise ValueError(
                f"Silero VAD exige {FRAME_SAMPLES} amostras, recebeu {frame.size}"
            )
        feeds: dict[str, np.ndarray] = {
            self._input_name: frame.reshape(1, FRAME_SAMPLES)
        }
        if "sr" in self._names:
            feeds["sr"] = self._sr
        feeds.update(self._state)

        outputs = self._session.run(None, feeds)
        probability = float(outputs[0].item())
        if self._version == 5:
            self._state = {"state": outputs[1]}
        else:
            self._state = {"h": outputs[1], "c": outputs[2]}
        return probability


# ---------------------------------------------------------------------------
# Segmentação
# ---------------------------------------------------------------------------


class SpeechSegmenter:
    """Máquina de estados que converte frames em elocuções delimitadas.

    Args:
        vad: Modelo já carregado. ``None`` constrói um com a configuração dada.
        config: Limiares e temporizações.

    Example:
        >>> seg = SpeechSegmenter()                       # doctest: +SKIP
        >>> for event in seg.process(frame, t):           # doctest: +SKIP
        ...     handle(event)                             # doctest: +SKIP
    """

    __slots__ = (
        "_config",
        "_end_frames_needed",
        "_last_probability",
        "_phase",
        "_preroll",
        "_preroll_samples",
        "_speech_run",
        "_started_at",
        "_trailing_frames",
        "_utterance_frames",
        "_voiced_frames",
        "_vad",
        "stats",
    )

    def __init__(
        self,
        vad: SileroVad | None = None,
        config: VadConfig | None = None,
    ) -> None:
        self._config = config or VadConfig()
        self._vad = vad or SileroVad(
            self._config.model_path,
            num_threads=self._config.num_threads,
            provider=self._config.provider,
        )

        frame_seconds = FRAME_SAMPLES / TARGET_RATE
        self._end_frames_needed = max(
            1, int(round(self._config.trailing_ms / 1000.0 / frame_seconds))
        )
        # A capacidade tem um mínimo técnico (o ring buffer não pode ser vazio),
        # mas a quantidade efectivamente lida respeita ``preroll_ms``, de modo a
        # que 0 ms signifique mesmo zero pre-roll.
        self._preroll_samples = max(
            0, int(round(self._config.preroll_ms / 1000.0 * TARGET_RATE))
        )
        self._preroll = AudioRingBuffer(max(FRAME_SAMPLES, self._preroll_samples))

        self._phase = VadPhase.LISTENING
        self._speech_run = 0
        self._trailing_frames = 0
        self._utterance_frames = 0
        self._voiced_frames = 0
        self._started_at = 0.0
        self._last_probability = 0.0
        self.stats = SegmenterStats()

    # -- introspecção ----------------------------------------------------

    @property
    def phase(self) -> VadPhase:
        """Estado actual da FSM."""
        return self._phase

    @property
    def last_probability(self) -> float:
        """Probabilidade de voz do último frame processado."""
        return self._last_probability

    def reset(self) -> None:
        """Volta a LISTENING e limpa o estado recorrente e o pre-roll."""
        self._phase = VadPhase.LISTENING
        self._speech_run = 0
        self._trailing_frames = 0
        self._utterance_frames = 0
        self._voiced_frames = 0
        self._preroll.clear()
        self._vad.reset()

    # -- ciclo principal -------------------------------------------------

    def process(
        self, frame: NDArray[np.float32], captured_at: float
    ) -> Sequence[VadEvent]:
        """Consome um frame de 512 amostras e devolve os eventos resultantes.

        Args:
            frame: 512 amostras float32 a 16 kHz.
            captured_at: ``time.monotonic()`` do último sample do frame.

        Returns:
            Zero a dois eventos. Devolver uma sequência (e não um só evento)
            permite emitir SPEECH_END e, no mesmo frame, o SPEECH_START da
            elocução seguinte.
        """
        probability = self._vad(frame)
        self._last_probability = probability
        self.stats.frames += 1

        is_speech = probability >= self._config.start_probability
        still_speech = probability >= self._config.end_probability
        if is_speech:
            self.stats.speech_frames += 1

        if self._phase is VadPhase.LISTENING:
            return self._on_listening(frame, captured_at, probability, is_speech)
        if self._phase is VadPhase.SPEAKING:
            return self._on_speaking(frame, captured_at, probability, still_speech)
        return self._on_trailing(frame, captured_at, probability, is_speech)

    def _on_listening(
        self,
        frame: NDArray[np.float32],
        captured_at: float,
        probability: float,
        is_speech: bool,
    ) -> Sequence[VadEvent]:
        if not is_speech:
            self._speech_run = 0
            self._preroll.write(frame)
            return ()

        self._speech_run += 1
        if self._speech_run < self._config.start_frames:
            self._preroll.write(frame)
            return ()

        # O pre-roll é lido antes de o frame actual entrar no ring buffer, para
        # que o segmento fique com [pre-roll | frame] e sem amostras repetidas.
        preroll = self._preroll.read_last(self._preroll_samples)
        self._preroll.clear()
        self._phase = VadPhase.SPEAKING
        self._speech_run = 0
        self._trailing_frames = 0
        self._utterance_frames = 1
        self._voiced_frames = 1
        self._started_at = captured_at
        samples = np.concatenate((preroll, frame)) if preroll.size else frame.copy()
        return (
            VadEvent(
                kind=VadEventKind.SPEECH_START,
                samples=samples,
                captured_at=captured_at,
                probability=probability,
            ),
        )

    def _on_speaking(
        self,
        frame: NDArray[np.float32],
        captured_at: float,
        probability: float,
        still_speech: bool,
    ) -> Sequence[VadEvent]:
        self._utterance_frames += 1
        if still_speech:
            self._voiced_frames += 1
        else:
            self._phase = VadPhase.TRAILING
            self._trailing_frames = 1
        return (
            VadEvent(
                kind=VadEventKind.SPEECH_AUDIO,
                samples=frame.copy(),
                captured_at=captured_at,
                probability=probability,
            ),
        )

    def _on_trailing(
        self,
        frame: NDArray[np.float32],
        captured_at: float,
        probability: float,
        is_speech: bool,
    ) -> Sequence[VadEvent]:
        self._utterance_frames += 1
        # O áudio do trailing segue sempre para o STT: pode conter o fim da
        # frase, e é barato transcrevê-lo a mais.
        audio = VadEvent(
            kind=VadEventKind.SPEECH_AUDIO,
            samples=frame.copy(),
            captured_at=captured_at,
            probability=probability,
        )

        if is_speech:
            # Era só uma pausa para respirar, não um fim de frase.
            self._phase = VadPhase.SPEAKING
            self._trailing_frames = 0
            self._voiced_frames += 1
            return (audio,)

        self._trailing_frames += 1
        if self._trailing_frames < self._end_frames_needed:
            return (audio,)

        frame_seconds = FRAME_SAMPLES / TARGET_RATE
        duration = self._utterance_frames * frame_seconds
        voiced = self._voiced_frames * frame_seconds
        self._phase = VadPhase.LISTENING
        self._trailing_frames = 0
        self._utterance_frames = 0
        self._voiced_frames = 0
        self._speech_run = 0
        self._preroll.clear()
        self._vad.reset()

        if voiced * 1000.0 < self._config.min_speech_ms:
            self.stats.discarded += 1
            kind = VadEventKind.SPEECH_DISCARDED
        else:
            self.stats.utterances += 1
            kind = VadEventKind.SPEECH_END

        return (
            audio,
            VadEvent(
                kind=kind,
                samples=None,
                captured_at=captured_at,
                probability=probability,
                duration=duration,
                voiced_duration=voiced,
            ),
        )


def frame_rms(frame: NDArray[np.float32]) -> float:
    """RMS de um frame, para o medidor de nível da UI."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))
