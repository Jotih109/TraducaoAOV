# -*- coding: utf-8 -*-
"""Transcrição contínua com faster-whisper sobre CTranslate2, em janela deslizante.

Porque não se fatia o áudio em blocos
--------------------------------------

Cortar o áudio em blocos fixos e transcrever cada um isoladamente — o que a
versão anterior fazia a cada 6 s — falha por duas razões independentes: o corte
cai a meio de uma palavra, e o Whisper perde o contexto necessário para
pontuar e desambiguar. O resultado é latência alta *e* qualidade baixa.

A abordagem aqui é a de *LocalAgreement-2* (Macháček et al., 2023): a mesma
região de áudio é redecodificada repetidamente à medida que cresce, e uma
palavra só é dada como definitiva quando **duas inferências consecutivas
concordam** nela. Isso separa naturalmente dois fluxos de saída:

``committed``
    Palavras estáveis. Nunca mudam. Alimentam a tradução.

``interim``
    Cauda especulativa da hipótese mais recente. Aparece de imediato no overlay
    e pode ser reescrita no passo seguinte.

Assim o utilizador vê texto em aproximadamente 300 ms sem que a legenda esteja
constantemente a piscar e a corrigir-se.

Gestão da janela
----------------

Depois de confirmar palavras, o áudio à esquerda é cortado, retendo apenas
``context_seconds`` (1,5 a 2 s) de áudio já confirmado como contexto acústico.
A janela é portanto deslizante e limitada, mas o corte cai sempre numa fronteira
de palavra confirmada — nunca a meio de uma — e ``max_window_seconds`` é o tecto
absoluto para o caso patológico em que nada consegue ser confirmado.

Nota sobre ``condition_on_previous_text``
-----------------------------------------

Fica desligado de propósito. Em streaming, realimentar a transcrição anterior
faz o Whisper entrar em ciclos de repetição quando encontra silêncio ou ruído.
O contexto textual é passado como ``initial_prompt``, que influencia sem
realimentar.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Final, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from pipeline.config import TARGET_RATE, SttConfig
from pipeline.protocol import Stage, Status, Transcript
from pipeline.vad_engine import VadEvent, VadEventKind

__all__ = [
    "Backend",
    "HypothesisBuffer",
    "StepResult",
    "StreamingTranscriber",
    "TranscriberWorker",
    "TranscriberStats",
    "TranscriptionError",
    "Word",
    "ends_sentence",
    "select_backend",
]

LOG = logging.getLogger(__name__)

_STRONG_PUNCTUATION: Final[frozenset[str]] = frozenset(".?!…。？！")
_PUNCTUATION_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s]|_", re.UNICODE)
_MAX_NGRAM_OVERLAP: Final[int] = 5
_SHUTDOWN: Final[object] = object()

# Ordem de preferência de tipos de cómputo. Em GPUs anteriores a Turing
# (compute capability < 7.0, ex. GTX 10xx) o CTranslate2 não suporta float16 e
# a entrada correcta é int8_float32, que usa as instruções dp4a do Pascal.
_CUDA_PREFERENCE: Final[tuple[str, ...]] = (
    "float16",
    "int8_float16",
    "int8_float32",
    "int8",
    "float32",
)
_CPU_PREFERENCE: Final[tuple[str, ...]] = ("int8", "int8_float32", "float32")


class TranscriptionError(RuntimeError):
    """O modelo de STT não pôde ser carregado."""


@dataclass(frozen=True, slots=True)
class Word:
    """Palavra com marcação temporal absoluta na stream de captura."""

    text: str
    start: float
    end: float
    probability: float = 1.0

    def shifted(self, offset: float) -> Word:
        """Devolve a palavra deslocada no tempo por ``offset`` segundos."""
        return Word(self.text, self.start + offset, self.end + offset, self.probability)


@dataclass(frozen=True, slots=True)
class Backend:
    """Combinação de dispositivo e tipo de cómputo efectivamente utilizada."""

    device: str
    compute_type: str
    note: str = ""

    def __str__(self) -> str:
        suffix = f" ({self.note})" if self.note else ""
        return f"{self.device}/{self.compute_type}{suffix}"


@dataclass(slots=True)
class StepResult:
    """Resultado de uma iteração de transcrição."""

    delta: str
    """Texto acrescentado às palavras confirmadas nesta iteração."""

    committed: str
    """Todo o texto confirmado da elocução em curso."""

    interim: str
    """Cauda especulativa."""

    audio_end: float
    """Fim do áudio considerado, em segundos contados sobre o áudio entregue ao
    STT (o silêncio entre elocuções não conta)."""

    captured_at: float
    """``time.monotonic()`` do último sample considerado."""

    is_final: bool = False
    inference_ms: float = 0.0


def _normalize(text: str) -> str:
    """Forma canónica para comparar palavras entre hipóteses."""
    return _PUNCTUATION_RE.sub("", text).strip().casefold()


def _join(words: Iterable[Word]) -> str:
    """Reconstrói o texto. Os tokens do Whisper já trazem o espaço à cabeça."""
    return "".join(word.text for word in words).strip()


# ---------------------------------------------------------------------------
# Selecção de backend
# ---------------------------------------------------------------------------


def select_backend(
    device: str = "auto", compute_type: str = "auto"
) -> Backend:
    """Escolhe o melhor par (dispositivo, tipo de cómputo) para esta máquina.

    Interroga o CTranslate2 em vez de assumir: ``float16`` está disponível em
    GPUs Turing ou posteriores, mas não em Pascal, onde pedi-lo faz o modelo
    falhar ou cair silenciosamente para float32.

    Args:
        device: ``"auto"``, ``"cuda"`` ou ``"cpu"``.
        compute_type: Tipo explícito, ou ``"auto"``.

    Returns:
        O backend a usar. Nunca levanta excepção: na dúvida devolve CPU/int8.
    """
    try:
        import ctranslate2
    except ImportError:
        return Backend("cpu", "int8", "ctranslate2 em falta")

    def resolve(target: str, preference: Sequence[str]) -> Backend | None:
        try:
            supported = set(ctranslate2.get_supported_compute_types(target))
        except (RuntimeError, ValueError):
            return None
        if not supported:
            return None
        if compute_type != "auto":
            if compute_type in supported:
                return Backend(target, compute_type)
            LOG.warning(
                "%s não é suportado em %s (disponíveis: %s); a escolher automaticamente",
                compute_type,
                target,
                ", ".join(sorted(supported)),
            )
        for candidate in preference:
            if candidate in supported:
                note = (
                    "float16 indisponível nesta GPU (compute capability < 7.0)"
                    if target == "cuda" and candidate != "float16"
                    else ""
                )
                return Backend(target, candidate, note)
        return None

    if device in ("auto", "cuda"):
        try:
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except (RuntimeError, OSError):
            has_cuda = False
        if has_cuda:
            chosen = resolve("cuda", _CUDA_PREFERENCE)
            if chosen is not None:
                return chosen
        elif device == "cuda":
            LOG.warning("CUDA pedido mas indisponível; a usar CPU")

    return resolve("cpu", _CPU_PREFERENCE) or Backend("cpu", "int8")


def load_model(config: SttConfig) -> tuple[object, Backend]:
    """Carrega o modelo faster-whisper, com recuo para CPU se a GPU falhar.

    A GPU pode falhar tarde: o CTranslate2 só carrega as bibliotecas CUDA e
    cuDNN na construção do modelo, e a ausência delas manifesta-se como um erro
    de carregamento de biblioteca dinâmica, não como ausência de GPU.

    Raises:
        TranscriptionError: Se nem sequer a CPU funcionar.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper não está instalado. Execute: pip install faster-whisper"
        ) from exc

    backend = select_backend(config.device, config.compute_type)
    attempts: list[Backend] = [backend]
    if backend.device == "cuda":
        attempts.append(select_backend("cpu", "auto"))

    last: Exception | None = None
    for attempt in attempts:
        started = time.monotonic()
        try:
            model = WhisperModel(
                config.model,
                device=attempt.device,
                compute_type=attempt.compute_type,
                cpu_threads=config.cpu_threads,
                download_root=config.download_root,
            )
        except Exception as exc:  # o CTranslate2 lança RuntimeError genérico
            LOG.warning("falha ao carregar o modelo em %s: %s", attempt, exc)
            last = exc
            continue
        LOG.info(
            "faster-whisper %r carregado em %s (%.1f s)",
            config.model,
            attempt,
            time.monotonic() - started,
        )
        return model, attempt

    raise TranscriptionError(f"não foi possível carregar o modelo de STT: {last}")


# ---------------------------------------------------------------------------
# LocalAgreement
# ---------------------------------------------------------------------------


class HypothesisBuffer:
    """Confirma palavras por acordo entre duas hipóteses consecutivas.

    Uma palavra passa a definitiva quando aparece, na mesma posição, na hipótese
    actual e na anterior. O prefixo comum é confirmado e removido; o resto fica a
    aguardar a hipótese seguinte.

    Example:
        >>> buffer = HypothesisBuffer()
        >>> buffer.insert([Word(" olá", 0.0, 0.4), Word(" mun", 0.4, 0.6)])
        >>> buffer.flush()
        []
        >>> buffer.insert([Word(" olá", 0.0, 0.4), Word(" mundo", 0.4, 0.8)])
        >>> [word.text for word in buffer.flush()]
        [' olá']
    """

    __slots__ = ("_committed", "_last_end", "_pending", "_previous")

    def __init__(self) -> None:
        self._previous: list[Word] = []
        self._pending: list[Word] = []
        self._committed: list[Word] = []
        self._last_end = 0.0

    @property
    def committed(self) -> list[Word]:
        """Todas as palavras confirmadas desde o último :meth:`reset`."""
        return list(self._committed)

    @property
    def pending(self) -> list[Word]:
        """Cauda ainda não confirmada da última hipótese."""
        return list(self._pending)

    @property
    def last_committed_end(self) -> float:
        """Instante de fim da última palavra confirmada."""
        return self._last_end

    def reset(self) -> None:
        self._previous.clear()
        self._pending.clear()
        self._committed.clear()
        self._last_end = 0.0

    def insert(self, words: Sequence[Word]) -> None:
        """Regista a hipótese mais recente (com marcas temporais absolutas)."""
        # Descarta o que cai antes do que já foi confirmado. A janela retém
        # contexto à esquerda, logo o Whisper volta a transcrever essas palavras.
        fresh = [word for word in words if word.start > self._last_end - 0.1]
        fresh = self._strip_overlap(fresh)
        self._previous = self._pending
        self._pending = fresh

    def _strip_overlap(self, fresh: list[Word]) -> list[Word]:
        """Remove um n-grama inicial que repita a cauda já confirmada."""
        if not fresh or not self._committed:
            return fresh
        if abs(fresh[0].start - self._last_end) >= 1.0:
            return fresh
        limit = min(len(self._committed), len(fresh), _MAX_NGRAM_OVERLAP)
        for size in range(limit, 0, -1):
            tail = [_normalize(w.text) for w in self._committed[-size:]]
            head = [_normalize(w.text) for w in fresh[:size]]
            if tail == head:
                return fresh[size:]
        return fresh

    def flush(self) -> list[Word]:
        """Confirma e devolve o prefixo comum às duas últimas hipóteses."""
        confirmed: list[Word] = []
        index = 0
        limit = min(len(self._pending), len(self._previous))
        while index < limit:
            if _normalize(self._pending[index].text) != _normalize(
                self._previous[index].text
            ):
                break
            confirmed.append(self._pending[index])
            index += 1

        if confirmed:
            self._last_end = confirmed[-1].end
            self._committed.extend(confirmed)
            del self._pending[:index]
            del self._previous[:index]
        return confirmed

    def flush_all(self) -> list[Word]:
        """Confirma tudo o que resta. Usado no EOS, onde não há hipótese futura."""
        confirmed = self._pending
        if confirmed:
            self._last_end = confirmed[-1].end
            self._committed.extend(confirmed)
        self._pending = []
        self._previous = []
        return confirmed


# ---------------------------------------------------------------------------
# Transcrição em janela deslizante
# ---------------------------------------------------------------------------


class StreamingTranscriber:
    """Mantém a janela de áudio e produz texto confirmado e especulativo.

    Args:
        model: Instância de ``faster_whisper.WhisperModel``.
        config: Parâmetros da janela e da descodificação.
        language: Código ISO da língua falada, ou ``None`` para detectar.

    Note:
        Não é thread-safe. Destina-se a ser usada por :class:`TranscriberWorker`
        a partir de uma única thread.
    """

    def __init__(
        self,
        model: object,
        config: SttConfig | None = None,
        language: str | None = None,
    ) -> None:
        self._model = model
        self._config = config or SttConfig()
        self._language = language if language is not None else self._config.language

        self._audio: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._offset = 0.0
        self._stream_end = 0.0
        self._captured_at = 0.0
        self._samples_at_last_inference = 0
        self._last_inference_at = 0.0

        self._buffer = HypothesisBuffer()
        self._utterance: list[Word] = []
        self._prompt_tail = ""
        self._ceiling_warned = False

    # -- estado ----------------------------------------------------------

    @property
    def window_seconds(self) -> float:
        """Duração do áudio actualmente na janela."""
        return self._audio.size / TARGET_RATE

    @property
    def unseen_seconds(self) -> float:
        """Áudio ainda não visto por nenhuma inferência."""
        return (self._audio.size - self._samples_at_last_inference) / TARGET_RATE

    @property
    def committed_text(self) -> str:
        """Texto confirmado da elocução em curso."""
        return _join(self._utterance)

    def prompt(self) -> str:
        """Contexto textual para o ``initial_prompt`` da próxima inferência.

        Junta o texto de elocuções anteriores com as palavras desta que já
        saíram pela esquerda da janela. As que ainda estão na janela ficam de
        fora: o modelo vai voltar a ouvi-las, e prometer-lhe no prompt o que ele
        próprio está prestes a transcrever enviesa-o para se repetir.
        """
        scrolled = _join(word for word in self._utterance if word.end <= self._offset)
        text = f"{self._prompt_tail} {scrolled}".strip()
        return text[-self._config.prompt_chars :]

    def reset(self, *, keep_prompt: bool = True) -> None:
        """Limpa a janela e a hipótese, preparando a próxima elocução."""
        if keep_prompt:
            text = self.committed_text
            if text:
                self._prompt_tail = (self._prompt_tail + " " + text).strip()[
                    -self._config.prompt_chars :
                ]
        else:
            self._prompt_tail = ""
        self._audio = np.zeros(0, dtype=np.float32)
        self._offset = self._stream_end
        self._samples_at_last_inference = 0
        self._buffer.reset()
        self._utterance = []

    # -- entrada ---------------------------------------------------------

    def append(self, samples: NDArray[np.float32], captured_at: float) -> None:
        """Acrescenta áudio à janela."""
        if samples.size == 0:
            return
        self._audio = (
            samples.astype(np.float32, copy=True)
            if self._audio.size == 0
            else np.concatenate((self._audio, samples))
        )
        self._stream_end += samples.size / TARGET_RATE
        self._captured_at = captured_at

    def ready(self, now: float) -> bool:
        """Indica se compensa correr nova inferência.

        Duas condições: chegou áudio novo suficiente para valer a pena, e
        passou o intervalo mínimo. A segunda é o mecanismo de auto-regulação —
        numa máquina lenta as inferências espaçam-se sozinhas em vez de
        acumularem atraso.
        """
        if self._audio.size == 0:
            return False
        if self.unseen_seconds < self._config.min_chunk_seconds:
            return False
        return now - self._last_inference_at >= self._config.min_infer_interval

    # -- inferência ------------------------------------------------------

    def step(self, now: float | None = None) -> StepResult | None:
        """Corre uma inferência e confirma o que houver a confirmar.

        Args:
            now: Instante a registar como o desta inferência, na mesma escala de
                tempo passada a :meth:`ready`. Omitir usa ``time.monotonic()``.
        """
        self._last_inference_at = time.monotonic() if now is None else now
        words, elapsed = self._infer()
        if words is None:
            return None
        self._buffer.insert(words)
        confirmed = self._buffer.flush()
        return self._build_result(confirmed, elapsed, is_final=False)

    def finalize(self) -> StepResult:
        """Fecha a elocução no EOS, confirmando toda a hipótese pendente.

        Corre uma última inferência sobre o áudio que ainda não foi visto — sem
        ela, a cauda da frase perder-se-ia. Depois confirma tudo, porque já não
        haverá hipótese futura com que comparar.
        """
        elapsed = 0.0
        confirmed: list[Word] = []
        if self._audio.size and self.unseen_seconds > 0.0:
            words, elapsed = self._infer()
            if words is not None:
                self._buffer.insert(words)
                # O que esta última passagem confirma tem de entrar no
                # resultado: descartá-lo perderia palavras em cada EOS.
                confirmed.extend(self._buffer.flush())
        confirmed.extend(self._buffer.flush_all())
        result = self._build_result(confirmed, elapsed, is_final=True)
        self.reset(keep_prompt=True)
        return result

    def _infer(self) -> tuple[list[Word] | None, float]:
        """Descodifica a janela inteira. Devolve palavras em tempo absoluto."""
        audio = self._audio
        if audio.size == 0:
            return None, 0.0

        started = time.monotonic()
        self._samples_at_last_inference = audio.size
        try:
            segments, _info = self._model.transcribe(  # type: ignore[attr-defined]
                audio,
                language=self._language,
                task="transcribe",
                beam_size=self._config.beam_size,
                best_of=1,
                # Um único valor de temperatura desliga a cascata de repetições
                # do Whisper, que provocaria picos de latência imprevisíveis.
                temperature=0.0,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=self.prompt() or None,
                vad_filter=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            words = [
                Word(
                    text=word.word,
                    start=self._offset + float(word.start),
                    end=self._offset + float(word.end),
                    probability=float(getattr(word, "probability", 1.0)),
                )
                for segment in segments
                if getattr(segment, "no_speech_prob", 0.0) <= 0.9
                for word in (segment.words or ())
            ]
        except Exception:
            LOG.exception("inferência de STT falhou")
            return None, (time.monotonic() - started) * 1000.0
        return words, (time.monotonic() - started) * 1000.0

    def _build_result(
        self, confirmed: Sequence[Word], elapsed_ms: float, *, is_final: bool
    ) -> StepResult:
        if confirmed:
            self._ceiling_warned = False
            self._utterance.extend(confirmed)
            self._trim(confirmed[-1].end)
        # Fora do ``if``: quando nada é confirmado é justamente quando o tecto
        # tem de agir, porque é aí que a janela cresce sem ninguém a cortar.
        self._enforce_ceiling()
        return StepResult(
            delta=_join(confirmed),
            committed=_join(self._utterance),
            interim=_join(self._buffer.pending),
            audio_end=self._stream_end,
            captured_at=self._captured_at,
            is_final=is_final,
            inference_ms=elapsed_ms,
        )

    def _trim(self, committed_end: float) -> None:
        """Encolhe a janela, retendo ``context_seconds`` de contexto confirmado."""
        cut_time = committed_end - self._config.context_seconds
        if cut_time <= self._offset:
            return
        cut = int((cut_time - self._offset) * TARGET_RATE)
        if cut <= 0 or cut >= self._audio.size:
            return
        self._audio = self._audio[cut:].copy()
        self._offset = cut_time
        self._samples_at_last_inference = max(
            0, self._samples_at_last_inference - cut
        )

    def _enforce_ceiling(self) -> None:
        """Rede de segurança: nada confirmado e a janela a crescer sem parar.

        Acontece com ruído contínuo ou música. Corta à força e reinicia a
        hipótese, aceitando perder o contexto em troca de latência limitada.
        """
        ceiling = int(self._config.max_window_seconds * TARGET_RATE)
        if self._audio.size <= ceiling:
            return
        excess = self._audio.size - ceiling
        if not self._ceiling_warned:
            # Uma linha por episódio, não por inferência: em ruído contínuo
            # isto dispara a cada 250 ms e afogaria o log.
            self._ceiling_warned = True
            LOG.warning(
                "janela no tecto de %.1f s sem confirmações (ruído ou música?); "
                "a cortar à força",
                self._config.max_window_seconds,
            )
        self._audio = self._audio[excess:].copy()
        self._offset += excess / TARGET_RATE
        self._samples_at_last_inference = max(
            0, self._samples_at_last_inference - excess
        )
        self._buffer.reset()


def ends_sentence(text: str) -> bool:
    """Indica se o texto termina em pontuação forte (fronteira de frase)."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _STRONG_PUNCTUATION


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TranscriberStats:
    """Contadores de diagnóstico do worker de STT."""

    inferences: int = 0
    utterances: int = 0
    total_inference_ms: float = 0.0
    max_inference_ms: float = 0.0

    @property
    def mean_inference_ms(self) -> float:
        return self.total_inference_ms / self.inferences if self.inferences else 0.0


class TranscriberWorker(threading.Thread):
    """Thread que liga os eventos do VAD ao :class:`StreamingTranscriber`.

    Corre numa thread própria porque o CTranslate2 liberta o GIL durante a
    descodificação: enquanto o modelo trabalha, a thread de captura continua a
    ler o áudio sem perder frames.

    Args:
        events: Fila alimentada pelo segmentador de VAD.
        sink: Recebe cada :class:`~pipeline.protocol.Transcript` produzido.
        config: Parâmetros de STT.
        language: Língua falada.
        on_status: Notificações de estado para a UI.
    """

    def __init__(
        self,
        events: queue.Queue[VadEvent | object],
        sink: Callable[[Transcript], None],
        config: SttConfig | None = None,
        *,
        language: str | None = None,
        on_status: Callable[[Status], None] | None = None,
        name: str = "stt-worker",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._events = events
        self._sink = sink
        self._config = config or SttConfig()
        self._language = language
        self._on_status = on_status or (lambda _: None)

        self._shutdown = threading.Event()
        self._transcriber: StreamingTranscriber | None = None
        self._backend: Backend | None = None
        self._sequence = 0
        self._revision = 0
        self._active = False
        self._pending_eos = False
        self.stats = TranscriberStats()

    @property
    def backend(self) -> Backend | None:
        """Backend em uso, disponível depois de o modelo carregar."""
        return self._backend

    def stop(self) -> None:
        """Pede a paragem ordeira da thread."""
        self._shutdown.set()
        try:
            self._events.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            model, backend = load_model(self._config)
        except TranscriptionError as exc:
            LOG.error("%s", exc)
            self._on_status(Status(Stage.STT, "Modelo de STT indisponível", str(exc)))
            return

        self._backend = backend
        self._transcriber = StreamingTranscriber(model, self._config, self._language)
        self._on_status(Status(Stage.STT, f"STT pronto ({backend})"))

        while not self._shutdown.is_set():
            if self._consume() is _SHUTDOWN:
                break
            self._maybe_infer()

    # -- ciclo interno ---------------------------------------------------

    def _consume(self) -> object | None:
        """Esvazia a fila de eventos sem bloquear a inferência.

        Drenar tudo o que já está na fila antes de inferir é o que impede o
        áudio de se acumular atrás de uma inferência lenta.
        """
        assert self._transcriber is not None
        first = True
        while True:
            try:
                event = (
                    self._events.get(timeout=0.02)
                    if first
                    else self._events.get_nowait()
                )
            except queue.Empty:
                return None
            first = False
            if event is _SHUTDOWN:
                return _SHUTDOWN
            if isinstance(event, VadEvent):
                self._apply_event(event)

    def _apply_event(self, event: VadEvent) -> None:
        assert self._transcriber is not None
        if event.kind is VadEventKind.SPEECH_START:
            self._sequence += 1
            self._revision = 0
            self._active = True
            self._pending_eos = False
            if event.samples is not None:
                self._transcriber.append(event.samples, event.captured_at)
        elif event.kind is VadEventKind.SPEECH_AUDIO:
            if self._active and event.samples is not None:
                self._transcriber.append(event.samples, event.captured_at)
        elif event.kind is VadEventKind.SPEECH_END:
            self._pending_eos = True
        elif event.kind is VadEventKind.SPEECH_DISCARDED:
            self._transcriber.reset(keep_prompt=False)
            self._active = False
            self._pending_eos = False

    def _maybe_infer(self) -> None:
        assert self._transcriber is not None
        if not self._active:
            return

        if self._pending_eos:
            self._pending_eos = False
            self._active = False
            result = self._transcriber.finalize()
            self.stats.utterances += 1
            if result is not None:
                self._record(result)
                self._emit(result)
            return

        now = time.monotonic()
        if not self._transcriber.ready(now):
            return
        result = self._transcriber.step(now)
        if result is None:
            return
        self._record(result)
        # Só vale a pena acordar a jusante se algo mudou de facto.
        if result.delta or result.interim:
            self._emit(result)

    def _record(self, result: StepResult) -> None:
        self.stats.inferences += 1
        self.stats.total_inference_ms += result.inference_ms
        self.stats.max_inference_ms = max(
            self.stats.max_inference_ms, result.inference_ms
        )

    def _emit(self, result: StepResult) -> None:
        self._revision += 1
        self._sink(
            Transcript(
                seq=self._sequence,
                revision=self._revision,
                committed=result.committed,
                interim=result.interim,
                delta=result.delta,
                is_final=result.is_final,
                audio_end=result.audio_end,
                captured_at=result.captured_at,
            )
        )
