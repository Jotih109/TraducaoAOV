# -*- coding: utf-8 -*-
r"""Mensagens IPC trocadas entre os processos do pipeline.

Todas as mensagens são ``dataclass`` imutáveis com ``slots``: baratas de criar,
picklam de forma eficiente entre processos e são seguras para partilhar entre
threads (nunca são mutadas depois de construídas).

Topologia (ver :mod:`pipeline.main`)::

    [engine]  --(Transcript)-->  [translator]  --(Translation)--> [ui]
        \-------------------(Status/Level/Fatal)------------------/
    [ui]      --(Control)---->   [engine]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Literal

__all__ = [
    "Control",
    "Fatal",
    "Level",
    "Message",
    "SENTINEL",
    "Status",
    "Stage",
    "Transcript",
    "Translation",
    "VadPhase",
]


class Stage(str, Enum):
    """Origem de uma mensagem de estado — usada apenas para log/diagnóstico."""

    AUDIO = "audio"
    VAD = "vad"
    STT = "stt"
    MT = "mt"
    UI = "ui"


class VadPhase(str, Enum):
    """Estados da máquina de estados finita do VAD."""

    LISTENING = "listening"
    SPEAKING = "speaking"
    TRAILING = "trailing"


@dataclass(frozen=True, slots=True)
class Transcript:
    """Saída do motor STT.

    O par ``(committed, interim)`` descreve a legenda completa da elocução em
    curso: ``committed`` já não muda mais (foi confirmado por LocalAgreement ou
    por um evento de fim-de-fala), ``interim`` é especulativo e pode ser
    reescrito na próxima iteração.

    Attributes:
        seq: Identificador monotónico da elocução (incrementa a cada EOS).
        revision: Contador de revisões dentro da mesma elocução.
        committed: Texto já fixado desta elocução.
        interim: Cauda especulativa, ainda sujeita a alteração.
        delta: Apenas o texto acrescentado a ``committed`` nesta revisão.
              É isto (e não ``committed``) que deve alimentar a tradução, para
              não retraduzir o que já foi traduzido.
        is_final: ``True`` quando o VAD emitiu EOS e a elocução fechou.
        audio_end: Posição, em segundos, do fim do áudio considerado, medida
            desde o início da stream de captura.
        captured_at: ``time.monotonic()`` do instante em que o último sample
            desta revisão foi capturado. Base para medir latência ponta-a-ponta.
        emitted_at: ``time.monotonic()`` do instante em que o STT terminou.
    """

    seq: int
    revision: int
    committed: str
    interim: str
    delta: str
    is_final: bool
    audio_end: float
    captured_at: float
    emitted_at: float = field(default_factory=time.monotonic)

    @property
    def text(self) -> str:
        """Legenda completa da elocução (confirmada + especulativa)."""
        if self.committed and self.interim:
            return f"{self.committed} {self.interim}"
        return self.committed or self.interim

    @property
    def stt_latency_ms(self) -> float:
        """Milissegundos entre a captura do último sample e o fim da inferência."""
        return (self.emitted_at - self.captured_at) * 1000.0


@dataclass(frozen=True, slots=True)
class Translation:
    """Saída do motor de tradução, espelhando a estrutura de :class:`Transcript`."""

    seq: int
    revision: int
    source_text: str
    committed: str
    interim: str
    is_final: bool
    backend: str
    captured_at: float
    emitted_at: float = field(default_factory=time.monotonic)

    @property
    def text(self) -> str:
        if self.committed and self.interim:
            return f"{self.committed} {self.interim}"
        return self.committed or self.interim

    @property
    def end_to_end_ms(self) -> float:
        """Latência total: do sample capturado até à tradução pronta."""
        return (self.emitted_at - self.captured_at) * 1000.0


@dataclass(frozen=True, slots=True)
class Level:
    """Telemetria por frame para o medidor da UI (emitida com decimação)."""

    rms: float
    speech_probability: float
    phase: VadPhase


@dataclass(frozen=True, slots=True)
class Status:
    """Mudança de estado não fatal (dispositivo trocado, modelo carregado, ...)."""

    stage: Stage
    text: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Fatal:
    """Falha irrecuperável num processo de trabalho."""

    stage: Stage
    text: str
    traceback: str = ""


@dataclass(frozen=True, slots=True)
class Control:
    """Comando da UI para o processo de engine."""

    action: Literal["start", "stop", "set_languages", "shutdown"]
    source_lang: str = ""
    target_lang: str = ""


Message = Transcript | Translation | Level | Status | Fatal | Control
"""União de tudo o que pode circular nas filas IPC."""

SENTINEL: Final[None] = None
"""Valor colocado numa fila para terminar o consumidor de forma ordeira."""
