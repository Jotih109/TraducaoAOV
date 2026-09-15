# -*- coding: utf-8 -*-
"""Configuração tipada do pipeline, com persistência em JSON.

Um único ``pipeline.json`` na raiz do projecto controla todos os módulos. Chaves
desconhecidas são ignoradas silenciosamente (permite downgrade sem partir o
ficheiro) e chaves em falta caem no valor por omissão declarado aqui.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

__all__ = [
    "AudioConfig",
    "CONFIG_PATH",
    "FRAME_SAMPLES",
    "MODELS_DIR",
    "MtConfig",
    "OverlayConfig",
    "PROJECT_ROOT",
    "PipelineConfig",
    "SttConfig",
    "TARGET_RATE",
    "VadConfig",
]

LOG = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CONFIG_PATH: Final[Path] = PROJECT_ROOT / "pipeline.json"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

TARGET_RATE: Final[int] = 16_000
"""Taxa de amostragem interna. Fixa: Silero VAD e Whisper só aceitam 16 kHz."""

FRAME_SAMPLES: Final[int] = 512
"""Amostras por frame = 32 ms @ 16 kHz. Silero VAD v5 exige exactamente 512."""

FRAME_SECONDS: Final[float] = FRAME_SAMPLES / TARGET_RATE

_T = TypeVar("_T")


@dataclass(slots=True)
class AudioConfig:
    """Captura WASAPI loopback.

    Attributes:
        device_index: Índice PortAudio de um dispositivo de loopback específico.
            ``None`` (recomendado) segue o dispositivo de saída padrão do Windows
            e acompanha-o quando o utilizador o troca.
        read_ms: Tamanho do bloco lido ao PortAudio. 20 ms é o mínimo seguro em
            WASAPI shared mode (o período nativo ronda os 10 ms); descer daqui
            troca latência por risco de overrun.
        silence_probe_seconds: Silêncio digital contínuo a partir do qual se
            suspeita que o dispositivo padrão mudou e se re-sonda o sistema.
        restart_backoff_seconds: Espera inicial entre tentativas de reabrir a
            stream (cresce exponencialmente até 5 s).
        resampler_taps_per_phase: Comprimento do FIR polifásico por fase.
    """

    device_index: int | None = None
    read_ms: float = 20.0
    silence_probe_seconds: float = 8.0
    silence_floor: float = 1e-5
    restart_backoff_seconds: float = 0.5
    resampler_taps_per_phase: int = 20


@dataclass(slots=True)
class VadConfig:
    """Silero VAD (ONNX Runtime) e a máquina de estados de segmentação.

    Attributes:
        start_probability: Probabilidade acima da qual o frame conta como voz.
        end_probability: Limiar de saída, mais baixo que ``start_probability``
            para criar histerese e não picar em consoantes surdas.
        start_frames: Frames consecutivos de voz para transitar LISTENING para
            SPEAKING. 1 = resposta imediata (spec); 2 filtra estalidos.
        trailing_ms: Silêncio tolerado em TRAILING antes de emitir EOS.
        preroll_ms: Áudio anterior ao início da fala guardado em ring buffer e
            prefixado ao segmento, para não decapitar a primeira sílaba.
        min_speech_ms: Elocuções mais curtas do que isto são descartadas.
    """

    model_path: str | None = None
    start_probability: float = 0.5
    end_probability: float = 0.35
    start_frames: int = 1
    trailing_ms: float = 300.0
    preroll_ms: float = 200.0
    min_speech_ms: float = 120.0
    num_threads: int = 1
    provider: Literal["cpu", "cuda"] = "cpu"


@dataclass(slots=True)
class SttConfig:
    """faster-whisper sobre CTranslate2, em janela deslizante.

    Attributes:
        model: Tamanho ou caminho local ("tiny", "base", "small", "medium").
        device: "auto", "cuda" ou "cpu".
        compute_type: "auto" escolhe o melhor tipo suportado pelo hardware.
            Em GPUs Pascal (compute capability 6.1, ex. GTX 10xx) o float16
            NÃO é suportado pelo CTranslate2 e int8_float32 é o correcto.
        language: Sobreposição da língua falada. ``None`` (o normal) faz seguir
            ``source_lang``, evitando que os dois valores divirjam. Use
            ``"auto"`` para detecção automática, que custa uma passagem extra
            por elocução.
        context_seconds: Áudio já confirmado retido à esquerda da janela como
            contexto acústico. É o "sliding window" de 1.5 a 2 s.
        max_window_seconds: Tecto absoluto da janela; evita crescimento sem fim
            quando nada consegue ser confirmado.
        min_chunk_seconds: Áudio novo necessário para disparar nova inferência.
        min_infer_interval: Intervalo mínimo entre inferências (auto-regulação).
        prompt_chars: Caracteres de texto confirmado passados como
            ``initial_prompt``. Substitui ``condition_on_previous_text``, que
            provoca ciclos de alucinação em streaming.
    """

    model: str = "small"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "auto"
    language: str | None = None
    context_seconds: float = 2.0
    max_window_seconds: float = 15.0
    min_chunk_seconds: float = 0.4
    min_infer_interval: float = 0.25
    beam_size: int = 1
    prompt_chars: int = 200
    cpu_threads: int = 0
    download_root: str | None = None


@dataclass(slots=True)
class MtConfig:
    """Tradução: CTranslate2 local (MarianMT int8) ou serviço remoto.

    Attributes:
        backend: "auto" usa o modelo local se existir, senão cai para cloud.
        model_dir: Directório de um modelo Marian já convertido para CTranslate2
            (deve conter ``model.bin``, ``source.spm`` e ``target.spm``).
        target_token: Token de língua-alvo exigido por modelos multi-alvo como
            ``opus-mt-en-ROMANCE`` (ex.: ">>por<<"). Vazio = inferido do nome do
            modelo e da língua-alvo.
        interim_debounce_ms: Intervalo mínimo entre traduções de texto
            especulativo. Protege a GPU/rede de retraduzir 20 vezes por segundo.
    """

    backend: Literal["auto", "local", "cloud", "off"] = "auto"
    model_dir: str | None = None
    device: Literal["auto", "cuda", "cpu"] = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    target_token: str = ""
    max_decoding_length: int = 256
    interim_debounce_ms: float = 220.0
    cache_size: int = 4096
    api: Literal["google", "deepl", "openai"] = "google"
    api_key: str = ""
    request_timeout: float = 2.5


@dataclass(slots=True)
class OverlayConfig:
    """HUD flutuante."""

    font_size: int = 26
    opacity: int = 88
    bg_color: str = "#0b1120"
    text_color: str = "#38bdf8"
    source_color: str = "#94a3b8"
    interim_color: str = "#64748b"
    show_source: bool = True
    click_through: bool = False
    width: int = 900
    height: int = 170
    pos_x: int | None = None
    pos_y: int | None = None
    autohide_seconds: float = 8.0
    refresh_ms: int = 33
    show_metrics: bool = True


@dataclass(slots=True)
class PipelineConfig:
    """Raiz da configuração."""

    source_lang: str = "en"
    target_lang: str = "pt"
    log_level: str = "INFO"
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    mt: MtConfig = field(default_factory=MtConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> PipelineConfig:
        """Carrega a configuração, tolerando ficheiro ausente ou corrompido."""
        target = Path(path) if path is not None else CONFIG_PATH
        if not target.exists():
            cfg = cls()
            cfg._seed_from_legacy()
            return cfg
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOG.exception("pipeline.json ilegível; a usar valores por omissão")
            return cls()
        if not isinstance(raw, dict):
            LOG.error("pipeline.json não contém um objecto JSON; ignorado")
            return cls()
        return _build_root(raw)

    def save(self, path: Path | str | None = None) -> None:
        """Escreve a configuração de forma atómica (write + replace)."""
        target = Path(path) if path is not None else CONFIG_PATH
        tmp = target.with_name(target.name + ".tmp")
        try:
            tmp.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(target)
        except OSError:
            LOG.exception("não foi possível gravar %s", target)
            tmp.unlink(missing_ok=True)

    def _seed_from_legacy(self) -> None:
        """Importa as chaves reaproveitáveis do ``settings.json`` da v1."""
        legacy = PROJECT_ROOT / "settings.json"
        if not legacy.exists():
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self.source_lang = str(data.get("source_lang", self.source_lang))
        self.target_lang = str(data.get("target_lang", self.target_lang))
        legacy_overlay = {
            "overlay_font_size": "font_size",
            "overlay_opacity": "opacity",
            "overlay_bg_color": "bg_color",
            "overlay_text_color": "text_color",
            "overlay_pt_color": "source_color",
            "overlay_show_original": "show_source",
            "overlay_click_through": "click_through",
            "overlay_width": "width",
            "overlay_height": "height",
            "overlay_pos_x": "pos_x",
            "overlay_pos_y": "pos_y",
            "overlay_autohide_seconds": "autohide_seconds",
        }
        for old_key, new_key in legacy_overlay.items():
            if old_key in data:
                setattr(self.overlay, new_key, data[old_key])
        LOG.info("configuração inicial migrada de settings.json")


# ``fields(cls)[i].type`` devolve strings porque ``from __future__ import
# annotations`` está activo, por isso o mapa de secções aninhadas é explícito.
_SECTIONS: Final[dict[str, type]] = {
    "audio": AudioConfig,
    "vad": VadConfig,
    "stt": SttConfig,
    "mt": MtConfig,
    "overlay": OverlayConfig,
}


def _build_section(cls: type[_T], raw: dict[str, Any]) -> _T:
    """Instancia uma dataclass a partir de um dict, ignorando chaves estranhas."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kwargs = {key: value for key, value in raw.items() if key in known}
    try:
        return cls(**kwargs)  # type: ignore[call-arg]
    except TypeError:
        LOG.exception("secção %s inválida; a usar valores por omissão", cls.__name__)
        return cls()  # type: ignore[call-arg]


def _build_root(raw: dict[str, Any]) -> PipelineConfig:
    kwargs: dict[str, Any] = {}
    for f in fields(PipelineConfig):
        if f.name not in raw:
            continue
        value = raw[f.name]
        section = _SECTIONS.get(f.name)
        if section is None:
            kwargs[f.name] = value
        elif isinstance(value, dict):
            kwargs[f.name] = _build_section(section, value)
    try:
        return PipelineConfig(**kwargs)
    except TypeError:
        LOG.exception("pipeline.json inválido; a usar valores por omissão")
        return PipelineConfig()
