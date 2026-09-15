# -*- coding: utf-8 -*-
"""Coordenador de processos do pipeline de tradução em tempo real.

Topologia::

    processo "engine"                processo "translator"      processo principal
    ┌──────────────────────────┐     ┌───────────────────┐      ┌──────────────┐
    │ thread captura (WASAPI)  │     │ MarianMT int8     │      │ Qt + overlay │
    │        ↓ frames 32 ms    │     │   ou serviço web  │      │              │
    │ VAD Silero + FSM         │     └───────────────────┘      └──────────────┘
    │        ↓ eventos            ▲             │                      ▲
    │ thread STT (CTranslate2) ───┘ transcritos │ traduções            │
    └──────────────────────────┘                └──────────────────────┘
              │                                                        ▲
              └──────────────── estado e transcrições ─────────────────┘

Porquê três processos e não três threads: o CTranslate2 liberta o GIL durante a
inferência, mas o Qt não, e a tradução pode ficar pendurada num pedido de rede.
Separar garante que nenhum dos três consegue atrasar os outros.

Uso::

    python main.py                 # arranca o pipeline completo
    python main.py --check         # diagnostica o ambiente e sai
    python main.py --list-devices  # lista os endpoints de loopback WASAPI
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import queue
import signal
import sys
import threading
from typing import Final, Sequence

from pipeline.config import PipelineConfig
from pipeline.protocol import Control, Fatal, Level, Stage, Status, Transcript

__all__ = ["main", "run_engine_process"]

LOG = logging.getLogger("main")

_LEVEL_DECIMATION: Final[int] = 4
"""Só 1 em cada 4 frames alimenta o medidor da UI (cerca de 8 Hz)."""

_STT_QUEUE_SIZE: Final[int] = 256
_UI_QUEUE_SIZE: Final[int] = 512
_VAD_QUEUE_SIZE: Final[int] = 4096


def _speech_language(config: PipelineConfig) -> str | None:
    """Língua a passar ao Whisper.

    ``stt.language`` é uma sobreposição: vazio segue ``source_lang``, e "auto"
    entrega ``None`` ao modelo, que então detecta a língua sozinho.
    """
    override = (config.stt.language or "").strip()
    if not override:
        return config.source_lang
    return None if override.lower() == "auto" else override


def _offer(sink: "mp.Queue[object]", message: object) -> None:
    """Publica sem bloquear; um consumidor lento nunca trava o produtor."""
    try:
        sink.put_nowait(message)
    except (queue.Full, ValueError, OSError):
        pass


# ---------------------------------------------------------------------------
# Processo de engine
# ---------------------------------------------------------------------------


def run_engine_process(
    config: PipelineConfig,
    stt_outbox: "mp.Queue[object]",
    ui_outbox: "mp.Queue[object]",
    control: "mp.Queue[object]",
) -> None:
    """Captura, detecta voz e transcreve, até receber ordem de paragem."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [engine] %(message)s",
    )
    # Importações adiadas para depois do fork/spawn: carregar o ONNX Runtime e o
    # CTranslate2 no processo pai duplicaria centenas de MB por processo filho.
    from pipeline.audio_capture import LoopbackCapture
    from pipeline.transcriber import TranscriberWorker
    from pipeline.vad_engine import SpeechSegmenter, VadModelError, frame_rms

    try:
        segmenter = SpeechSegmenter(config=config.vad)
    except VadModelError as exc:
        LOG.error("VAD indisponível: %s", exc)
        _offer(ui_outbox, Fatal(Stage.VAD, "VAD indisponível", str(exc)))
        return

    events: queue.Queue[object] = queue.Queue(maxsize=_VAD_QUEUE_SIZE)

    def publish(transcript: Transcript) -> None:
        _offer(stt_outbox, transcript)
        _offer(ui_outbox, transcript)

    worker = TranscriberWorker(
        events,
        publish,
        config.stt,
        language=_speech_language(config),
        on_status=lambda status: _offer(ui_outbox, status),
    )
    worker.start()

    counter = 0

    def on_frame(frame, captured_at: float) -> None:
        nonlocal counter
        counter += 1
        for event in segmenter.process(frame, captured_at):
            try:
                events.put_nowait(event)
            except queue.Full:
                # A fila só enche se o STT não acompanhar o tempo real. Perder
                # um frame é preferível a bloquear a thread de captura, o que
                # provocaria um overrun no PortAudio e perderia muito mais.
                LOG.warning("fila do STT cheia; frame descartado")
        if counter % _LEVEL_DECIMATION == 0:
            _offer(
                ui_outbox,
                Level(frame_rms(frame), segmenter.last_probability, segmenter.phase),
            )

    capture = LoopbackCapture(
        on_frame,
        config.audio,
        on_status=lambda text: _offer(ui_outbox, Status(Stage.AUDIO, text)),
    )
    capture.start()
    _offer(ui_outbox, Status(Stage.AUDIO, "À escuta do áudio do sistema"))

    try:
        while True:
            try:
                command = control.get(timeout=0.5)
            except queue.Empty:
                if not capture.running:
                    LOG.error("a thread de captura morreu; a terminar")
                    break
                continue
            except (EOFError, OSError):
                break
            if isinstance(command, Control) and command.action == "shutdown":
                break
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        capture.stop()
        worker.stop()
        worker.join(timeout=5.0)
        _offer(stt_outbox, None)
        LOG.info(
            "engine terminada: %d frames, %d elocuções, %d inferências "
            "(média %.0f ms, máx %.0f ms)",
            capture.stats.frames_emitted,
            segmenter.stats.utterances,
            worker.stats.inferences,
            worker.stats.mean_inference_ms,
            worker.stats.max_inference_ms,
        )


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------


def check_environment() -> int:
    """Verifica dependências e hardware. Devolve o código de saída."""
    print("Dependências")
    print("-" * 62)
    required = {
        "pyaudiowpatch": "captura WASAPI loopback",
        "onnxruntime": "Silero VAD",
        "faster_whisper": "transcrição",
        "ctranslate2": "motor de inferência",
        "numpy": "processamento de sinal",
        "scipy": "reamostragem polifásica",
    }
    optional = {
        "PyQt6": "overlay (ou PySide6)",
        "PySide6": "overlay (alternativa)",
        "sentencepiece": "tradução local MarianMT",
    }
    missing: list[str] = []
    for module, purpose in required.items():
        ok = _probe(module)
        print(f"  [{'ok' if ok else '--'}] {module:<16} {purpose}")
        if not ok:
            missing.append(module)
    for module, purpose in optional.items():
        print(f"  [{'ok' if _probe(module) else '  '}] {module:<16} {purpose} (opcional)")

    print("\nBackend de inferência")
    print("-" * 62)
    try:
        from pipeline.transcriber import select_backend

        backend = select_backend("auto", "auto")
        print(f"  seleccionado: {backend}")
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        print(f"  GPUs CUDA visíveis: {count}")
        if count:
            supported = ", ".join(sorted(ctranslate2.get_supported_compute_types("cuda")))
            print(f"  tipos suportados em CUDA: {supported}")
            if "float16" not in supported:
                print(
                    "  nota: esta GPU é anterior a Turing (compute capability < 7.0);\n"
                    "        float16 não existe aqui e int8_float32 é o correcto."
                )
    except ImportError:
        print("  ctranslate2 em falta")

    print("\nÁudio")
    print("-" * 62)
    try:
        from pipeline.audio_capture import list_loopback_devices

        devices = list_loopback_devices()
        print(f"  {len(devices)} endpoint(s) de loopback WASAPI")
    except Exception as exc:  # noqa: BLE001 - diagnóstico
        print(f"  indisponível: {exc}")

    print("\nModelos")
    print("-" * 62)
    try:
        from pipeline.vad_engine import ensure_silero_model

        print(f"  Silero VAD: {ensure_silero_model(allow_download=False)}")
    except Exception as exc:  # noqa: BLE001 - diagnóstico
        print(f"  Silero VAD: em falta ({exc})")

    if missing:
        print("\nInstale o que falta com:")
        print("  pip install -r requirements.txt")
        return 1
    print("\nAmbiente pronto.")
    return 0


def _probe(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def list_devices() -> int:
    """Imprime os endpoints de loopback WASAPI disponíveis."""
    from pipeline.audio_capture import AudioDeviceError, list_loopback_devices

    try:
        devices = list_loopback_devices()
    except AudioDeviceError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    if not devices:
        print("nenhum endpoint de loopback encontrado")
        return 1
    for device in devices:
        print(
            f"  [{device.index:>3}] {device.name}\n"
            f"        {device.sample_rate} Hz, {device.channels} canal(is)"
        )
    return 0


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada."""
    parser = argparse.ArgumentParser(
        prog="tradutor-rt",
        description="Legendas traduzidas em tempo real a partir do áudio do sistema.",
    )
    parser.add_argument("--check", action="store_true", help="diagnostica o ambiente e sai")
    parser.add_argument("--list-devices", action="store_true", help="lista endpoints de loopback")
    parser.add_argument("--source", help="língua falada (ex.: en)")
    parser.add_argument("--target", help="língua da legenda (ex.: pt)")
    parser.add_argument("--model", help="modelo Whisper (tiny, base, small, medium)")
    parser.add_argument("--log-level", help="DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args(argv)

    if args.check:
        return check_environment()
    if args.list_devices:
        return list_devices()

    config = PipelineConfig.load()
    if args.source:
        config.source_lang = args.source
    if args.target:
        config.target_lang = args.target
    if args.model:
        config.stt.model = args.model
    if args.log_level:
        config.log_level = args.log_level

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from pipeline.translator import run_translator_process
    from pipeline.ui_overlay import run_overlay

    # "spawn" é o único método no Windows e evita partilhar estado de CUDA.
    context = mp.get_context("spawn")
    stt_queue: "mp.Queue[object]" = context.Queue(maxsize=_STT_QUEUE_SIZE)
    ui_queue: "mp.Queue[object]" = context.Queue(maxsize=_UI_QUEUE_SIZE)
    control_queue: "mp.Queue[object]" = context.Queue(maxsize=16)

    engine = context.Process(
        target=run_engine_process,
        args=(config, stt_queue, ui_queue, control_queue),
        name="engine",
        daemon=False,
    )
    translator = context.Process(
        target=run_translator_process,
        args=(config, stt_queue, ui_queue),
        name="translator",
        daemon=False,
    )
    engine.start()
    translator.start()
    LOG.info("engine pid=%s, tradução pid=%s", engine.pid, translator.pid)

    shutting_down = threading.Event()

    def shutdown() -> None:
        if shutting_down.is_set():
            return
        shutting_down.set()
        LOG.info("a encerrar o pipeline")
        _offer(control_queue, Control(action="shutdown"))

    def on_signal(_number: int, _frame: object) -> None:  # pragma: no cover
        shutdown()
        app = _qt_application()
        if app is not None:
            app.quit()

    signal.signal(signal.SIGINT, on_signal)

    try:
        code = run_overlay(ui_queue, config, on_close=shutdown)
    finally:
        shutdown()
        _stop(engine, timeout=8.0)
        _stop(translator, timeout=5.0)
        for pending in (stt_queue, ui_queue, control_queue):
            pending.close()
            pending.cancel_join_thread()
    return code


def _qt_application() -> object | None:  # pragma: no cover - só em sinal
    try:
        from pipeline.ui_overlay import QtWidgets

        return QtWidgets.QApplication.instance()
    except ImportError:
        return None


def _stop(process: "mp.Process", *, timeout: float) -> None:
    """Espera pelo fim ordeiro; se não chegar, termina à força."""
    if not process.is_alive():
        return
    process.join(timeout)
    if process.is_alive():
        LOG.warning("%s não terminou em %.0f s; a forçar", process.name, timeout)
        process.terminate()
        process.join(2.0)
    if process.is_alive():  # pragma: no cover
        process.kill()


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
