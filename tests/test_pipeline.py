# -*- coding: utf-8 -*-
"""Testes do pipeline que não dependem de modelos nem de hardware de áudio.

Executar::

    python tests/test_pipeline.py     # sem dependências extra
    pytest tests/test_pipeline.py     # se tiver pytest

Os modelos são substituídos por duplos determinísticos. O duplo do Whisper é
temporalmente coerente de propósito — a palavra *i* ocupa sempre o mesmo
intervalo absoluto, independentemente da janela em que é pedida — porque é isso
que um modelo real faz, e um duplo incoerente produziria falhas fantasma.
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.audio_capture import (  # noqa: E402
    AudioRingBuffer,
    LoopbackCapture,
    PolyphaseStreamResampler,
    _to_mono_float32,
)
from pipeline.config import (  # noqa: E402
    FRAME_SAMPLES,
    TARGET_RATE,
    MtConfig,
    PipelineConfig,
    SttConfig,
    VadConfig,
)
from pipeline.protocol import Transcript, VadPhase  # noqa: E402
from pipeline.transcriber import (  # noqa: E402
    HypothesisBuffer,
    StreamingTranscriber,
    TranscriberWorker,
    Word,
    _join,
    ends_sentence,
)
from pipeline.translator import (  # noqa: E402
    SentenceAssembler,
    TranslatorEngine,
    _first_boundary,
)
from pipeline.vad_engine import SpeechSegmenter, VadEventKind, frame_rms  # noqa: E402

FRAME_MS = FRAME_SAMPLES / TARGET_RATE * 1000.0
SILENCE, SPEECH = 0.05, 0.95


# ---------------------------------------------------------------------------
# Duplos
# ---------------------------------------------------------------------------


class ScriptedVad:
    """VAD guiado por uma lista de probabilidades."""

    def __init__(self, script: list[float]) -> None:
        self.script, self.index = script, 0

    def __call__(self, frame: np.ndarray) -> float:
        value = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return value

    def reset(self) -> None:
        pass


class _Segment:
    no_speech_prob = 0.0

    def __init__(self, words: list[object]) -> None:
        self.words = words


class _Word:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.word, self.start, self.end, self.probability = text, start, end, 0.9


class ScriptedWhisper:
    """A palavra *i* ocupa sempre o intervalo absoluto [i*step, (i+1)*step)."""

    def __init__(self, sentence: str, step: float = 0.3) -> None:
        self.tokens = sentence.split()
        self.step = step
        self.calls = 0
        self.prompts: list[str | None] = []
        self.windows: list[float] = []
        self.owner: StreamingTranscriber | None = None

    def transcribe(self, audio: np.ndarray, **kwargs: object):
        assert self.owner is not None
        self.calls += 1
        duration = len(audio) / TARGET_RATE
        self.windows.append(duration)
        self.prompts.append(kwargs.get("initial_prompt"))  # type: ignore[arg-type]
        assert kwargs["beam_size"] == 1
        assert kwargs["temperature"] == 0.0
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["word_timestamps"] is True
        assert kwargs["vad_filter"] is False

        offset = self.owner._offset
        found = []
        for index, token in enumerate(self.tokens):
            start, end = index * self.step, (index + 1) * self.step
            if start >= offset - 1e-9 and end <= offset + duration + 1e-9:
                found.append(_Word(f" {token}", start - offset, end - offset))
        return ([_Segment(found)] if found else []), None


class SilentWhisper(ScriptedWhisper):
    """Nunca reconhece nada. Exercita o tecto absoluto da janela."""

    def transcribe(self, audio: np.ndarray, **kwargs: object):
        self.calls += 1
        self.windows.append(len(audio) / TARGET_RATE)
        return [], None


class SpyTranslator:
    name = "spy"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, text: str, source: str, target: str) -> str:
        self.calls.append(text)
        return f"<{text}>"


# ---------------------------------------------------------------------------
# Reamostragem e captura
# ---------------------------------------------------------------------------


def test_resampler_matches_offline_reference() -> None:
    """O reamostrador com estado tem de igualar um resample_poly da stream toda."""
    rng = np.random.default_rng(7)
    cases = [
        (48000, 960), (48000, 137), (48000, 1), (44100, 882),
        (44100, 441), (32000, 640), (96000, 1920), (22050, 441), (16000, 512),
    ]
    for source_rate, chunk in cases:
        total = source_rate * 2
        data = (rng.standard_normal(total) * 0.3).astype(np.float32)
        divisor = np.gcd(source_rate, TARGET_RATE)
        reference = signal.resample_poly(
            data, TARGET_RATE // divisor, source_rate // divisor
        ).astype(np.float32)

        resampler = PolyphaseStreamResampler(source_rate, TARGET_RATE)
        blocks = [resampler.process(data[i : i + chunk]) for i in range(0, total, chunk)]
        produced = np.concatenate([b for b in blocks if b.size]) if blocks else np.zeros(0)

        overlap = len(produced)
        assert overlap > 0
        error = float(np.max(np.abs(reference[:overlap] - produced[:overlap])))
        assert error < 3e-6, f"{source_rate} Hz, blocos de {chunk}: erro {error}"
        # O défice é só o atraso de grupo do filtro, não amostras perdidas.
        assert 0 <= len(reference) - overlap <= 12


def test_ring_buffer_wraps_without_losing_order() -> None:
    ring = AudioRingBuffer(8)
    ring.write(np.arange(5, dtype=np.float32))
    assert np.array_equal(ring.read_last(3), [2, 3, 4])
    ring.write(np.arange(5, 11, dtype=np.float32))
    assert np.array_equal(ring.read_last(8), np.arange(3, 11, dtype=np.float32))
    ring.write(np.arange(100, 120, dtype=np.float32))  # maior que a capacidade
    assert np.array_equal(ring.read_last(), np.arange(112, 120, dtype=np.float32))
    ring.clear()
    assert len(ring) == 0 and ring.read_last().size == 0


def test_pcm_downmix_handles_truncated_buffers() -> None:
    opposing = np.array([100, -100, 200, -200], dtype=np.int16).tobytes()
    assert np.allclose(_to_mono_float32(opposing, 2), 0.0)
    assert _to_mono_float32(b"", 1).size == 0
    assert _to_mono_float32(b"\x01", 2).size == 0
    assert _to_mono_float32(b"\x00\x40\x00", 1).size == 1
    mono = _to_mono_float32(np.array([16384], dtype=np.int16).tobytes(), 1)
    assert abs(float(mono[0]) - 0.5) < 1e-6


def test_framer_preserves_every_sample() -> None:
    capture = LoopbackCapture(lambda frame, when: None)
    rng = np.random.default_rng(1)
    source = rng.standard_normal(20000).astype(np.float32)
    frames, position = [], 0
    while position < source.size:
        size = int(rng.integers(1, 700))
        frames.extend(f.copy() for f in capture._framer(source[position : position + size]))
        position += size
    assert all(f.size == FRAME_SAMPLES for f in frames)
    joined = np.concatenate(frames)
    assert np.array_equal(joined, source[: joined.size])
    assert source.size - joined.size < FRAME_SAMPLES


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


def _run_segmenter(script: list[float], **options: object):
    segmenter = SpeechSegmenter(
        vad=ScriptedVad(script), config=VadConfig(**options)  # type: ignore[arg-type]
    )
    frames = [np.full(FRAME_SAMPLES, i + 1, dtype=np.float32) for i in range(len(script))]
    events = []
    for index, frame in enumerate(frames):
        events.extend(segmenter.process(frame, index * FRAME_MS / 1000.0))
    return segmenter, events, frames


def test_preroll_precedes_speech_without_duplicating() -> None:
    _, events, _ = _run_segmenter(
        [SILENCE] * 10 + [SPEECH] * 5 + [SILENCE] * 12, min_speech_ms=0
    )
    start = events[0]
    assert start.kind is VadEventKind.SPEECH_START
    assert start.samples is not None
    assert start.samples.size == 3200 + FRAME_SAMPLES  # 200 ms + o frame actual
    assert np.all(start.samples[-FRAME_SAMPLES:] == 11.0)
    assert np.all(start.samples[-2 * FRAME_SAMPLES : -FRAME_SAMPLES] == 10.0)


def test_short_pause_does_not_split_utterance() -> None:
    segmenter, events, _ = _run_segmenter(
        [SILENCE] * 8 + [SPEECH] * 6 + [SILENCE] * 5 + [SPEECH] * 6 + [SILENCE] * 12,
        min_speech_ms=0,
    )
    starts = sum(e.kind is VadEventKind.SPEECH_START for e in events)
    ends = sum(e.kind is VadEventKind.SPEECH_END for e in events)
    assert (starts, ends) == (1, 1)  # 160 ms de pausa < 300 ms de trailing
    assert segmenter.phase is VadPhase.LISTENING


def test_long_pause_splits_utterances() -> None:
    _, events, _ = _run_segmenter(
        [SILENCE] * 6 + [SPEECH] * 10 + [SILENCE] * 10 + [SPEECH] * 10 + [SILENCE] * 12,
        min_speech_ms=0,
    )
    assert sum(e.kind is VadEventKind.SPEECH_START for e in events) == 2
    assert sum(e.kind is VadEventKind.SPEECH_END for e in events) == 2


def test_hysteresis_keeps_marginal_speech_alive() -> None:
    _, events, _ = _run_segmenter(
        [SILENCE] * 4 + [SPEECH] * 3 + [0.42] * 15 + [SILENCE] * 12, min_speech_ms=0
    )
    assert sum(e.kind is VadEventKind.SPEECH_END for e in events) == 1


def test_min_speech_measures_voice_not_trailing_silence() -> None:
    """Regressão: contar o trailing tornava min_speech_ms < 300 ms inoperante."""
    segmenter, events, _ = _run_segmenter(
        [SILENCE] * 4 + [SPEECH] * 2 + [SILENCE] * 12, min_speech_ms=200
    )
    discarded = [e for e in events if e.kind is VadEventKind.SPEECH_DISCARDED]
    assert discarded and segmenter.stats.discarded == 1
    assert discarded[0].voiced_duration * 1000 < 200 <= discarded[0].duration * 1000

    segmenter, _, _ = _run_segmenter(
        [SILENCE] * 4 + [SPEECH] * 8 + [SILENCE] * 12, min_speech_ms=200
    )
    assert segmenter.stats.utterances == 1 and segmenter.stats.discarded == 0


def test_forwarded_audio_is_gapless() -> None:
    _, events, frames = _run_segmenter(
        [SILENCE] * 8 + [SPEECH] * 20 + [SILENCE] * 12, min_speech_ms=0, preroll_ms=0
    )
    audio = np.concatenate([e.samples for e in events if e.samples is not None])
    assert np.array_equal(audio, np.concatenate(frames[8 : 8 + 20 + 9]))


def test_continuous_speech_never_emits_false_eos() -> None:
    segmenter, events, _ = _run_segmenter([SPEECH] * 80, min_speech_ms=0)
    assert not any(e.kind is VadEventKind.SPEECH_END for e in events)
    assert segmenter.phase is VadPhase.SPEAKING


def test_frame_rms() -> None:
    assert abs(frame_rms(np.full(512, 0.5, dtype=np.float32)) - 0.5) < 1e-6
    assert frame_rms(np.zeros(0, dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# LocalAgreement
# ---------------------------------------------------------------------------


def test_local_agreement_requires_two_hypotheses() -> None:
    buffer = HypothesisBuffer()
    buffer.insert([Word(" the", 0, 0.3), Word(" cat", 0.3, 0.6), Word(" sat", 0.6, 0.9)])
    assert buffer.flush() == []
    assert _join(buffer.pending) == "the cat sat"

    buffer.insert(
        [Word(" the", 0, 0.3), Word(" cat", 0.3, 0.6), Word(" sap", 0.6, 0.95),
         Word(" on", 0.95, 1.1)]
    )
    assert _join(buffer.flush()) == "the cat"

    buffer.insert([Word(" sat", 0.6, 0.9), Word(" on", 0.9, 1.1), Word(" it", 1.1, 1.3)])
    assert _join(buffer.flush()) == ""  # 'sat' discorda de 'sap'
    buffer.insert([Word(" sat", 0.6, 0.9), Word(" on", 0.9, 1.1), Word(" it.", 1.1, 1.4)])
    assert _join(buffer.flush()) == "sat on it."
    assert _join(buffer.committed) == "the cat sat on it."


def test_comparison_ignores_case_and_punctuation() -> None:
    buffer = HypothesisBuffer()
    buffer.insert([Word(" Ola", 0, 0.3)])
    buffer.flush()
    buffer.insert([Word(" ola,", 0, 0.3)])
    assert _join(buffer.flush()) == "ola,"


def test_retained_context_does_not_duplicate_words() -> None:
    buffer = HypothesisBuffer()
    for _ in range(2):
        buffer.insert([Word(" hello", 0, 0.4), Word(" world", 0.4, 0.8)])
    assert _join(buffer.flush()) == "hello world"
    buffer.insert(
        [Word(" hello", 0, 0.4), Word(" world", 0.4, 0.8), Word(" again", 0.85, 1.2)]
    )
    assert _join(buffer.pending) == "again"


def test_ends_sentence() -> None:
    assert ends_sentence("Ola mundo.") and ends_sentence("Certo?")
    assert not ends_sentence("Ola mundo")


# ---------------------------------------------------------------------------
# Janela deslizante
# ---------------------------------------------------------------------------


def _drive(model: ScriptedWhisper, config: SttConfig, steps: int, chunk: float = 0.3):
    transcriber = StreamingTranscriber(model, config, "en")
    model.owner = transcriber
    samples = np.zeros(int(TARGET_RATE * chunk), dtype=np.float32)
    now, deltas = 0.0, []
    for _ in range(steps):
        transcriber.append(samples, now)
        now += chunk
        if transcriber.ready(now):
            result = transcriber.step(now)
            if result and result.delta:
                deltas.append(result.delta)
    return transcriber, deltas


def test_window_stays_bounded_and_text_is_contiguous() -> None:
    model = ScriptedWhisper(" ".join(f"w{i}" for i in range(60)))
    config = SttConfig(
        context_seconds=2.0, max_window_seconds=15.0,
        min_chunk_seconds=0.3, min_infer_interval=0.0,
    )
    transcriber, deltas = _drive(model, config, steps=50)
    indices = [int(token[1:]) for token in " ".join(deltas).split()]
    assert indices == list(range(indices[0], indices[0] + len(indices)))
    assert len(indices) >= 40
    # Estabiliza perto de context_seconds mais a cauda por confirmar.
    assert max(model.windows) < 6.0
    assert transcriber.window_seconds < 6.0


def test_ceiling_applies_when_nothing_is_ever_confirmed() -> None:
    """Regressão: o tecto só corria dentro do corte por confirmação."""
    model = SilentWhisper("ignorado")
    config = SttConfig(max_window_seconds=10.0, min_chunk_seconds=0.3, min_infer_interval=0.0)
    transcriber, _ = _drive(model, config, steps=120, chunk=0.5)
    assert max(model.windows) <= config.max_window_seconds + 0.6
    assert transcriber.window_seconds <= config.max_window_seconds + 0.6


def test_prompt_excludes_words_still_inside_the_window() -> None:
    model = ScriptedWhisper(" ".join(f"w{i}" for i in range(60)))
    transcriber, _ = _drive(
        model,
        SttConfig(context_seconds=2.0, min_chunk_seconds=0.3, min_infer_interval=0.0),
        steps=40,
    )
    prompts = [p for p in model.prompts if p]
    assert prompts, "o contexto textual nunca foi aproveitado"
    assert len(prompts[-1]) <= SttConfig().prompt_chars
    inside = {
        f"w{i}"
        for i in range(int(transcriber._offset / model.step), 60)
    }
    assert not set(prompts[-1].split()) & inside


def test_finalize_keeps_every_word() -> None:
    """Regressão: o flush intermédio do EOS era descartado e perdia palavras."""
    sentence = "Hello there. How are you doing today. I am fine."
    model = ScriptedWhisper(sentence)
    transcriber, deltas = _drive(
        model,
        SttConfig(context_seconds=2.0, min_chunk_seconds=0.3, min_infer_interval=0.0),
        steps=11,
    )
    final = transcriber.finalize()
    assert final.is_final
    assert final.committed == sentence, final.committed
    assert transcriber.window_seconds == 0.0


def test_inference_is_self_paced() -> None:
    model = ScriptedWhisper("um dois tres")
    config = SttConfig(min_chunk_seconds=0.4, min_infer_interval=0.25)
    _drive(model, config, steps=40, chunk=0.1)  # 4 s de áudio
    assert 8 <= model.calls <= 17


# ---------------------------------------------------------------------------
# Tradução
# ---------------------------------------------------------------------------


def _transcript(seq: int, revision: int, committed: str, interim: str, delta: str,
                is_final: bool = False) -> Transcript:
    return Transcript(seq, revision, committed, interim, delta, is_final, 1.0,
                      time.monotonic())


def test_sentence_assembler() -> None:
    assembler = SentenceAssembler()
    assert assembler.feed(" Bom") == []
    assert assembler.feed(" dia.") == ["Bom dia."]
    assert assembler.feed(" Tudo bem? Sim") == ["Tudo bem?"]
    assert assembler.pending == "Sim"
    assert assembler.flush() == ["Sim"] and assembler.flush() == []
    assert _first_boundary("3.14 euros") == -1  # um decimal não fecha frase
    assert _first_boundary("Fim. Depois") == 3


def test_committed_text_is_never_assembled_from_fragments() -> None:
    """A cauda especulativa traduz fragmentos de propósito — é para isso que serve.

    O que não pode acontecer é o texto *confirmado* ser a colagem de traduções
    de meias-frases, porque entre línguas com ordens de palavras diferentes isso
    produz disparates que depois nunca mais são corrigidos.
    """
    spy = SpyTranslator()
    engine = TranslatorEngine(spy, source_lang="en", target_lang="pt",
                              config=MtConfig(interim_debounce_ms=0))
    engine.handle(_transcript(1, 1, "Hello", " there", " Hello"), 0.0)
    engine.handle(_transcript(1, 2, "Hello there", " world", " there"), 0.0)
    result = engine.handle(_transcript(1, 3, "Hello there world.", "", " world."), 0.0)

    assert result is not None
    # Uma só tradução, da frase inteira — e não três coladas.
    assert result.committed == "<Hello there world.>"
    assert result.committed.count("<") == 1
    assert "Hello there world." in spy.calls


def test_interim_is_debounced_but_never_delayed_on_first_use() -> None:
    """Regressão: _interim_at a 0.0 atrasava o primeiro especulativo."""
    spy = SpyTranslator()
    engine = TranslatorEngine(spy, source_lang="en", target_lang="pt",
                              config=MtConfig(interim_debounce_ms=1000))
    engine.handle(_transcript(2, 1, "", "a b", ""), 0.0)
    assert len(spy.calls) == 1, "o primeiro especulativo tem de passar já"
    engine.handle(_transcript(2, 2, "", "a b c", ""), 0.1)
    engine.handle(_transcript(2, 3, "", "a b c d", ""), 0.2)
    assert len(spy.calls) == 1
    engine.handle(_transcript(2, 4, "", "a b c d e", ""), 1.5)
    assert len(spy.calls) == 2


def test_cache_and_language_switch() -> None:
    spy = SpyTranslator()
    engine = TranslatorEngine(spy, source_lang="en", target_lang="pt", config=MtConfig())
    engine.handle(_transcript(3, 1, "Ok.", "", "Ok."), 0.0)
    engine.handle(_transcript(4, 1, "Ok.", "", "Ok."), 0.0)
    assert spy.calls == ["Ok."] and engine.stats.cache_hits == 1

    final = engine.handle(_transcript(4, 2, "Ok.", "", "", True), 0.0)
    assert final and final.is_final and final.interim == ""

    engine.set_languages("en", "es")
    assert not engine._cache


def test_identical_languages_skip_the_backend() -> None:
    spy = SpyTranslator()
    engine = TranslatorEngine(spy, source_lang="pt", target_lang="pt", config=MtConfig())
    result = engine.handle(_transcript(5, 1, "Ola mundo.", "", "Ola mundo."), 0.0)
    assert result and result.committed == "Ola mundo." and not spy.calls


def test_accumulation_has_no_double_spaces() -> None:
    spy = SpyTranslator()
    engine = TranslatorEngine(spy, source_lang="en", target_lang="pt", config=MtConfig())
    engine.handle(_transcript(6, 1, "", "", " A."), 0.0)
    engine.handle(_transcript(6, 2, "", "", " B."), 0.0)
    assert engine._committed == "<A.> <B.>"


# ---------------------------------------------------------------------------
# Integração
# ---------------------------------------------------------------------------


def test_full_chain_from_48khz_stereo_pcm() -> None:
    """Percorre captura, reamostragem, VAD, STT e tradução, sem threads."""
    sentence = "Hello there. How are you doing today. I am fine."
    source_rate, duration = 48000, 5.0

    axis = np.arange(int(source_rate * duration)) / source_rate
    interleaved = np.empty(axis.size * 2, dtype=np.int16)
    interleaved[0::2] = 0.3 * np.sin(2 * np.pi * 180 * axis) * 32767
    interleaved[1::2] = 0.3 * np.sin(2 * np.pi * 181 * axis) * 32767
    raw = interleaved.tobytes()

    resampler = PolyphaseStreamResampler(source_rate, TARGET_RATE)
    per_second = TARGET_RATE / FRAME_SAMPLES
    segmenter = SpeechSegmenter(
        vad=ScriptedVad(
            [SILENCE] * int(per_second) + [SPEECH] * int(3 * per_second)
            + [SILENCE] * int(per_second)
        ),
        config=VadConfig(min_speech_ms=100),
    )

    events: queue.Queue[object] = queue.Queue(maxsize=4096)
    produced: list[Transcript] = []
    worker = TranscriberWorker(
        events, produced.append,
        SttConfig(min_chunk_seconds=0.3, min_infer_interval=0.0), language="en",
    )
    model = ScriptedWhisper(sentence)
    worker._transcriber = StreamingTranscriber(model, worker._config, "en")
    model.owner = worker._transcriber

    block = int(source_rate * 0.02) * 4  # 20 ms, 2 canais, 2 bytes por amostra
    pending, frames = np.zeros(0, dtype=np.float32), 0
    for offset in range(0, len(raw), block):
        converted = resampler.process(_to_mono_float32(raw[offset : offset + block], 2))
        pending = np.concatenate((pending, converted)) if pending.size else converted
        while pending.size >= FRAME_SAMPLES:
            frame, pending = pending[:FRAME_SAMPLES], pending[FRAME_SAMPLES:]
            frames += 1
            for event in segmenter.process(frame, time.monotonic()):
                events.put_nowait(event)
        worker._consume()
        worker._maybe_infer()
    for _ in range(4):
        worker._consume()
        worker._maybe_infer()

    assert abs(frames - duration * TARGET_RATE / FRAME_SAMPLES) < 3
    assert segmenter.stats.utterances == 1

    finals = [item for item in produced if item.is_final]
    assert finals and finals[-1].committed == sentence

    spy = SpyTranslator()
    engine = TranslatorEngine(spy, source_lang="en", target_lang="pt",
                              config=MtConfig(interim_debounce_ms=200))
    translations = [
        item for item in
        (engine.handle(t, index * 0.05) for index, t in enumerate(produced))
        if item
    ]
    assert translations[-1].is_final
    assert translations[-1].committed.count("<") == 3  # três frases inteiras
    assert len(spy.calls) < len(produced)  # a deduplicação poupou chamadas


def test_config_round_trip(tmp_path: Path | None = None) -> None:
    import tempfile

    directory = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    target = directory / "pipeline.json"
    config = PipelineConfig()
    config.stt.model = "medium"
    config.vad.trailing_ms = 250.0
    config.save(target)

    loaded = PipelineConfig.load(target)
    assert loaded.stt.model == "medium"
    assert loaded.vad.trailing_ms == 250.0

    target.write_text('{"stt": {"desconhecido": 1}, "lixo": 2}', encoding="utf-8")
    recovered = PipelineConfig.load(target)
    assert recovered.stt.model == SttConfig().model  # chaves estranhas ignoradas

    target.write_text("isto nao e json", encoding="utf-8")
    assert PipelineConfig.load(target).stt.model == SttConfig().model


def main() -> int:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - relatório de testes
            failures += 1
            print(f"  FALHOU  {test.__name__}: {exc}")
        else:
            print(f"  ok      {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
