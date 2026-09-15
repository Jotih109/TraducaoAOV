# -*- coding: utf-8 -*-
"""Tradução desacoplada, local (CTranslate2/MarianMT) ou remota.

Corre num processo próprio: a tradução não pode nunca bloquear a captura nem a
transcrição, e um pedido de rede pendurado tem de ser invisível para o resto do
pipeline.

Deduplicação e actualização por diferenças
------------------------------------------

O STT emite revisões sucessivas da mesma elocução. Traduzir o texto inteiro a
cada revisão seria dezenas de traduções por frase — caro, lento e instável, já
que o tradutor pode produzir resultados diferentes para prefixos diferentes.

O que se faz em vez disso:

1. Só o ``delta`` (texto acabado de confirmar) é acumulado, nunca o texto todo.
2. O delta acumulado só é traduzido quando fecha uma **frase** — pontuação
   forte ou fim de elocução. Traduzir meias-frases produz péssima qualidade,
   sobretudo entre línguas com ordem de palavras diferente.
3. A tradução de cada frase é fixada e nunca reprocessada.
4. A cauda especulativa é traduzida à parte, com *debounce*, e só aparece como
   texto provisório.
5. Uma cache LRU absorve as repetições inevitáveis entre revisões.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, Sequence

from pipeline.config import MtConfig, PipelineConfig
from pipeline.protocol import Fatal, Stage, Status, Transcript, Translation
from pipeline.transcriber import ends_sentence

__all__ = [
    "CloudTranslator",
    "CTranslate2Translator",
    "NullTranslator",
    "SentenceAssembler",
    "TranslationBackend",
    "TranslatorEngine",
    "build_backend",
    "run_translator_process",
]

LOG = logging.getLogger(__name__)

_LANG_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^>>[a-z]{2,3}(_[A-Za-z]+)?<<$")
_USER_AGENT: Final[str] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) tradutor-rt/2.0"
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _join_text(left: str, right: str) -> str:
    """Concatena dois fragmentos colapsando espaços.

    Os tokens do Whisper já trazem o espaço à cabeça, pelo que uma concatenação
    ingénua produz espaços duplos que depois alteram a chave da cache.
    """
    return _WHITESPACE_RE.sub(" ", f"{left} {right}").strip()


def _tidy(text: str) -> str:
    """Normaliza espaços e capitaliza o início, para estilo de legenda."""
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


class TranslationBackend(Protocol):
    """Contrato mínimo de um motor de tradução."""

    name: str

    def translate(self, text: str, source: str, target: str) -> str:
        """Traduz ``text``. Devolve string vazia se não conseguir."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class NullTranslator:
    """Não traduz. Usado quando a tradução está desligada."""

    name = "off"

    def translate(self, text: str, source: str, target: str) -> str:
        return ""


class CTranslate2Translator:
    """MarianMT convertido para CTranslate2, quantizado em int8.

    Sem rede e sem servidor: a inferência é uma chamada em processo, o que
    tipicamente custa 15 a 40 ms por frase curta em CPU int8 — uma ordem de
    grandeza abaixo de qualquer ida à rede.

    O modelo tem de estar previamente convertido, contendo ``model.bin``,
    ``source.spm`` e ``target.spm``::

        pip install transformers[torch] sentencepiece
        ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ROMANCE \\
            --output_dir models/opus-mt-en-ROMANCE --quantization int8 \\
            --copy_files source.spm target.spm

    Args:
        model_dir: Directório do modelo convertido.
        device: ``"cpu"``, ``"cuda"`` ou ``"auto"``.
        compute_type: Tipo de cómputo do CTranslate2.
        beam_size: 1 (guloso) para latência mínima.
        target_token: Token de língua-alvo para modelos multi-alvo. Vazio faz a
            detecção automática a partir do vocabulário do modelo.

    Raises:
        FileNotFoundError: Se faltar algum ficheiro do modelo.
        ImportError: Se ``ctranslate2`` ou ``sentencepiece`` não existirem.
    """

    name = "ctranslate2"

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        target_token: str = "",
        max_decoding_length: int = 256,
        target_lang: str = "",
    ) -> None:
        import ctranslate2
        import sentencepiece

        path = Path(model_dir)
        source_spm = path / "source.spm"
        target_spm = path / "target.spm"
        for required in (path / "model.bin", source_spm, target_spm):
            if not required.exists():
                raise FileNotFoundError(f"modelo de tradução incompleto: {required}")

        resolved = device
        if device == "auto":
            try:
                resolved = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except (RuntimeError, OSError):
                resolved = "cpu"

        self._translator = ctranslate2.Translator(
            str(path), device=resolved, compute_type=compute_type
        )
        self._source = sentencepiece.SentencePieceProcessor(str(source_spm))
        self._target = sentencepiece.SentencePieceProcessor(str(target_spm))
        self._beam_size = beam_size
        self._max_decoding_length = max_decoding_length
        self._token = target_token or detect_target_token(path, target_lang)
        LOG.info(
            "tradutor local carregado de %s (%s/%s%s)",
            path.name,
            resolved,
            compute_type,
            f", token {self._token}" if self._token else "",
        )

    def translate(self, text: str, source: str, target: str) -> str:
        if not text:
            return ""
        tokens = self._source.encode(text, out_type=str)
        if self._token:
            tokens = [self._token, *tokens]
        tokens.append("</s>")
        try:
            results = self._translator.translate_batch(
                [tokens],
                beam_size=self._beam_size,
                max_decoding_length=self._max_decoding_length,
                return_scores=False,
            )
        except Exception:
            LOG.exception("tradução local falhou")
            return ""
        if not results or not results[0].hypotheses:
            return ""
        return self._target.decode(results[0].hypotheses[0])


def detect_target_token(model_dir: Path, target_lang: str) -> str:
    """Descobre o token de língua-alvo de um modelo Marian multi-alvo.

    Modelos como ``opus-mt-en-ROMANCE`` exigem um prefixo do tipo ``>>por<<``,
    mas a convenção varia (ISO-639-1 ou ISO-639-3). Em vez de adivinhar, lê-se o
    vocabulário do modelo e procura-se o token que corresponde à língua pedida.

    Returns:
        O token, ou string vazia para modelos de par único (que não o usam).
    """
    if not target_lang:
        return ""
    vocabulary: Sequence[str] = ()
    for candidate in ("shared_vocabulary.json", "target_vocabulary.json"):
        path = model_dir / candidate
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(loaded, list):
                vocabulary = loaded
                break
    if not vocabulary:
        for candidate in ("shared_vocabulary.txt", "target_vocabulary.txt"):
            path = model_dir / candidate
            if path.exists():
                try:
                    vocabulary = path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                break

    tokens = [token for token in vocabulary if _LANG_TOKEN_RE.match(token)]
    if not tokens:
        return ""  # modelo de par único
    prefix = target_lang.lower()[:2]
    for token in tokens:
        if token[2:-2].lower().startswith(prefix):
            return token
    LOG.warning(
        "o modelo exige um token de língua mas nenhum corresponde a %r (tem: %s)",
        target_lang,
        ", ".join(tokens[:8]),
    )
    return ""


class CloudTranslator:
    """Tradução por serviço remoto, com recuo entre fornecedores.

    O fornecedor por omissão é o ponto final JSON directo do Google Translate,
    que não exige chave e responde tipicamente em 80 a 200 ms.

    Args:
        api: ``"google"``, ``"deepl"`` ou ``"openai"``.
        api_key: Credencial, obrigatória para DeepL e OpenAI.
        timeout: Tempo limite por pedido.
    """

    name = "cloud"

    def __init__(
        self, *, api: str = "google", api_key: str = "", timeout: float = 2.5
    ) -> None:
        self._api = api
        self._api_key = api_key
        self._timeout = timeout
        if api in ("deepl", "openai") and not api_key:
            LOG.warning("%s sem chave configurada; a usar o Google", api)
            self._api = "google"
        self.name = f"cloud:{self._api}"

    def translate(self, text: str, source: str, target: str) -> str:
        if not text:
            return ""
        handlers = {
            "google": self._google,
            "deepl": self._deepl,
            "openai": self._openai,
        }
        primary = handlers[self._api]
        for handler in (primary, self._google):
            try:
                result = handler(text, source, target)
            except (urllib.error.URLError, OSError, ValueError, KeyError, TimeoutError):
                LOG.debug("fornecedor de tradução falhou", exc_info=True)
                continue
            if result:
                return result
            if handler is self._google:
                break
        return ""

    def _request(
        self,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        request = urllib.request.Request(  # noqa: S310 - esquemas https fixos
            url,
            data=data,
            headers={"User-Agent": _USER_AGENT, **(headers or {})},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return response.read()

    def _google(self, text: str, source: str, target: str) -> str:
        query = urllib.parse.urlencode(
            {"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": text}
        )
        payload = json.loads(
            self._request(f"https://translate.googleapis.com/translate_a/single?{query}")
        )
        if not payload or not payload[0]:
            return ""
        return "".join(part[0] for part in payload[0] if part and part[0]).strip()

    def _deepl(self, text: str, source: str, target: str) -> str:
        host = "api-free" if self._api_key.endswith(":fx") else "api"
        body = urllib.parse.urlencode(
            {
                "text": text,
                "source_lang": source.upper(),
                "target_lang": target.upper(),
            }
        ).encode()
        payload = json.loads(
            self._request(
                f"https://{host}.deepl.com/v2/translate",
                data=body,
                headers={
                    "Authorization": f"DeepL-Auth-Key {self._api_key}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        )
        return str(payload["translations"][0]["text"]).strip()

    def _openai(self, text: str, source: str, target: str) -> str:
        body = json.dumps(
            {
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"Translate from {source} to {target}. "
                            "Reply with the translation only, no commentary."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            }
        ).encode()
        payload = json.loads(
            self._request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        )
        return str(payload["choices"][0]["message"]["content"]).strip()


def build_backend(config: MtConfig, target_lang: str) -> TranslationBackend:
    """Instancia o backend pedido, recuando para a nuvem se o local falhar."""
    if config.backend == "off":
        return NullTranslator()

    if config.backend in ("auto", "local") and config.model_dir:
        try:
            return CTranslate2Translator(
                config.model_dir,
                device=config.device,
                compute_type=config.compute_type,
                beam_size=config.beam_size,
                target_token=config.target_token,
                max_decoding_length=config.max_decoding_length,
                target_lang=target_lang,
            )
        except (ImportError, FileNotFoundError, RuntimeError) as exc:
            if config.backend == "local":
                LOG.error("tradutor local indisponível: %s", exc)
                return NullTranslator()
            LOG.warning("tradutor local indisponível (%s); a usar a nuvem", exc)

    return CloudTranslator(
        api=config.api, api_key=config.api_key, timeout=config.request_timeout
    )


# ---------------------------------------------------------------------------
# Montagem de frases
# ---------------------------------------------------------------------------


class SentenceAssembler:
    """Acumula deltas confirmados e liberta-os em frases completas.

    Traduzir fragmentos à medida que chegam degrada muito a qualidade entre
    línguas com ordens de palavras diferentes. Acumular até uma fronteira de
    frase custa algumas centenas de milissegundos na parte *confirmada* — que
    não é a que o utilizador está a ler nesse instante, porque a cauda
    especulativa já está no ecrã.

    Example:
        >>> assembler = SentenceAssembler()
        >>> assembler.feed(" Hello there")
        []
        >>> assembler.feed(" world. And more")
        ['Hello there world.']
        >>> assembler.flush()
        ['And more']
    """

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending = ""

    @property
    def pending(self) -> str:
        """Fragmento ainda por fechar."""
        return self._pending

    def reset(self) -> None:
        self._pending = ""

    def feed(self, delta: str) -> list[str]:
        """Acrescenta texto confirmado; devolve as frases que ficaram completas."""
        if not delta:
            return []
        self._pending = _join_text(self._pending, delta)
        sentences: list[str] = []
        while True:
            cut = _first_boundary(self._pending)
            if cut < 0:
                break
            sentences.append(self._pending[: cut + 1].strip())
            self._pending = self._pending[cut + 1 :].strip()
        return [sentence for sentence in sentences if sentence]

    def flush(self) -> list[str]:
        """Liberta o fragmento pendente. Usado no fim da elocução."""
        remaining = self._pending.strip()
        self._pending = ""
        return [remaining] if remaining else []


def _first_boundary(text: str) -> int:
    """Índice do primeiro fim de frase, ou -1.

    Um ponto só fecha a frase se for seguido de espaço ou fim de texto, para não
    partir abreviaturas nem números decimais.
    """
    for index, char in enumerate(text):
        if not ends_sentence(char):
            continue
        if index + 1 >= len(text) or text[index + 1].isspace():
            return index
    return -1


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TranslatorStats:
    """Contadores de diagnóstico da tradução."""

    calls: int = 0
    cache_hits: int = 0
    skipped: int = 0
    total_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class TranslatorEngine:
    """Aplica a política de deduplicação sobre um :class:`TranslationBackend`.

    Args:
        backend: Motor de tradução.
        source_lang: Língua de partida.
        target_lang: Língua de chegada.
        config: Parâmetros de debounce e cache.
    """

    def __init__(
        self,
        backend: TranslationBackend,
        *,
        source_lang: str,
        target_lang: str,
        config: MtConfig | None = None,
    ) -> None:
        self._backend = backend
        self._source = source_lang
        self._target = target_lang
        self._config = config or MtConfig()

        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._assembler = SentenceAssembler()
        self._sequence = -1
        self._committed = ""
        self._interim = ""
        self._interim_source = ""
        # -inf e não 0.0: o primeiro especulativo tem de passar sempre, mesmo
        # que o relógio monotónico da plataforma arranque perto de zero.
        self._interim_at = float("-inf")
        self.stats = TranslatorStats()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def set_languages(self, source: str, target: str) -> None:
        """Troca o par de línguas e invalida o que estava em curso."""
        if (source, target) == (self._source, self._target):
            return
        self._source, self._target = source, target
        self._cache.clear()
        self._reset_utterance()

    def handle(self, transcript: Transcript, now: float | None = None) -> Translation | None:
        """Processa uma revisão de transcrição.

        Returns:
            A tradução actualizada, ou ``None`` se nada mudou o suficiente para
            valer uma nova mensagem (o caso mais comum, e é isso que mantém o
            custo baixo).
        """
        moment = time.monotonic() if now is None else now
        if transcript.seq != self._sequence:
            self._sequence = transcript.seq
            self._reset_utterance()

        changed = False
        for sentence in self._assembler.feed(transcript.delta):
            rendered = self._translate(sentence)
            if rendered:
                self._committed = _join_text(self._committed, rendered)
                changed = True

        if transcript.is_final:
            for sentence in self._assembler.flush():
                rendered = self._translate(sentence)
                if rendered:
                    self._committed = _join_text(self._committed, rendered)
            self._interim = ""
            self._interim_source = ""
            changed = True
        elif self._refresh_interim(transcript, moment):
            changed = True

        if not changed:
            self.stats.skipped += 1
            return None

        return Translation(
            seq=transcript.seq,
            revision=transcript.revision,
            source_text=transcript.text,
            committed=self._committed,
            interim=self._interim,
            is_final=transcript.is_final,
            backend=self._backend.name,
            captured_at=transcript.captured_at,
        )

    def _refresh_interim(self, transcript: Transcript, now: float) -> bool:
        """Retraduz a cauda especulativa, com debounce."""
        tail = _join_text(self._assembler.pending, transcript.interim)
        if tail == self._interim_source:
            return False
        if not tail:
            self._interim = ""
            self._interim_source = ""
            return True
        if (now - self._interim_at) * 1000.0 < self._config.interim_debounce_ms:
            return False
        self._interim_at = now
        self._interim_source = tail
        self._interim = self._translate(tail)
        return True

    def _translate(self, text: str) -> str:
        tidy = _tidy(text)
        if not tidy:
            return ""
        if self._source == self._target:
            return tidy
        key = (self._source, self._target, tidy.casefold())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats.cache_hits += 1
            return cached

        started = time.monotonic()
        rendered = _tidy(self._backend.translate(tidy, self._source, self._target))
        self.stats.calls += 1
        self.stats.total_ms += (time.monotonic() - started) * 1000.0
        if not rendered:
            return ""
        self._cache[key] = rendered
        while len(self._cache) > self._config.cache_size:
            self._cache.popitem(last=False)
        return rendered

    def _reset_utterance(self) -> None:
        self._assembler.reset()
        self._committed = ""
        self._interim = ""
        self._interim_source = ""


# ---------------------------------------------------------------------------
# Processo
# ---------------------------------------------------------------------------


def run_translator_process(
    config: PipelineConfig,
    inbox: "mp.Queue[Transcript | None]",
    outbox: "mp.Queue[object]",
) -> None:
    """Ponto de entrada do processo de tradução.

    Consome :class:`Transcript` e publica :class:`Translation`. Termina quando
    recebe ``None``.
    """
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [mt] %(message)s",
    )
    try:
        backend = build_backend(config.mt, config.target_lang)
        engine = TranslatorEngine(
            backend,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            config=config.mt,
        )
    except Exception as exc:  # pragma: no cover - falha de arranque
        LOG.exception("processo de tradução não arrancou")
        _offer(outbox, Fatal(Stage.MT, "Tradução indisponível", str(exc)))
        return

    _offer(outbox, Status(Stage.MT, f"Tradução pronta ({backend.name})"))
    while True:
        try:
            message = inbox.get()
        except (EOFError, OSError):
            break
        if message is None:
            break
        if not isinstance(message, Transcript):
            continue
        try:
            translation = engine.handle(message)
        except Exception:  # pragma: no cover - nunca derrubar o processo
            LOG.exception("falha ao traduzir")
            continue
        if translation is not None:
            _offer(outbox, translation)

    LOG.info(
        "tradução terminada: %d chamadas, %d acertos de cache, média %.0f ms",
        engine.stats.calls,
        engine.stats.cache_hits,
        engine.stats.mean_ms,
    )


def _offer(outbox: "mp.Queue[object]", message: object) -> None:
    """Publica sem nunca bloquear: a UI a atrasar não pode travar a tradução."""
    try:
        outbox.put_nowait(message)
    except queue.Full:
        LOG.debug("fila da UI cheia; mensagem descartada")
