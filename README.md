# Tradutor de Live em Tempo Real

Legendas traduzidas a partir do áudio que o Windows está a reproduzir, com
latência-alvo entre **350 ms e 650 ms**.

---

## Pipeline v2 (`main.py`)

```
processo "engine"                 processo "translator"     processo principal
┌───────────────────────────┐     ┌──────────────────┐      ┌───────────────┐
│ captura WASAPI loopback   │     │ MarianMT int8    │      │ Qt + overlay  │
│   48 kHz → 16 kHz mono    │     │  ou serviço web  │      │ click-through │
│ Silero VAD + FSM (32 ms)  │     └──────────────────┘      └───────────────┘
│ faster-whisper / CT2      │            ▲    │                     ▲
└───────────────────────────┘            │    └─── traduções ───────┘
              └──── transcrições ────────┘
```

Três processos, não três threads: o CTranslate2 liberta o GIL durante a
inferência, mas o Qt não, e a tradução pode ficar pendurada num pedido de rede.
Separados, nenhum consegue atrasar os outros.

### Arranque

```bash
pip install -r requirements.txt
python main.py --check          # diagnostica dependências, GPU e modelos
python main.py --list-devices   # endpoints de loopback WASAPI
python main.py                  # arranca o pipeline
python main.py --source en --target pt --model small
```

O `silero_vad.onnx` é descarregado para `models/` na primeira execução. O modelo
Whisper é descarregado pelo `faster-whisper` para a cache do Hugging Face.

### Configuração

Tudo em `pipeline.json`, na raiz. Os parâmetros que mais afectam a latência:

| Chave | Omissão | Efeito |
|---|---|---|
| `stt.model` | `small` | `base` é mais rápido, `medium` mais exacto |
| `stt.min_infer_interval` | `0.25` | intervalo mínimo entre inferências |
| `stt.min_chunk_seconds` | `0.4` | áudio novo necessário para reinferir |
| `stt.context_seconds` | `2.0` | áudio confirmado retido como contexto |
| `vad.trailing_ms` | `300` | silêncio até declarar fim de frase |
| `vad.preroll_ms` | `200` | áudio guardado antes do início da fala |
| `mt.interim_debounce_ms` | `220` | trava de retradução do texto especulativo |

`stt.language` é uma sobreposição: vazio segue `source_lang`, `"auto"` deixa o
Whisper detectar (custa uma passagem extra por elocução).

### Atalhos do overlay

| Tecla | Acção |
|---|---|
| `Ctrl+L` | liga/desliga click-through |
| `+` / `-` | tamanho da letra |
| `Esc` | fechar |

Arrastar move a janela; o canto inferior direito redimensiona. A posição e o
tamanho ficam gravados.

### Tradução local (opcional)

Sem isto, a tradução vai pelo endpoint JSON do Google (sem chave, 80–200 ms).
Com modelo local, 15–40 ms e sem rede:

```bash
pip install transformers[torch] sentencepiece
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ROMANCE \
    --output_dir models/opus-mt-en-ROMANCE --quantization int8 \
    --copy_files source.spm target.spm
```

Depois aponte `mt.model_dir` para essa pasta. O token de língua-alvo
(`>>por<<` ou `>>pt<<`, conforme o modelo) é detectado a partir do vocabulário.

---

## Decisões de desenho que não são óbvias

**Sem Stereo Mix nem VB-CABLE.** A captura usa WASAPI loopback nativo sobre o
dispositivo de saída padrão. Quando o utilizador troca de saída, o PortAudio não
dá por isso — a lista de dispositivos é fotografada no `Pa_Initialize`. Por isso
o silêncio digital prolongado desencadeia uma re-sondagem com uma instância nova
de PortAudio.

**Reamostragem com estado.** O WASAPI em shared mode entrega o *mix format* do
endpoint (tipicamente 48 kHz estéreo); não há como pedir 16 kHz ao driver.
Aplicar `resample_poly` bloco a bloco injectaria uma descontinuidade a cada
32 ms, o que degrada o VAD e o log-mel do Whisper. O reamostrador aqui mantém
histórico e fase entre chamadas e produz, amostra a amostra, o mesmo resultado
que um `resample_poly` sobre a stream inteira — validado em teste.

**LocalAgreement-2 em vez de fatiamento.** Cortar o áudio em blocos de 6 s corta
a meio de palavras e rouba ao Whisper o contexto de que ele precisa: dá latência
alta *e* qualidade baixa. Aqui a janela é redecodificada à medida que cresce, e
uma palavra só é fixada quando duas inferências consecutivas concordam nela. Daí
saem dois fluxos: `committed`, estável, que alimenta a tradução, e `interim`,
especulativo, que aparece de imediato no overlay.

**`condition_on_previous_text` desligado.** Em streaming, realimentar a
transcrição anterior faz o Whisper entrar em ciclos de repetição ao encontrar
silêncio ou ruído. O contexto vai como `initial_prompt`, que influencia sem
realimentar — e exclui as palavras ainda dentro da janela, para não induzir o
modelo a repetir-se.

**`temperature=0.0` fixo.** A omissão do faster-whisper é uma cascata de
temperaturas com reinferência, o que produz picos de latência imprevisíveis.

**Tradução por frases, não por fragmentos.** Só o `delta` confirmado é
acumulado, e só é traduzido ao fechar uma frase. Traduzir meias-frases entre
línguas com ordens de palavras diferentes produz disparates que depois nunca
mais são corrigidos.

---

## Tipo de cómputo e GPU

`compute_type: "auto"` interroga o CTranslate2 em vez de assumir. Isto importa:

- **`float16` exige compute capability ≥ 7.0** (Turing, RTX 20xx ou superior).
- Em **Pascal (GTX 10xx, cc 6.1)** o `float16` não existe no CTranslate2; o
  correcto é **`int8_float32`**, que usa as instruções dp4a.

Confirme com `python main.py --check`. Para GPU sem CUDA Toolkit instalado:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Se as bibliotecas CUDA faltarem, o carregamento do modelo recua sozinho para CPU.

---

## Testes

```bash
python tests/test_pipeline.py     # 29 testes, sem modelos nem hardware
pytest tests/test_pipeline.py     # se tiver pytest
```

Cobrem o reamostrador contra uma referência offline, a FSM do VAD (pre-roll,
histerese, pausas curtas, blips), o LocalAgreement, os limites da janela
deslizante, a deduplicação da tradução, e a cadeia completa a partir de PCM
48 kHz estéreo.

---

## Versão anterior

`tradutor.pyw` e `modules/` são a v1 (Google STT por HTTP, janelas de 6 s,
Stereo Mix) e continuam a funcionar de forma independente. O `pipeline.json` é
criado a partir do `settings.json` antigo na primeira execução, reaproveitando o
par de línguas e o aspecto do overlay.
