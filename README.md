# Podcast Cutter AI

[![CI](https://github.com/seu-usuario/podcast-cutter-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/seu-usuario/podcast-cutter-ai/actions/workflows/ci.yml)

Pipeline local para detecção e geração automática de cortes virais em podcasts e lives do YouTube.

## O que faz

Dado uma URL do YouTube, o pipeline executa 8 etapas automaticamente:

| Stage | Descrição | Tecnologia |
|---|---|---|
| 1 — Audio Download | Baixa apenas o áudio (~180 MB para 3h) | yt-dlp |
| 2 — Transcription | Transcreve localmente com timestamps | faster-whisper |
| 3 — Content Analysis | LLM lê a transcrição e identifica momentos virais | Mistral via Ollama |
| 4 — Video Download | Baixa o vídeo em 720p somente após confirmar clips | yt-dlp |
| 5 — Video Cutting | Corta cada clip detectado como MP4 independente | FFmpeg (stream-copy) |
| 6 — Subtitle Generation | Gera arquivos SRT e ASS para cada clip | Python puro |
| 7 — Smart Crop Analysis | Detecta rostos e calcula enquadramento ideal | MediaPipe BlazeFace |
| 8 — Video Rendering | Renderiza vídeo vertical 9:16 com legendas embutidas | FFmpeg + libx264 |

Tudo roda **100% localmente**. Custo zero de API.

Todas as etapas possuem **cache em disco** — se o pipeline for interrompido, retoma de onde parou.

## Pré-requisitos

| Ferramenta | Versão mínima | Instalação |
|---|---|---|
| Python | 3.13+ | [python.org](https://www.python.org/downloads/) |
| uv | qualquer | `pip install uv` |
| Ollama | qualquer | [ollama.com](https://ollama.com/download) |

> FFmpeg é instalado automaticamente via `static-ffmpeg` — não é necessário instalá-lo no sistema.

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/podcast-cutter-ai.git
cd podcast-cutter-ai

# 2. Criar e ativar o ambiente virtual
uv sync

# Windows (Git Bash)
source .venv/Scripts/activate
# Linux/macOS
source .venv/bin/activate

# 3. Copiar e configurar variáveis de ambiente
cp .env.example .env
```

## Configuração do Ollama

```bash
# Baixar o modelo recomendado (feito apenas uma vez, ~4 GB)
ollama pull mistral

# Iniciar o servidor Ollama (manter rodando em segundo plano)
ollama serve
```

> O `mistral` é fortemente recomendado. Modelos menores (ex: `llama3.2`) tendem a gerar
> timestamps imprecisos e não seguem bem as instruções de duração.

## Uso

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Exemplos

```bash
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python main.py "https://youtu.be/dQw4w9WgXcQ"
python main.py "https://www.youtube.com/live/VIDEO_ID"
```

### Saída no terminal

```
══════════════════════════════════════════════════════════════
  Verificação pré-pipeline
══════════════════════════════════════════════════════════════
  ✓  Ollama        rodando  |  modelo=mistral
  ✓  FFmpeg        ok
  ✓  yt-dlp        ok
  ✓  MediaPipe     ok  |  smart crop ativo
  ⚠  cookies.txt   não configurado
  ✓  Espaço livre  45.3 GB
══════════════════════════════════════════════════════════════

── Stage 1/8: Audio Download        ✓  13.1s
── Stage 2/8: Transcription         ✓  199.4s
── Stage 3/8: Content Analysis      ✓  265.3s
── Stage 4/8: Video Download        ✓  83.3s
── Stage 5/8: Video Cutting         ✓  0.2s
── Stage 6/8: Subtitle Generation   ✓  0.1s
── Stage 7/8: Smart Crop Analysis   ✓  4.7s
── Stage 8/8: Video Rendering       ✓  42.1s

Detected 5 viral clip(s):
   1. [ 8.5/10]  1m 12s  —  O momento que viralizou
   2. [ 8.0/10]  1m 05s  —  A revelação surpreendente
   ...
```

### Saída em disco

```
storage/
├── audio/{video_id}/        ← Stage 1: áudio baixado
├── transcripts/{video_id}.json  ← Stage 2: transcrição
├── jobs/{video_id}.json     ← Stage 3: clips detectados
├── videos/{video_id}/       ← Stage 4: vídeo completo
├── cuts/{video_id}/         ← Stage 5: clips cortados
│   ├── cut_001.mp4
│   └── cut_002.mp4
├── subtitles/{video_id}/    ← Stage 6: legendas
│   ├── cut_001.srt
│   ├── cut_001.ass
│   └── cut_002.srt / .ass
└── rendered/{video_id}/     ← Stage 8: vídeos finais 9:16  ← OUTPUT FINAL
    ├── cut_001_final.mp4
    └── cut_002_final.mp4
```

## Resolução de problemas

### HTTP 403 Forbidden no download do vídeo

O YouTube bloqueia downloads sem autenticação. A solução é exportar os cookies do seu navegador:

1. Instale a extensão **[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)** no Brave/Chrome
2. Acesse `youtube.com` enquanto logado
3. Clique na extensão → **Export**
4. Salve o arquivo como `cookies.txt` na raiz do projeto
5. Garanta que o `.env` contém: `YTDLP_COOKIES_FILE=cookies.txt`

### O pipeline foi interrompido no meio

Basta rodar o mesmo comando novamente. Cada stage verifica o cache antes de executar:

- Audio já existe → pula Stage 1
- Transcrição já existe → pula Stage 2
- JSON de clips já existe → pula Stage 3
- Vídeo já existe → pula Stage 4
- Cuts já existem → pula individualmente no Stage 5
- Subtítulos já existem → pula individualmente no Stage 6
- Rendered já existe → pula individualmente no Stage 8

Para forçar a re-detecção de clips, delete o arquivo `storage/jobs/{video_id}.json`.

### Smart Crop não está funcionando

O Smart Crop requer o `mediapipe`. Se não estiver instalado, o pipeline usa crop central fixo.

```bash
# Verificar se mediapipe está instalado
python -c "import mediapipe; print(mediapipe.__version__)"

# Instalar se necessário
uv add mediapipe
```

O log mostrará `smart_crop=False` quando o fallback estiver ativo.

## Estrutura do projeto

```
podcast-cutter-ai/
├── main.py                              # Ponto de entrada CLI
├── pyproject.toml                       # Dependências + Ruff + pytest
├── .env.example                         # Template de configuração
│
├── app/
│   ├── config/
│   │   └── settings.py                  # Configurações centralizadas (Pydantic)
│   │
│   ├── pipeline/                        # Infraestrutura de orquestração
│   │   ├── context.py                   # PipelineContext — estado compartilhado
│   │   ├── stage.py                     # Stage ABC — contrato de toda etapa
│   │   └── orchestrator.py              # PipelineOrchestrator + create_default_pipeline()
│   │
│   ├── stages/                          # Adaptadores finos (1 por etapa)
│   │   ├── audio_download.py            # Stage 1 — AudioDownloadStage
│   │   ├── transcription.py             # Stage 2 — TranscriptionStage
│   │   ├── content_analysis.py          # Stage 3 — ContentAnalysisStage
│   │   ├── video_download.py            # Stage 4 — VideoDownloadStage
│   │   ├── video_cutting.py             # Stage 5 — VideoCuttingStage
│   │   ├── subtitle_generation.py       # Stage 6 — SubtitleGenerationStage
│   │   ├── smart_crop.py                # Stage 7 — SmartCropStage
│   │   └── rendering.py                 # Stage 8 — RenderingStage
│   │
│   ├── services/                        # Lógica de negócio pura (sem dependência de pipeline)
│   │   ├── downloader.py                # Download de áudio (yt-dlp)
│   │   ├── transcriber.py               # Transcrição (faster-whisper)
│   │   ├── chunker.py                   # Divisão de transcrição em chunks
│   │   ├── clip_detector.py             # Detecção de clips (Ollama LLM)
│   │   ├── video_downloader.py          # Download de vídeo (yt-dlp)
│   │   ├── video_cutter.py              # Corte de clips (FFmpeg)
│   │   ├── subtitle_service.py          # Geração de legendas SRT/ASS
│   │   ├── smart_crop.py                # Análise de rostos (MediaPipe)
│   │   └── video_renderer.py            # Renderização 9:16 (FFmpeg)
│   │
│   ├── schemas/                         # Modelos de dados Pydantic
│   │   ├── transcript.py                # Transcript, TranscriptSegment
│   │   ├── clip.py                      # ViralClip, ClipDetectionResult
│   │   └── crop.py                      # CropKeyframe, CropTimeline
│   │
│   ├── prompts/
│   │   └── clip_detection.py            # Prompts para o LLM (candidate + ranking)
│   │
│   ├── core/
│   │   ├── preflight.py                 # Verificação pré-pipeline de ferramentas
│   │   └── pipeline.py                  # Shim de retrocompatibilidade
│   │
│   └── utils/
│       ├── logger.py                    # Logging centralizado
│       └── validators.py                # Validação de URL do YouTube
│
├── tests/
│   ├── test_validators.py
│   ├── test_schemas.py
│   ├── test_chunker.py
│   └── test_clip_detector.py
│
├── storage/                             # Gerado em runtime — ignorado pelo git
│   ├── audio/, transcripts/, jobs/
│   ├── videos/, cuts/, subtitles/
│   └── rendered/
│
└── logs/                                # Logs de execução
```

## Arquitetura do pipeline

```
URL do YouTube
     │
     ▼
PreflightChecker          Verifica Ollama, FFmpeg, yt-dlp, MediaPipe, disco
     │
     ▼
PipelineOrchestrator      Executa stages em sequência, controla erros e logs
     │
     ├─► AudioDownloadStage      → ctx.audio_path
     │
     ├─► TranscriptionStage      → ctx.transcript
     │
     ├─► ContentAnalysisStage    → ctx.clip_result
     │
     ├─► VideoDownloadStage      → ctx.video_path
     │
     ├─► VideoCuttingStage       → ctx.cut_paths
     │
     ├─► SubtitleGenerationStage → ctx.subtitle_pairs
     │
     ├─► SmartCropStage          → ctx.crop_timelines
     │
     └─► RenderingStage          → ctx.rendered_paths
                                        │
                                        ▼
                              storage/rendered/{video_id}/
                              cut_001_final.mp4  ← OUTPUT FINAL
```

### Como adicionar uma nova etapa

1. Crie `app/stages/minha_etapa.py` implementando `Stage`:

```python
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.meu_servico import MeuServico

class MinhaEtapaStage(Stage):
    def __init__(self, settings):
        self._service = MeuServico(settings)

    @property
    def name(self) -> str:
        return "Minha Etapa"

    def run(self, ctx: PipelineContext) -> None:
        entrada = ctx.require_clip_result()         # lê do contexto
        ctx.meu_campo = self._service.processar(entrada)  # escreve no contexto
```

2. Adicione o campo em `app/pipeline/context.py`:

```python
meu_campo: MeuTipo | None = None
```

3. Registre a etapa em `create_default_pipeline()` dentro de `app/pipeline/orchestrator.py`.

Nenhum outro arquivo precisa ser modificado.

## Configuração avançada

Todas as opções são configuradas via `.env`:

```env
# ── Whisper ───────────────────────────────────────────────────────────────────
# Modelos: tiny | base | small | medium | large-v2 | large-v3
# Recomendado para CPU: base ou small
WHISPER_MODEL=base
WHISPER_DEVICE=cpu           # cpu | cuda
WHISPER_COMPUTE_TYPE=int8    # int8 | float16 | float32
WHISPER_LANGUAGE=            # Forçar idioma (ex: "pt"). Vazio = autodetect

# ── Vídeo ─────────────────────────────────────────────────────────────────────
VIDEO_QUALITY=720            # Resolução máxima: 360 | 480 | 720 | 1080

# ── yt-dlp (fix para HTTP 403) ────────────────────────────────────────────────
YTDLP_COOKIES_FILE=cookies.txt   # Arquivo de cookies exportado do navegador
YTDLP_COOKIE_BROWSER=            # Alternativa: extrair direto do browser (Firefox)

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL=mistral         # Modelo LLM (recomendado: mistral)
OLLAMA_TEMPERATURE=0.3       # 0.0 = determinístico, 1.0 = criativo
OLLAMA_TIMEOUT_SECONDS=600   # Timeout por chamada ao LLM

# ── Detecção de clips ─────────────────────────────────────────────────────────
MIN_CLIP_DURATION=60         # Duração mínima em segundos (60s para monetização)
MAX_CLIP_DURATION=120        # Duração máxima em segundos
MAX_CLIPS=10                 # Número máximo de clips por vídeo

# ── Renderização ──────────────────────────────────────────────────────────────
RENDER_CRF=23                # Qualidade H.264: 18=quase-lossless, 28=menor arquivo
RENDER_PRESET=fast           # Velocidade de encode: ultrafast|fast|medium|slow
```

## Desenvolvimento

```bash
# Verificar qualidade do código
ruff check app/ main.py

# Formatar código
ruff format app/ main.py

# Rodar testes
pytest

# Rodar testes com cobertura
pytest --cov=app --cov-report=term-missing

# Gerar relatório HTML de cobertura (abre em browser)
pytest --cov=app --cov-report=html
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS/Linux
```

O CI/CD roda automaticamente no GitHub Actions a cada push nas branches `main` e `develop`, executando lint e testes.

## Roadmap

- [x] Stage 1: download de áudio
- [x] Stage 2: transcrição local com timestamps
- [x] Stage 3: detecção de momentos virais via LLM
- [x] Stage 4: download do vídeo completo
- [x] Stage 5: geração automática dos cortes em MP4
- [x] Stage 6: geração de legendas SRT e ASS
- [x] Stage 7: smart crop com detecção de rostos (MediaPipe)
- [x] Stage 8: renderização vertical 9:16 com legendas embutidas
- [x] Verificação pré-pipeline de ferramentas (preflight checks)
- [x] Arquitetura de pipeline por etapas independentes
- [ ] Legendas animadas (efeitos ASS)
- [ ] API REST (FastAPI)
- [ ] Processamento assíncrono com filas
- [ ] Interface web
