# Podcast Cutter AI

Pipeline local para detecção e geração automática de cortes virais em podcasts e lives do YouTube.

## O que faz

Dado uma URL do YouTube, o pipeline executa 5 etapas automaticamente:

| Step | Descrição | Tecnologia |
|---|---|---|
| 1 — Audio Download | Baixa apenas o áudio (leve, ~180 MB para 3h) | yt-dlp |
| 2 — Transcription | Transcreve localmente com timestamps | faster-whisper |
| 3 — Clip Detection | LLM lê a transcrição e identifica momentos virais | Mistral via Ollama |
| 4 — Video Download | Baixa o vídeo completo em 720p somente após confirmar clips | yt-dlp |
| 5 — Clip Cutting | Corta cada clip detectado como arquivo MP4 independente | FFmpeg |

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

### Saída

```
── Step 1/5: Audio Download     ✓  13.1s
── Step 2/5: Transcription      ✓  199.4s
── Step 3/5: Clip Detection     ✓  265.3s
── Step 4/5: Video Download     ✓  83.3s
── Step 5/5: Clip Cutting       ✓  0.2s

Detected 5 viral clip(s):
   1. [ 8.5/10]  1m 12s  —  O momento que viralizou
   2. [ 8.0/10]  1m 05s  —  A revelação surpreendente
   ...

5 cut(s) saved to: storage/cuts/VIDEO_ID/
Results saved to: storage/jobs/VIDEO_ID.json
```

Os arquivos de corte ficam em `storage/cuts/{video_id}/`:

```
storage/cuts/dQw4w9WgXcQ/
├── cut_001.mp4   ← clip com maior viral_score
├── cut_002.mp4
└── cut_003.mp4
```

O JSON de detecção fica em `storage/jobs/{video_id}.json`:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://youtu.be/dQw4w9WgXcQ",
  "clips": [
    {
      "title": "O momento mais impactante do episódio",
      "start_time": 312.5,
      "end_time": 372.5,
      "justification": "Revelação surpreendente com alta carga emocional.",
      "viral_score": 9.2
    }
  ]
}
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

Basta rodar o mesmo comando novamente. Cada etapa verifica o cache antes de executar:
- Audio já existe → pula Step 1
- Transcrição já existe → pula Step 2
- JSON de clips já existe → pula Step 3
- Vídeo já existe → pula Step 4
- Cuts já existem → pula individualmente no Step 5

Para forçar a re-detecção de clips, delete o arquivo `storage/jobs/{video_id}.json`.

## Estrutura do projeto

```
podcast-cutter-ai/
├── main.py                          # Ponto de entrada
├── pyproject.toml                   # Dependências + Ruff + pytest
├── .env.example                     # Template de configuração
│
├── app/
│   ├── config/
│   │   └── settings.py              # Configurações centralizadas (Pydantic)
│   ├── core/
│   │   └── pipeline.py              # Orquestrador dos 5 steps
│   ├── services/
│   │   ├── downloader.py            # Step 1: download de áudio
│   │   ├── transcriber.py           # Step 2: transcrição (faster-whisper)
│   │   ├── chunker.py               # Divisão da transcrição em chunks
│   │   ├── clip_detector.py         # Step 3: detecção de clips (Ollama)
│   │   ├── video_downloader.py      # Step 4: download de vídeo
│   │   └── video_cutter.py          # Step 5: corte dos clips (FFmpeg)
│   ├── schemas/
│   │   ├── transcript.py            # Modelos de transcrição
│   │   └── clip.py                  # Modelos de clips virais
│   ├── prompts/
│   │   └── clip_detection.py        # Prompts para o LLM
│   └── utils/
│       ├── logger.py                # Logging centralizado
│       └── validators.py            # Validação de URL do YouTube
│
├── storage/
│   ├── audio/                       # Áudios baixados (Step 1)
│   ├── transcripts/                 # Transcrições JSON (Step 2)
│   ├── jobs/                        # Clips detectados JSON (Step 3)
│   ├── videos/                      # Vídeos baixados (Step 4)
│   └── cuts/                        # Clips cortados MP4 (Step 5) ← OUTPUT FINAL
│
└── logs/                            # Logs de execução
```

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
```

## Desenvolvimento

```bash
# Verificar qualidade do código
ruff check app/ main.py

# Formatar código
ruff format app/ main.py

# Rodar testes
pytest
```

## Roadmap

- [x] Step 1: download de áudio
- [x] Step 2: transcrição local com timestamps
- [x] Step 3: detecção de momentos virais via LLM
- [x] Step 4: download do vídeo completo
- [x] Step 5: geração automática dos cortes em MP4
- [ ] Legendas animadas sobrepostas nos cortes
- [ ] Renderização vertical 9:16 (Reels / Shorts / TikTok)
- [ ] API REST (FastAPI)
- [ ] Processamento assíncrono com filas
- [ ] Interface web
