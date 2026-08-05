# Private Journal Digitizer

Turn photographs of your handwritten journal into searchable Markdown notes —
**entirely on your own machine**.

No cloud. No Google, Anthropic, Microsoft, or any other company ever sees your
pages. Your photos, your transcripts, and the AI model that reads them all stay
local.

## Why this project?

Most people have never been told that a private option exists. The common path
for "digitizing my journals" is uploading every page to a cloud service — a
sacrifice most people don't really want to make for their most personal writing.

Local AI has gotten good enough that this is no longer necessary:

- A small open-weights vision model reads handwriting well, including print.
- Your photos never leave the computer.
- You keep the raw images, the transcripts, and the model files yourself.

This project is a working, minimal scaffold to do exactly that. It is built to
be readable and easy to modify — one Python file, standard library only.

## What it does (the fix)

- Walks a folder of journal page photos (`jpg`/`jpeg`/`png`).
- Rotates and downscales each photo, then sends it to a **local** vision model
  server running on your machine.
- The model transcribes the page (verbatim style, not a summary).
- A date is inferred from the page's own text and used in the note filename
  (e.g. `2024-05-15 page-027.md`); pages without a trustworthy date fall back to
  `page-NNN.md` and are left for your review.
- Unreadable pages are flagged `needs-review` instead of silently guessing.
- Writes Markdown notes with YAML frontmatter into your Obsidian vault and
  builds an `_Index.md`.

## How it works

```
photos/*.jpg ──> ffmpeg (rotate + downscale) ──> base64 image
                                                     │
                                                     ▼
                                  local vision server (llama.cpp)
                                  OpenAI-compatible /v1/chat/completions
                                                     │
                                                     ▼
                              Markdown note + frontmatter in your vault
```

The script only ever talks to the URL in `vision_base` (default
`http://127.0.0.1:9011` — your own machine).

## Requirements

- **Python 3.11+** (stdlib only — no pip packages needed)
- **ffmpeg** (image rotate + downscale)
- A **local vision model server**. This project was built against
  [llama.cpp](https://github.com/ggerganov/llama.cpp) with a
  Qwen2.5-VL model. Any OpenAI-compatible endpoint works.

## Quick start

### 1. Download a model

Recommended (best handwriting quality, needs a decent GPU):

```bash
./download_model.sh 7b          # downloads to ./models
```

On a smaller / CPU-only machine (lighter, weaker on messy handwriting):

```bash
./download_model.sh 3b
```

No `hf` CLI? Download the two files directly:

| Model | GGUF (weights) | Vision projector |
|---|---|---|
| 7B (recommended) | [Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf](https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf) | [mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf) |
| 3B (light) | [Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf) | [mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf) |

Both are **Apache-2.0** and run locally.

### 2. Start the vision server

With Docker (what this project is developed against):

```bash
docker run -d --name vision --gpus all \
  -p 9011:9011 \
  -v "$PWD/models:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --mmproj /models/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf \
  -ngl 999 -c 32768 --host 0.0.0.0 --port 9011
```

Or with a pre-built `llama-server` binary from the
[llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases), run
the same flags directly. Any server that speaks the OpenAI chat API and accepts
images will work — just point `vision_base` at it.

### 3. Configure and run

```bash
cp config.example.json config.json
# edit config.json: set images_root (your photos) and vault_root (your Obsidian vault)

python3 digitize_journal.py --check   # pre-flight: ffmpeg + config + server
python3 digitize_journal.py           # digitize everything
```

Output notes land in `<vault_root>/<journal_subdir>/...` (default `Journal/`),
mirroring your photo folder layout, plus an `_Index.md`.

## Choosing a model (tradeoffs)

| | 7B (recommended) | 3B (light) |
|---|---|---|
| Files | ~4.7 GB + 853 MB projector | ~1.9 GB + projector |
| Needs | ~8-10 GB VRAM, or a patient CPU | runs on much less, incl. CPU |
| Handwriting | best quality, handles messy print | good but noticeably weaker on messy pages |
| Speed | faster per page | slower if CPU-bound, but fits small GPUs |

At any size, **review is recommended**. These models are a working scaffold —
they massively cut the time to digitize a journal, but they are not a guarantee
of perfect verbatim transcription. Pages the model itself doubts are flagged
`needs-review` so you know exactly where to look.

## What you get

Each page becomes a note like:

```markdown
---
date: "2024-05-15"
source: "2024/May/IMG_1234.jpg"
page: 27
folder: 2024/May
status: transcribed
tags: journal
---

May 15th - the garden is finally coming up. The tomatoes are about a foot tall...
```

- `status: needs-review` marks illegible or suspicious pages.
- Dates come from the page's own text (with conservative rules); untrusted
  dates become plain `page-NNN.md` instead of being guessed wrong.
- Re-running is safe: pages already digitized are skipped.

## Configuration

| Key | Meaning |
|---|---|
| `images_root` | folder tree containing your journal photos |
| `vault_root` | root of your Obsidian vault (any folder works) |
| `journal_subdir` | subfolder in the vault where notes are written |
| `output_prefix` | optional extra subfolder under `journal_subdir` |
| `vision_base` | URL of your local vision server |
| `model` | model name reported to the server |
| `ffmpeg` | `ffmpeg` binary name/path (default: on PATH) |
| `year_range` | `[min, max]` years to accept in dates |
| `max_image_px` | downscale photos to this max dimension |
| `max_tokens` | response length limit per page |
| `timeout_s` | per-request timeout |
| `temp_dir` / `log_dir` / `log_file` | scratch space and log location |
| `write_index` | build `_Index.md` after processing |

## Privacy

- Every request goes to `vision_base` — localhost by default. There is **no**
  cloud component, telemetry, or account.
- The model files are open-weights and run on your hardware.
- Your `config.json` contains your private paths — it is git-ignored and should
  never be committed.

## Limitations

- Not a perfect verbatim transcription system. Legibility, ink quality, and
  model size all matter; proofread flagged pages.
- Date inference is heuristic by design — when in doubt, no date is assigned
  rather than a wrong one.
- One request per page with no chat context; extremely long or dense pages may
  be truncated (the log reports this).

## License

[MIT](LICENSE). The recommended models are separately licensed **Apache-2.0**.
