# MangaForge Space

MangaForge is a **Hugging Face Spaces-ready** comic generator that prioritizes:

1. high-quality anime/manga/comic visuals,
2. strict reference-image adherence,
3. cross-panel continuity,
4. readable comic page layout,
5. deployment readiness,
6. low local compute by using remote inference APIs.

## Why this architecture

Free Space CPU/GPU tiers are not reliable for heavy local diffusion pipelines. This app runs planning and image generation through hosted APIs and uses local compute only for lightweight composition.

- **Planning (continuity + panel writing):** OpenAI Chat Completions (`OPENAI_API_KEY`)
- **Panel rendering:**
  - Primary: OpenAI image generation (`gpt-image-1`) with uploaded reference images.
  - Optional fallback: Hugging Face Inference API text-to-image (no strong reference adherence).
- **Page assembly:** Pillow only.

## Features

- Character bible + style guide injection into every panel prompt.
- Reference images sent to image generation API for better visual adherence.
- Panel-by-panel continuity memory (each panel receives previous panel summary).
- Readable layout with panel borders, captions, and speech text blocks.
- Exports page PNGs and a downloadable ZIP bundle.

## Limitations (explicit)

- On free Space hardware, this app cannot run state-of-the-art local diffusion + control models at production quality.
- **Strongest practical path is API-backed generation.**
- Hugging Face Inference fallback may reduce reference-image adherence versus OpenAI image APIs.

## Hugging Face Spaces setup

### `README` metadata (for Spaces)

This repository includes a compatible Gradio app entrypoint (`app.py`).

### Environment variables

Set in Space Settings → Variables:

- `OPENAI_API_KEY` (recommended / primary)
- `OPENAI_TEXT_MODEL` (optional, default `gpt-4.1-mini`)
- `OPENAI_IMAGE_MODEL` (optional, default `gpt-image-1`)
- `HF_TOKEN` (optional fallback provider)

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Deployment notes

- This project is intentionally API-first for reliability on constrained hardware.
- If budgets are strict, tune panel count and image size down in the UI.
