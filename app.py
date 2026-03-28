import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr

from comic_pipeline import generate_comic


def _run(
    premise: str,
    style: str,
    character_bible: str,
    continuity_focus: str,
    panel_count: int,
    provider: str,
    size: str,
    refs: List[str],
) -> Tuple[List[str], str, str, str]:
    with tempfile.TemporaryDirectory(prefix="mangaforge_") as tmp:
        run_dir = Path(tmp)
        result = generate_comic(
            premise=premise,
            style=style,
            character_bible=character_bible,
            continuity_focus=continuity_focus,
            panel_count=panel_count,
            provider=provider,
            reference_paths=refs or [],
            size=size,
            run_dir=run_dir,
        )

        panels = result["panel_images"]
        page = result["page_image"]
        bundle = result["bundle_zip"]
        manifest = result["manifest"]

        persistent_dir = Path("/tmp/mangaforge_last")
        persistent_dir.mkdir(parents=True, exist_ok=True)

        copied_panels = []
        for p in panels:
            src = Path(p)
            dst = persistent_dir / src.name
            dst.write_bytes(src.read_bytes())
            copied_panels.append(str(dst))

        dst_page = persistent_dir / Path(page).name
        dst_page.write_bytes(Path(page).read_bytes())
        dst_bundle = persistent_dir / Path(bundle).name
        dst_bundle.write_bytes(Path(bundle).read_bytes())
        dst_manifest = persistent_dir / Path(manifest).name
        dst_manifest.write_bytes(Path(manifest).read_bytes())

        return copied_panels, str(dst_page), str(dst_bundle), str(dst_manifest)


DESCRIPTION = """
# MangaForge (API-first for Space hardware)

Generate anime/manga/comic pages with **reference-image adherence and panel continuity**.

### Practical deployment choice
- Strongest output quality on free/low-tier Hugging Face Spaces comes from hosted APIs.
- This app uses remote model inference and keeps local compute lightweight.

### Provider guidance
- **openai** (recommended): best reference-image adherence in this app.
- **huggingface**: fallback when OpenAI image API is unavailable; weaker reference conditioning.
"""

with gr.Blocks(title="MangaForge") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        premise = gr.Textbox(
            label="Story premise",
            value="A cyberpunk ronin reunites with her lost brother in a rain-soaked neon market.",
            lines=3,
        )
        style = gr.Textbox(
            label="Visual style",
            value="High-detail seinen manga, dynamic camera angles, dramatic screentones, cinematic lighting",
            lines=3,
        )

    character_bible = gr.Textbox(
        label="Character bible (appearance anchors + outfit rules + prop continuity)",
        value=(
            "Aiko: short silver bob, amber eyes, red scarf, black long-coat with white kanji crest. "
            "Kenji: taller, dark braided hair, prosthetic left arm with blue glow seams."
        ),
        lines=4,
    )

    continuity_focus = gr.Textbox(
        label="Continuity focus",
        value="Keep outfits, scars, accessories, and weather consistent across all panels.",
        lines=2,
    )

    with gr.Row():
        panel_count = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Panel count")
        provider = gr.Dropdown(
            choices=["openai", "huggingface"],
            value="openai",
            label="Image provider",
        )
        size = gr.Dropdown(
            choices=["1024x1024", "1536x1024", "1024x1536"],
            value="1024x1024",
            label="Panel resolution",
        )

    refs = gr.File(label="Reference images (character sheets, moodboards, prop refs)", file_count="multiple")

    run = gr.Button("Generate comic bundle", variant="primary")

    panel_gallery = gr.Gallery(label="Generated panels", columns=4, rows=2, object_fit="cover")
    page_image = gr.Image(label="Composed comic page", type="filepath")
    bundle = gr.File(label="Download bundle ZIP")
    manifest = gr.File(label="Generation manifest JSON")

    run.click(
        _run,
        inputs=[premise, style, character_bible, continuity_focus, panel_count, provider, size, refs],
        outputs=[panel_gallery, page_image, bundle, manifest],
        api_name="generate_comic",
    )


if __name__ == "__main__":
    demo.launch()
