import base64
import io
import json
import os
import re
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


@dataclass
class PanelSpec:
    panel_number: int
    shot: str
    visual_prompt: str
    narration: str
    dialogue: List[str]
    continuity_notes: str


class ScriptPlanner:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def plan_panels(
        self,
        premise: str,
        style: str,
        character_bible: str,
        panel_count: int,
        continuity_focus: str,
    ) -> List[PanelSpec]:
        system = (
            "You are a manga/comic director. Return strict JSON with key `panels` only. "
            "Each panel has: panel_number(int), shot, visual_prompt, narration, dialogue(array), continuity_notes. "
            "Prompts must emphasize anime/manga composition, readable staging, and continuity."
        )
        user = {
            "premise": premise,
            "style": style,
            "character_bible": character_bible,
            "panel_count": panel_count,
            "continuity_focus": continuity_focus,
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        panels = payload.get("panels", [])
        results: List[PanelSpec] = []
        for i, p in enumerate(panels[:panel_count], start=1):
            results.append(
                PanelSpec(
                    panel_number=int(p.get("panel_number", i)),
                    shot=str(p.get("shot", "medium shot")),
                    visual_prompt=str(p.get("visual_prompt", "")),
                    narration=str(p.get("narration", "")),
                    dialogue=[str(x) for x in p.get("dialogue", [])][:3],
                    continuity_notes=str(p.get("continuity_notes", "")),
                )
            )
        if not results:
            raise RuntimeError("Planner returned no panels.")
        return results


class ImageGenerator:
    def __init__(self, provider: str, openai_client: Optional[OpenAI], image_model: str):
        self.provider = provider
        self.openai_client = openai_client
        self.image_model = image_model

    def _encode_path(self, path: Path) -> str:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def generate(
        self,
        prompt: str,
        reference_images: Sequence[Path],
        output_path: Path,
        size: str,
    ) -> Path:
        if self.provider == "openai":
            if not self.openai_client:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI image provider.")
            self._generate_openai(prompt, reference_images, output_path, size)
            return output_path
        if self.provider == "huggingface":
            self._generate_hf(prompt, output_path)
            return output_path
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def _generate_openai(
        self,
        prompt: str,
        reference_images: Sequence[Path],
        output_path: Path,
        size: str,
    ) -> None:
        kwargs: Dict[str, Any] = {
            "model": self.image_model,
            "prompt": prompt,
            "size": size,
        }
        image_data = []
        for path in reference_images[:4]:
            image_data.append(self._encode_path(path))
        if image_data:
            kwargs["image"] = image_data

        result = self.openai_client.images.generate(**kwargs)
        b64 = result.data[0].b64_json
        output_path.write_bytes(base64.b64decode(b64))

    def _generate_hf(self, prompt: str, output_path: Path) -> None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for Hugging Face provider.")
        endpoint = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt},
            timeout=180,
        )
        if response.status_code != 200:
            raise RuntimeError(f"HF API failed: {response.status_code} {response.text[:200]}")
        output_path.write_bytes(response.content)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return text[:80] or "comic"


def ensure_image(path: Path) -> None:
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Invalid image generated for {path.name}: {exc}") from exc


def make_layout(
    panels: Sequence[PanelSpec],
    panel_images: Sequence[Path],
    output_page: Path,
    columns: int = 2,
    panel_size: Tuple[int, int] = (1024, 1024),
) -> Path:
    rows = (len(panel_images) + columns - 1) // columns
    margin = 24
    title_h = 80
    bubble_h = 180
    panel_w, panel_h = panel_size
    page_w = margin + columns * (panel_w + margin)
    page_h = title_h + margin + rows * (panel_h + bubble_h + margin) + margin

    page = Image.new("RGB", (page_w, page_h), (250, 250, 250))
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()

    draw.text((margin, 26), "MangaForge Generated Page", fill=(20, 20, 20), font=font)

    for idx, (spec, image_path) in enumerate(zip(panels, panel_images)):
        r = idx // columns
        c = idx % columns
        x = margin + c * (panel_w + margin)
        y = title_h + margin + r * (panel_h + bubble_h + margin)

        with Image.open(image_path) as panel_img:
            panel_img = panel_img.convert("RGB").resize((panel_w, panel_h))
            page.paste(panel_img, (x, y))

        draw.rectangle([x - 2, y - 2, x + panel_w + 2, y + panel_h + 2], outline=(0, 0, 0), width=3)

        caption = textwrap.fill(spec.narration or "", width=70)
        speech = "\n".join([f"• {d}" for d in spec.dialogue])
        bubble_text = f"Panel {spec.panel_number}: {spec.shot}\n{caption}\n{speech}".strip()
        bubble_text = textwrap.shorten(bubble_text, width=420, placeholder="...")
        draw.rectangle(
            [x, y + panel_h + 10, x + panel_w, y + panel_h + bubble_h - 10],
            outline=(40, 40, 40),
            width=2,
            fill=(255, 255, 255),
        )
        draw.multiline_text((x + 10, y + panel_h + 20), bubble_text, fill=(20, 20, 20), font=font, spacing=4)

    page.save(output_page, format="PNG")
    return output_page


def generate_comic(
    premise: str,
    style: str,
    character_bible: str,
    continuity_focus: str,
    panel_count: int,
    provider: str,
    reference_paths: Sequence[str],
    size: str,
    run_dir: Path,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    text_model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
    image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY is required for script planning.")

    planner = ScriptPlanner(openai_client, text_model)
    image_gen = ImageGenerator(provider=provider, openai_client=openai_client, image_model=image_model)

    panels = planner.plan_panels(
        premise=premise,
        style=style,
        character_bible=character_bible,
        panel_count=panel_count,
        continuity_focus=continuity_focus,
    )

    ref_paths = [Path(p) for p in reference_paths if p]
    generated_images: List[Path] = []

    for i, panel in enumerate(panels, start=1):
        prev_context = panels[i - 2].continuity_notes if i > 1 else "Opening panel"
        prompt = (
            f"{style}\n"
            f"Character bible:\n{character_bible}\n"
            f"Continuity focus: {continuity_focus}\n"
            f"Previous panel continuity notes: {prev_context}\n"
            f"Current panel shot: {panel.shot}\n"
            f"Visual direction: {panel.visual_prompt}\n"
            "Render as polished manga/comic panel with clean linework, clear silhouettes,"
            " cinematic composition, legible foreground/background separation, and coherent anatomy."
        )
        out = run_dir / f"panel_{i:02d}.png"

        panel_refs: List[Path] = list(ref_paths)
        if generated_images:
            panel_refs.append(generated_images[-1])

        image_gen.generate(prompt=prompt, reference_images=panel_refs, output_path=out, size=size)
        ensure_image(out)
        generated_images.append(out)

    page = run_dir / "page_01.png"
    make_layout(panels, generated_images, page)

    manifest = run_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "panels": [p.__dict__ for p in panels],
                "provider": provider,
                "style": style,
                "size": size,
                "limitations": [
                    "API-backed generation is used to avoid heavy local diffusion compute.",
                    "Hugging Face fallback may have weaker reference-image adherence.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    zip_path = run_dir / f"{sanitize_filename(premise)}_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img in generated_images:
            zf.write(img, arcname=img.name)
        zf.write(page, arcname=page.name)
        zf.write(manifest, arcname=manifest.name)

    return {
        "panel_images": [str(p) for p in generated_images],
        "page_image": str(page),
        "bundle_zip": str(zip_path),
        "manifest": str(manifest),
    }
