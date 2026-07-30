"""Pretrained VQ-token autoregressive image generation experiments.

The important compatibility rule is simple: the autoregressive transformer must
have been trained on the same discrete image tokenizer/codebook.  Pretrained
pixel ImageGPT checkpoints are useful controls, but they are not drop-in heads
for VQGAN or ViT-VQGAN code IDs.

This script therefore supports two paths:

* ``--mode compatibility`` writes a short model-pair report.
* ``--mode llamagen-c2i`` samples from FoundationVision/LlamaGen, a modern
  matched VQ tokenizer + GPT-style class-conditional AR transformer pair.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PairCandidate:
    name: str
    tokenizer: str
    autoregressor: str
    compatible: bool
    reason: str
    source: str


PAIR_CANDIDATES = [
    PairCandidate(
        name="llamagen-c2i",
        tokenizer="FoundationVision/LlamaGen vq_ds16_c2i tokenizer",
        autoregressor="FoundationVision/LlamaGen c2i GPT checkpoints",
        compatible=True,
        reason="Released tokenizer and AR checkpoints are trained as a matched VQ-code pair.",
        source="https://github.com/FoundationVision/LlamaGen",
    ),
    PairCandidate(
        name="taming-transformers-cin",
        tokenizer="CompVis VQGAN ImageNet f=16 tokenizer",
        autoregressor="CompVis class-conditional ImageNet transformer",
        compatible=True,
        reason="Original Taming Transformers checkpoints use a shared VQGAN codebook and AR transformer.",
        source="https://github.com/CompVis/taming-transformers",
    ),
    PairCandidate(
        name="vqgan-plus-openai-imagegpt",
        tokenizer="VQGAN / ViT-VQGAN codebook IDs",
        autoregressor="OpenAI ImageGPT pixel-token checkpoints",
        compatible=False,
        reason="ImageGPT checkpoints model pixel/color-cluster tokens, not learned VQGAN code IDs.",
        source="https://huggingface.co/openai/imagegpt-small",
    ),
    PairCandidate(
        name="vit-vqgan-vim",
        tokenizer="Improved ViT-VQGAN tokenizer",
        autoregressor="VIM autoregressive transformer",
        compatible=True,
        reason="This is the paper-clean ViT-VQGAN + ImageGPT-like pairing, but public checkpoints are less plug-and-play than LlamaGen.",
        source="https://arxiv.org/abs/2110.04627",
    ),
]


LLAMAGEN_PROFILES: dict[str, dict[str, Any]] = {
    "c2i-B-256": {
        "repo_id": "FoundationVision/LlamaGen",
        "vq_file": "vq_ds16_c2i.pt",
        "gpt_file": "c2i_B_256.pt",
        "vq_model": "VQ-16",
        "gpt_model": "GPT-B",
        "image_size": 256,
        "downsample_size": 16,
        "codebook_size": 16384,
        "codebook_embed_dim": 8,
        "params": "111M AR + 72M tokenizer",
        "fid": 5.46,
    },
    "c2i-B-384": {
        "repo_id": "FoundationVision/LlamaGen",
        "vq_file": "vq_ds16_c2i.pt",
        "gpt_file": "c2i_B_384.pt",
        "vq_model": "VQ-16",
        "gpt_model": "GPT-B",
        "image_size": 384,
        "downsample_size": 16,
        "codebook_size": 16384,
        "codebook_embed_dim": 8,
        "params": "111M AR + 72M tokenizer",
        "fid": 6.09,
    },
    "c2i-L-256": {
        "repo_id": "FoundationVision/LlamaGen",
        "vq_file": "vq_ds16_c2i.pt",
        "gpt_file": "c2i_L_256.pt",
        "vq_model": "VQ-16",
        "gpt_model": "GPT-L",
        "image_size": 256,
        "downsample_size": 16,
        "codebook_size": 16384,
        "codebook_embed_dim": 8,
        "params": "343M AR + 72M tokenizer",
        "fid": 3.80,
    },
    "c2i-L-384": {
        "repo_id": "FoundationVision/LlamaGen",
        "vq_file": "vq_ds16_c2i.pt",
        "gpt_file": "c2i_L_384.pt",
        "vq_model": "VQ-16",
        "gpt_model": "GPT-L",
        "image_size": 384,
        "downsample_size": 16,
        "codebook_size": 16384,
        "codebook_embed_dim": 8,
        "params": "343M AR + 72M tokenizer",
        "fid": 3.07,
    },
    "c2i-XL-384": {
        "repo_id": "FoundationVision/LlamaGen",
        "vq_file": "vq_ds16_c2i.pt",
        "gpt_file": "c2i_X_384L.pt",
        "vq_model": "VQ-16",
        "gpt_model": "GPT-XL",
        "image_size": 384,
        "downsample_size": 16,
        "codebook_size": 16384,
        "codebook_embed_dim": 8,
        "params": "775M AR + 72M tokenizer",
        "fid": 2.62,
    },
}


def parse_class_labels(text: str, *, samples: int, seed: int = 0) -> list[int]:
    raw = str(text).strip()
    if raw.lower() == "random":
        rng = np.random.default_rng(int(seed))
        return [int(x) for x in rng.integers(0, 1000, size=max(1, int(samples))).tolist()]
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("class labels cannot be empty")
    out: list[int] = []
    while len(out) < int(samples):
        out.extend(values)
    return out[: int(samples)]


def resolve_device(text: str) -> torch.device:
    value = str(text).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_precision(text: str, device: torch.device) -> torch.dtype:
    value = str(text).lower()
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "none": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[value]


def resolve_llamagen_repo(path: str | None = None) -> Path:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("LLAMAGEN_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([REPO_ROOT / "external" / "LlamaGen", Path.cwd() / "external" / "LlamaGen"])
    for candidate in candidates:
        if (candidate / "autoregressive" / "models" / "gpt.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find LlamaGen repo. Clone https://github.com/FoundationVision/LlamaGen "
        "to external/LlamaGen or pass --llamagen-repo. You can also use --auto-clone."
    )


def maybe_clone_llamagen(path: Path) -> None:
    if (path / "autoregressive" / "models" / "gpt.py").exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/FoundationVision/LlamaGen", str(path)],
        check=True,
    )


@contextlib.contextmanager
def llamagen_import_context(repo_path: Path) -> Iterator[None]:
    old_path = list(sys.path)
    repo_str = str(repo_path)
    src_str = str(REPO_ROOT / "src")
    saved_utils = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "utils" or name.startswith("utils.")
    }
    for name in saved_utils:
        sys.modules.pop(name, None)
    filtered_path: list[str] = []
    for path in sys.path:
        if not path:
            filtered_path.append(path)
            continue
        try:
            if str(Path(path).resolve()) == src_str:
                continue
        except OSError:
            pass
        filtered_path.append(path)
    sys.path[:] = [repo_str] + filtered_path
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_utils)
        sys.path[:] = old_path


def load_weight_payload(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "module", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise ValueError(f"Unsupported checkpoint payload in {path}")


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu()
    if float(image.min()) < -0.05:
        image = image.add(1.0).mul(0.5)
    image = image.clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def save_grid(images: torch.Tensor, path: Path, *, labels: list[int] | None = None, title: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(images.shape[0])
    cols = min(4, max(1, n))
    rows = int(math.ceil(n / cols))
    pil_images = [tensor_to_pil(images[i]) for i in range(n)]
    width, height = pil_images[0].size
    title_h = 36 if title else 0
    label_h = 24 if labels else 0
    canvas = Image.new("RGB", (cols * width, title_h + rows * (height + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = str(Path(ImageFont.__file__).with_name("DejaVuSans.ttf"))
        font = ImageFont.truetype(font_path, 16)
        title_font = ImageFont.truetype(font_path, 22)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    if title:
        draw.text((8, 6), title, fill=(20, 20, 20), font=title_font)
    for idx, pil in enumerate(pil_images):
        row, col = divmod(idx, cols)
        x = col * width
        y = title_h + row * (height + label_h)
        canvas.paste(pil, (x, y))
        if labels:
            draw.text((x + 6, y + height + 3), f"class {labels[idx]}", fill=(35, 35, 35), font=font)
    canvas.save(path)
    return path


def build_compatibility_report() -> dict[str, Any]:
    return {
        "summary": {
            "recommended_first_run": "llamagen-c2i",
            "why": "It is a pretrained matched VQ tokenizer plus AR transformer pair, so the code IDs have the right semantics.",
        },
        "pairs": [asdict(candidate) for candidate in PAIR_CANDIDATES],
        "llamagen_profiles": LLAMAGEN_PROFILES,
    }


def write_compatibility_report(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "vq_ar_compatibility_report.json"
    md_path = out_dir / "vq_ar_compatibility_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Pretrained VQ Token + Autoregressive Model Compatibility",
        "",
        "Use a matched tokenizer and autoregressive checkpoint whenever possible.",
        "",
        "| Pair | Compatible | Tokenizer | Autoregressor | Note |",
        "|---|---:|---|---|---|",
    ]
    for row in report["pairs"]:
        lines.append(
            f"| {row['name']} | {'yes' if row['compatible'] else 'no'} | "
            f"{row['tokenizer']} | {row['autoregressor']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "Recommended first concrete run: `llamagen-c2i` with `c2i-B-256`.",
            "The `vqgan-plus-openai-imagegpt` row is intentionally marked incompatible because pixel ImageGPT has the wrong vocabulary.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


@torch.no_grad()
def run_llamagen_c2i(args: argparse.Namespace) -> dict[str, Any]:
    profile = dict(LLAMAGEN_PROFILES[args.profile])
    repo_path = Path(args.llamagen_repo).resolve() if args.llamagen_repo else REPO_ROOT / "external" / "LlamaGen"
    if args.auto_clone:
        maybe_clone_llamagen(repo_path)
    repo_path = resolve_llamagen_repo(str(repo_path))
    device = resolve_device(args.device)
    dtype = resolve_precision(args.precision, device)
    out_dir = Path(args.out_dir).resolve()
    if str(args.out_dir).strip() == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "runs" / "local" / "pretrained_vq_ar" / f"llamagen_{args.profile}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vq_path = hf_hub_download(repo_id=profile["repo_id"], filename=profile["vq_file"])
    gpt_path = hf_hub_download(repo_id=profile["repo_id"], filename=profile["gpt_file"])
    class_labels = parse_class_labels(args.class_labels, samples=args.samples, seed=args.seed)
    latent_size = int(profile["image_size"]) // int(profile["downsample_size"])
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    with llamagen_import_context(repo_path):
        from autoregressive.models.generate import generate
        from autoregressive.models.gpt import GPT_models
        from tokenizer.tokenizer_image.vq_model import VQ_models

        vq_model = VQ_models[profile["vq_model"]](
            codebook_size=int(profile["codebook_size"]),
            codebook_embed_dim=int(profile["codebook_embed_dim"]),
        ).to(device)
        vq_model.load_state_dict(load_weight_payload(vq_path), strict=True)
        vq_model.eval()

        gpt_model = GPT_models[profile["gpt_model"]](
            vocab_size=int(profile["codebook_size"]),
            block_size=latent_size ** 2,
            num_classes=1000,
            cls_token_num=1,
            model_type="c2i",
        ).to(device=device, dtype=dtype)
        missing, unexpected = gpt_model.load_state_dict(load_weight_payload(gpt_path), strict=False)
        gpt_model.eval()

        c_indices = torch.tensor(class_labels, dtype=torch.long, device=device)
        index_sample = generate(
            gpt_model,
            c_indices,
            latent_size ** 2,
            cfg_scale=float(args.cfg_scale),
            cfg_interval=float(args.cfg_interval),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            sample_logits=True,
        )
        qzshape = [
            len(class_labels),
            int(profile["codebook_embed_dim"]),
            latent_size,
            latent_size,
        ]
        samples = vq_model.decode_code(index_sample, qzshape)

    grid_path = save_grid(
        samples,
        out_dir / "llamagen_c2i_samples.png",
        labels=class_labels,
        title=f"LlamaGen {args.profile} samples",
    )
    tokens_path = out_dir / "llamagen_c2i_tokens.pt"
    torch.save(index_sample.detach().cpu(), tokens_path)

    summary = {
        "mode": "llamagen-c2i",
        "profile": args.profile,
        "profile_metadata": profile,
        "llamagen_repo": str(repo_path),
        "device": str(device),
        "dtype": str(dtype),
        "samples": int(args.samples),
        "class_labels": class_labels,
        "latent_size": latent_size,
        "sequence_length": latent_size ** 2,
        "grid": str(grid_path),
        "tokens": str(tokens_path),
        "missing_weight_keys": list(missing),
        "unexpected_weight_keys": list(unexpected),
    }
    summary_path = out_dir / "llamagen_c2i_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"llamagen-{args.profile}",
            tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
            config={k: v for k, v in summary.items() if isinstance(v, (str, int, float, bool))},
        )
        wandb.log({"vq_ar/llamagen_samples": wandb.Image(str(grid_path))})
        artifact = wandb.Artifact(f"{run.name}_outputs", type="generation")
        artifact.add_file(str(grid_path))
        artifact.add_file(str(summary_path))
        artifact.add_file(str(tokens_path))
        run.log_artifact(artifact)
        run.finish()

    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else REPO_ROOT / "runs" / "local" / "pretrained_vq_ar"
    if args.mode == "compatibility":
        report = build_compatibility_report()
        paths = write_compatibility_report(report, out_dir)
        return {"mode": "compatibility", "paths": paths, **report["summary"]}
    if args.mode == "llamagen-c2i":
        return run_llamagen_c2i(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["compatibility", "llamagen-c2i"], default="compatibility")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--profile", choices=sorted(LLAMAGEN_PROFILES), default="c2i-B-256")
    parser.add_argument("--llamagen-repo", default="")
    parser.add_argument("--auto-clone", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--class-labels", default="207,360,387,974")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-tags", default="vq-ar,llamagen,pretrained")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
