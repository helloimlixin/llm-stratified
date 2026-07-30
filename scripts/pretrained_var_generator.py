"""Sample images from pretrained FoundationVision/VAR checkpoints.

This is the non-toy generation path for this repo.  It loads both the
pretrained VAR VQ-VAE tokenizer and the pretrained visual autoregressive
transformer, then samples ImageNet-class-conditional images without local
training.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]

VAR_MODEL_SPECS: dict[int, dict[str, Any]] = {
    16: {"resolution": 256, "fid": 3.55, "params": "310M", "shared_aln": False},
    20: {"resolution": 256, "fid": 2.95, "params": "600M", "shared_aln": False},
    24: {"resolution": 256, "fid": 2.33, "params": "1.0B", "shared_aln": False},
    30: {"resolution": 256, "fid": 1.97, "params": "2.0B", "shared_aln": False},
    36: {"resolution": 512, "fid": 2.63, "params": "2.3B", "shared_aln": True},
}

PATCH_NUMS_BY_RESOLUTION: dict[int, tuple[int, ...]] = {
    256: (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    512: (1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
}


def _resolve_var_repo_path(var_repo_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if var_repo_path:
        candidates.append(Path(var_repo_path))
    env_path = os.environ.get("VAR_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([REPO_ROOT / "external" / "VAR", Path.cwd() / "external" / "VAR"])
    for candidate in candidates:
        if (candidate / "models" / "__init__.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find FoundationVision/VAR; expected external/VAR or VAR_REPO_PATH.")


def _import_var_models(var_repo_path: str | None = None) -> ModuleType:
    repo_path = _resolve_var_repo_path(var_repo_path)
    saved_models = sys.modules.get("models")
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, str(repo_path))
        if saved_models is not None:
            del sys.modules["models"]
        return importlib.import_module("models")
    finally:
        if saved_models is not None:
            sys.modules["models"] = saved_models
        else:
            sys.modules.pop("models", None)
        sys.path[:] = saved_path


def _torch_load(path: str | Path, *, map_location: torch.device, mmap: bool = False) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True, mmap=bool(mmap))
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_model_defaults(depth: int) -> dict[str, Any]:
    depth = int(depth)
    if depth not in VAR_MODEL_SPECS:
        known = ", ".join(str(key) for key in sorted(VAR_MODEL_SPECS))
        raise ValueError(f"Unsupported VAR depth {depth}; expected one of: {known}")
    spec = dict(VAR_MODEL_SPECS[depth])
    resolution = int(spec["resolution"])
    spec["patch_nums"] = PATCH_NUMS_BY_RESOLUTION[resolution]
    spec["filename"] = f"var_d{depth}.pth"
    return spec


def parse_patch_nums(text: str | None, *, resolution: int) -> tuple[int, ...]:
    if text is None or not str(text).strip() or str(text).strip().lower() in {"auto", "default"}:
        return PATCH_NUMS_BY_RESOLUTION[int(resolution)]
    lowered = str(text).strip().lower()
    if lowered in {"256", "imagenet256"}:
        return PATCH_NUMS_BY_RESOLUTION[256]
    if lowered in {"512", "imagenet512"}:
        return PATCH_NUMS_BY_RESOLUTION[512]
    values = tuple(int(part.strip()) for part in str(text).replace("_", ",").split(",") if part.strip())
    if not values:
        raise ValueError("patch nums cannot be empty")
    return values


def parse_class_labels(text: str | None, *, samples: int) -> list[int] | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if value in {"", "none", "random"}:
        return None
    labels = [int(part.strip()) for part in value.replace("_", ",").split(",") if part.strip()]
    if not labels:
        return None
    for label in labels:
        if label < 0 or label > 999:
            raise ValueError(f"ImageNet class labels must be in [0, 999], got {label}")
    if len(labels) >= int(samples):
        return labels[: int(samples)]
    repeats = int(np.ceil(int(samples) / max(1, len(labels))))
    return (labels * repeats)[: int(samples)]


def _configure_torch(seed: int, tf32: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    try:
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        pass


def build_pretrained_var(
    *,
    depth: int,
    repo_id: str,
    vae_filename: str,
    var_filename: str,
    patch_nums: tuple[int, ...],
    shared_aln: bool,
    device: torch.device,
    var_repo_path: str | None,
    dtype: torch.dtype | None = None,
    mmap_load: bool = False,
):
    var_models = _import_var_models(var_repo_path)
    old_dtype = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        vae, var = var_models.build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,
            device=device,
            patch_nums=patch_nums,
            num_classes=1000,
            depth=int(depth),
            shared_aln=bool(shared_aln),
            flash_if_available=True,
            fused_if_available=True,
        )
    finally:
        if dtype is not None:
            torch.set_default_dtype(old_dtype)
    if dtype is not None:
        vae = vae.to(dtype=dtype)
        var = var.to(dtype=dtype)
    vae_path = hf_hub_download(repo_id=repo_id, filename=vae_filename)
    var_path = hf_hub_download(repo_id=repo_id, filename=var_filename)
    load_device = torch.device("cpu") if mmap_load else device
    vae_state = _torch_load(vae_path, map_location=load_device, mmap=mmap_load)
    vae.load_state_dict(vae_state, strict=True)
    del vae_state
    var_state = _torch_load(var_path, map_location=load_device, mmap=mmap_load)
    var.load_state_dict(var_state, strict=True)
    del var_state
    vae.eval()
    var.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    for param in var.parameters():
        param.requires_grad_(False)
    return vae, var, vae_path, var_path


def _tensor_to_pil(image_chw: torch.Tensor) -> Image.Image:
    arr = image_chw.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="RGB")


def save_grid(images: torch.Tensor, out_path: Path, *, labels: list[int] | None, title: str, cols: int = 4) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = int(images.shape[0])
    cols = max(1, min(int(cols), count))
    rows = int(np.ceil(count / cols))
    tile = int(images.shape[-1])
    pad = 10
    title_h = 34
    label_h = 22 if labels is not None else 0
    width = cols * tile + (cols + 1) * pad
    height = title_h + rows * (tile + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 10), title, fill=(24, 32, 45), font=font)
    for idx in range(count):
        row, col = divmod(idx, cols)
        x = pad + col * (tile + pad)
        y = title_h + pad + row * (tile + label_h + pad)
        canvas.paste(_tensor_to_pil(images[idx]), (x, y))
        if labels is not None:
            draw.text((x, y + tile + 4), f"class {labels[idx]}", fill=(69, 79, 94), font=font)
    canvas.save(out_path)
    return out_path


def save_individual_images(images: torch.Tensor, out_dir: Path, *, labels: list[int] | None, seed: int) -> list[Path]:
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(int(images.shape[0])):
        label_suffix = f"_class{labels[idx]:03d}" if labels is not None else ""
        path = sample_dir / f"var_seed{int(seed)}_{idx:03d}{label_suffix}.png"
        _tensor_to_pil(images[idx]).save(path)
        paths.append(path)
    return paths


def _import_real_wandb():
    cached = sys.modules.pop("wandb", None)
    original_path = list(sys.path)
    repo_text = str(REPO_ROOT.resolve())
    try:
        sys.path = [
            item for item in sys.path
            if item not in {"", "."} and str(Path(item).resolve()) != repo_text
        ]
        module = importlib.import_module("wandb")
        if not hasattr(module, "init"):
            raise ImportError("imported wandb module has no init()")
        return module
    except Exception:
        if cached is not None:
            sys.modules["wandb"] = cached
        raise
    finally:
        sys.path = original_path


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any], image_paths: dict[str, Path]) -> str | None:
    if not args.wandb:
        return None
    try:
        wandb = _import_real_wandb()
    except Exception as exc:
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    try:
        wandb_dir = args.out_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=[tag for tag in str(args.wandb_tags).split(",") if tag],
            config=summary,
            dir=str(wandb_dir),
            mode=args.wandb_mode or None,
        )
        payload = {f"pretrained_var/{key}": value for key, value in summary.items() if isinstance(value, (int, float, str))}
        for key, path in image_paths.items():
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                payload[f"pretrained_var/{key}"] = wandb.Image(str(path))
        wandb.log(payload, step=0)
        artifact = wandb.Artifact(f"{args.wandb_name or 'pretrained_var'}_outputs", type="generation")
        for path in image_paths.values():
            artifact.add_file(str(path))
        summary_path = args.out_dir / "pretrained_var_summary.json"
        if summary_path.exists():
            artifact.add_file(str(summary_path))
        wandb.log_artifact(artifact)
        url = getattr(run, "url", None)
        wandb.finish()
        return url
    except Exception as exc:
        print(f"[wandb] logging failed: {exc}", flush=True)
        try:
            wandb.finish()
        except Exception:
            pass
        return None


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from pretrained FoundationVision/VAR checkpoints.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repo-id", default="FoundationVision/var")
    parser.add_argument("--vae-filename", default="vae_ch160v4096z32.pth")
    parser.add_argument("--var-filename", default=None)
    parser.add_argument("--var-repo-path", default=None)
    parser.add_argument("--model-depth", type=int, default=16, choices=sorted(VAR_MODEL_SPECS))
    parser.add_argument("--patch-nums", default="auto", help="'auto', '256', '512', or comma/underscore-separated patch nums.")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--class-labels", default="980,437,22,562", help="Comma-separated ImageNet class IDs, or 'random'.")
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--more-smooth", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default="pretrained-var,generation")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = REPO_ROOT / "runs" / "local" / "pretrained_var" / stamp
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_mode == "disabled":
        args.wandb = False

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but this Python environment has no CUDA runtime.")

    defaults = resolve_model_defaults(args.model_depth)
    var_filename = args.var_filename or str(defaults["filename"])
    patch_nums = parse_patch_nums(args.patch_nums, resolution=int(defaults["resolution"]))
    class_labels = parse_class_labels(args.class_labels, samples=args.samples)
    _configure_torch(seed=args.seed, tf32=args.tf32)

    if device.type == "cpu" and int(args.model_depth) >= 30:
        print(
            "[setup] warning: d30/d36 are real multi-billion-parameter checkpoints; CPU sampling can be very slow.",
            flush=True,
        )

    print(
        f"[setup] loading pretrained VAR-d{args.model_depth} ({var_filename}) on {device}; "
        f"patch_nums={patch_nums}",
        flush=True,
    )
    t0 = time.perf_counter()
    _vae, var, vae_path, var_path = build_pretrained_var(
        depth=args.model_depth,
        repo_id=args.repo_id,
        vae_filename=args.vae_filename,
        var_filename=var_filename,
        patch_nums=patch_nums,
        shared_aln=bool(defaults["shared_aln"]),
        device=device,
        var_repo_path=args.var_repo_path,
    )
    load_seconds = time.perf_counter() - t0

    print(
        f"[sample] B={args.samples} cfg={args.cfg} top_k={args.top_k} top_p={args.top_p} "
        f"labels={class_labels if class_labels is not None else 'random'}",
        flush=True,
    )
    if class_labels is None:
        label_b = None
    else:
        label_b = torch.tensor(class_labels, dtype=torch.long, device=device)

    t1 = time.perf_counter()
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
                samples = var.autoregressive_infer_cfg(
                    B=int(args.samples),
                    label_B=label_b,
                    cfg=float(args.cfg),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    g_seed=int(args.seed),
                    more_smooth=bool(args.more_smooth),
                )
        else:
            samples = var.autoregressive_infer_cfg(
                B=int(args.samples),
                label_B=label_b,
                cfg=float(args.cfg),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                g_seed=int(args.seed),
                more_smooth=bool(args.more_smooth),
            )
    sample_seconds = time.perf_counter() - t1
    samples = samples.detach().cpu().clamp(0.0, 1.0)

    label_list = class_labels if class_labels is not None else None
    image_paths: dict[str, Path] = {}
    title = f"Pretrained VAR-d{args.model_depth} samples"
    image_paths["sample_grid"] = save_grid(
        samples,
        args.out_dir / "pretrained_var_samples.png",
        labels=label_list,
        title=title,
        cols=args.grid_cols,
    )
    individual_paths = save_individual_images(samples, args.out_dir, labels=label_list, seed=args.seed)
    for idx, path in enumerate(individual_paths):
        image_paths[f"sample_{idx:03d}"] = path

    summary = {
        "repo_id": args.repo_id,
        "model_depth": int(args.model_depth),
        "var_filename": var_filename,
        "vae_filename": args.vae_filename,
        "reported_fid": float(defaults["fid"]),
        "reported_params": str(defaults["params"]),
        "reported_resolution": int(defaults["resolution"]),
        "shared_aln": bool(defaults["shared_aln"]),
        "patch_nums": list(patch_nums),
        "device": str(device),
        "samples": int(args.samples),
        "class_labels": class_labels if class_labels is not None else "random",
        "cfg": float(args.cfg),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "more_smooth": bool(args.more_smooth),
        "seed": int(args.seed),
        "load_seconds": float(load_seconds),
        "sample_seconds": float(sample_seconds),
        "vae_path": str(vae_path),
        "var_path": str(var_path),
        "outputs": {
            "sample_grid": str(image_paths["sample_grid"]),
            "samples": [str(path) for path in individual_paths],
        },
    }
    summary_path = args.out_dir / "pretrained_var_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(summary), fp, indent=2)

    url = maybe_log_wandb(args, summary, image_paths)
    print(json.dumps(to_jsonable(summary), indent=2), flush=True)
    print("[outputs]", flush=True)
    print(f"sample_grid: {image_paths['sample_grid']}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    if url:
        print(f"[wandb] {url}", flush=True)


if __name__ == "__main__":
    main()
