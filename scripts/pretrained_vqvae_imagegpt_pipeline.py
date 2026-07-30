"""Tiny pretrained-VQ-VAE + ImageGPT-style generation pipeline.

This is a deliberately small bridge between the repo's VAR/VQ-VAE tooling and
the older ImageGPT-style token modeling code:

1. Load the pretrained FoundationVision/VAR VQ-VAE.
2. Encode images into discrete VQ code sequences.
3. Train a compact causal transformer over those code IDs.
4. Sample new code sequences and decode them through the VQ-VAE.

The autoregressive model here is trained locally because pretrained pixel
ImageGPT checkpoints are not vocabulary-compatible with VQ-VAE code IDs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_var_repo_path(var_repo_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if var_repo_path:
        candidates.append(Path(var_repo_path))
    env_path = os.environ.get("VAR_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([REPO_ROOT / "external" / "VAR", Path.cwd() / "external" / "VAR"])
    for candidate in candidates:
        if (candidate / "models" / "vqvae.py").exists():
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


def _torch_load(path: str | Path, *, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_patch_nums(text: str) -> tuple[int, ...]:
    if str(text).strip().lower() in {"full", "var"}:
        return (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("patch nums cannot be empty")
    if values == (16,):
        # VAR's residual quantizer expects at least two scales because it indexes
        # quant_resi by si / (SN - 1).  The smallest useful smoke setup is a
        # global 1x1 code plus the final 16x16 grid.
        values = (1, 16)
    if values[-1] != 16:
        raise ValueError("VAR VQ-VAE image encoding expects the last patch number to be 16 for 256px images")
    if len(values) < 2:
        raise ValueError("VAR VQ-VAE residual quantization requires at least two scales")
    return values


def split_multiscale_tokens(tokens: torch.Tensor, patch_nums: tuple[int, ...]) -> list[torch.Tensor]:
    """Split a flat token sequence into VAR VQ-VAE scale tensors."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    chunks: list[torch.Tensor] = []
    start = 0
    for patch_num in patch_nums:
        length = int(patch_num) * int(patch_num)
        stop = start + length
        if stop > int(tokens.shape[1]):
            raise ValueError("token sequence is shorter than patch_nums require")
        chunks.append(tokens[:, start:stop])
        start = stop
    if start != int(tokens.shape[1]):
        raise ValueError("token sequence has extra entries after patch_nums split")
    return chunks


def load_var_vqvae(
    *,
    device: torch.device,
    repo_id: str,
    vae_filename: str,
    patch_nums: tuple[int, ...],
    var_repo_path: str | None,
):
    var_models = _import_var_models(var_repo_path)
    vae = var_models.VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=patch_nums,
    ).to(device)
    vae_path = hf_hub_download(repo_id=repo_id, filename=vae_filename)
    vae.load_state_dict(_torch_load(vae_path, map_location=device), strict=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def load_image_folder(image_dir: Path, *, image_size: int, limit: int = 0) -> tuple[torch.Tensor, list[str]]:
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if limit and limit > 0:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(tensor.mul(2.0).sub(1.0))
        names.append(path.name)
    return torch.stack(tensors, dim=0), names


@torch.no_grad()
def encode_images(vae, images_m11: torch.Tensor, *, patch_nums: tuple[int, ...], batch_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    device = next(vae.parameters()).device
    for start in range(0, int(images_m11.shape[0]), max(1, int(batch_size))):
        batch = images_m11[start:start + int(batch_size)].to(device)
        idx_bl = vae.img_to_idxBl(batch, v_patch_nums=patch_nums)
        chunks.append(torch.cat([part.detach().cpu() for part in idx_bl], dim=1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def decode_tokens(vae, tokens: torch.Tensor, *, patch_nums: tuple[int, ...], batch_size: int) -> torch.Tensor:
    images: list[torch.Tensor] = []
    device = next(vae.parameters()).device
    for start in range(0, int(tokens.shape[0]), max(1, int(batch_size))):
        batch = tokens[start:start + int(batch_size)].to(device)
        idx_bl = split_multiscale_tokens(batch, patch_nums)
        recon = vae.idxBl_to_img(idx_bl, same_shape=True, last_one=True)
        images.append(recon.detach().cpu().add(1.0).mul(0.5).clamp(0.0, 1.0))
    return torch.cat(images, dim=0)


class VQImageGPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.bos_token = int(vocab_size)
        self.seq_len = int(seq_len)
        self.token_emb = nn.Embedding(self.vocab_size + 1, n_embd)
        self.pos_emb = nn.Embedding(self.seq_len, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, self.vocab_size)

    def inputs_from_targets(self, tokens: torch.Tensor) -> torch.Tensor:
        bos = torch.full((tokens.shape[0], 1), self.bos_token, dtype=torch.long, device=tokens.device)
        return torch.cat([bos, tokens[:, :-1]], dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        if length != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {length}")
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_emb(input_ids) + self.pos_emb(pos)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        return self.head(self.norm(hidden))


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k=int(top_k), dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, -torch.inf)


@torch.no_grad()
def sample_gpt(
    model: VQImageGPT,
    *,
    samples: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    generated = torch.empty((int(samples), 0), dtype=torch.long, device=device)
    for _pos in range(model.seq_len):
        input_ids = torch.full(
            (int(samples), model.seq_len),
            model.bos_token,
            dtype=torch.long,
            device=device,
        )
        if generated.numel():
            input_ids[:, :generated.shape[1]] = generated
        logits = model(input_ids)[:, generated.shape[1], :] / max(float(temperature), 1e-6)
        logits = top_k_logits(logits, int(top_k))
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated.cpu()


def train_gpt(
    tokens: torch.Tensor,
    *,
    vocab_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    n_embd: int,
    n_head: int,
    n_layer: int,
    dropout: float,
    seed: int,
    device: torch.device,
) -> tuple[VQImageGPT, list[float]]:
    torch.manual_seed(int(seed))
    model = VQImageGPT(
        vocab_size=vocab_size,
        seq_len=int(tokens.shape[1]),
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    tokens = tokens.long()
    losses: list[float] = []
    generator = torch.Generator().manual_seed(int(seed))
    for epoch in range(max(1, int(epochs))):
        order = torch.randperm(tokens.shape[0], generator=generator)
        epoch_losses: list[float] = []
        model.train()
        for start in range(0, int(order.numel()), max(1, int(batch_size))):
            idx = order[start:start + int(batch_size)]
            target = tokens[idx].to(device)
            inp = model.inputs_from_targets(target)
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(mean_loss)
        print(f"[imagegpt] epoch {epoch + 1:03d}/{epochs:03d} loss={mean_loss:.4f}", flush=True)
    return model, losses


def tensor_to_pil(img01: torch.Tensor) -> Image.Image:
    arr = img01.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_grid(images01: torch.Tensor, out_path: Path, *, title: str, cols: int = 4) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_images = [tensor_to_pil(img) for img in images01]
    if not pil_images:
        raise ValueError("no images to save")
    cell_w, cell_h = pil_images[0].size
    cols = max(1, min(int(cols), len(pil_images)))
    rows = int(math.ceil(len(pil_images) / cols))
    title_h = 42
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((12, 10), title, fill=(25, 30, 38), font=font)
    for idx, img in enumerate(pil_images):
        row, col = divmod(idx, cols)
        canvas.paste(img, (col * cell_w, title_h + row * cell_h))
    canvas.save(out_path)
    return out_path


def save_loss_curve(losses: list[float], out_path: Path) -> Path:
    width, height = 760, 420
    margin_l, margin_r, margin_t, margin_b = 72, 38, 56, 60
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 20), "Token GPT training loss", fill=(25, 30, 38), font=ImageFont.load_default())
    if losses:
        vals = np.asarray(losses, dtype=np.float64)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if abs(hi - lo) < 1e-9:
            hi = lo + 1.0
        x0, y0 = margin_l, height - margin_b
        x1, y1 = width - margin_r, margin_t
        draw.line((x0, y0, x1, y0), fill=(25, 30, 38), width=2)
        draw.line((x0, y0, x0, y1), fill=(25, 30, 38), width=2)
        points = []
        for idx, val in enumerate(vals):
            x = x0 + (idx / max(1, len(vals) - 1)) * (x1 - x0)
            y = y0 - ((float(val) - lo) / max(hi - lo, 1e-12)) * (y0 - y1)
            points.append((int(x), int(y)))
        if len(points) > 1:
            draw.line(points, fill=(47, 99, 164), width=4)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(47, 99, 164))
        draw.text((x0, y0 + 14), f"epoch 1", fill=(104, 113, 128), font=ImageFont.load_default())
        draw.text((x1 - 70, y0 + 14), f"epoch {len(vals)}", fill=(104, 113, 128), font=ImageFont.load_default())
        draw.text((x0 - 58, y1 - 8), f"{hi:.2f}", fill=(104, 113, 128), font=ImageFont.load_default())
        draw.text((x0 - 58, y0 - 8), f"{lo:.2f}", fill=(104, 113, 128), font=ImageFont.load_default())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


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
        print(f"[wandb] unavailable: {exc}")
        return None
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=[tag for tag in str(args.wandb_tags).split(",") if tag],
            config=summary,
            dir=str(args.out_dir / "wandb"),
            mode=args.wandb_mode or None,
        )
        payload = {f"vqvae_imagegpt/{key}": value for key, value in summary.items() if isinstance(value, (int, float, str))}
        for key, path in image_paths.items():
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                payload[f"vqvae_imagegpt/{key}"] = wandb.Image(str(path))
        wandb.log(payload, step=0)
        artifact = wandb.Artifact(f"{args.wandb_name or 'vqvae_imagegpt'}_outputs", type="generation")
        for path in image_paths.values():
            artifact.add_file(str(path))
        wandb.log_artifact(artifact)
        url = getattr(run, "url", None)
        wandb.finish()
        return url
    except Exception as exc:
        print(f"[wandb] logging failed: {exc}")
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
    parser = argparse.ArgumentParser(description="Pretrained VAR VQ-VAE plus tiny ImageGPT-style token generator.")
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "docs" / "imgs" / "neurips_submission")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repo-id", default="FoundationVision/var")
    parser.add_argument("--vae-filename", default="vae_ch160v4096z32.pth")
    parser.add_argument("--var-repo-path", default=None)
    parser.add_argument("--patch-nums", default="1,16", help="'1,16' for smoke runs, or 'full' for VAR scales.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default="vqvae-imagegpt,smoke")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_nums = parse_patch_nums(args.patch_nums)
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = REPO_ROOT / "runs" / "local" / "vqvae_imagegpt" / stamp
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_mode == "disabled":
        args.wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    print(f"[setup] loading images from {args.image_dir}", flush=True)
    images_m11, image_names = load_image_folder(args.image_dir, image_size=args.image_size, limit=args.limit_images)
    image_paths: dict[str, Path] = {}
    image_paths["inputs"] = save_grid(images_m11.add(1.0).mul(0.5), args.out_dir / "input_images.png", title="Input images", cols=4)

    print(f"[setup] loading pretrained VQ-VAE {args.repo_id}/{args.vae_filename} on {device}", flush=True)
    vae = load_var_vqvae(
        device=device,
        repo_id=args.repo_id,
        vae_filename=args.vae_filename,
        patch_nums=patch_nums,
        var_repo_path=args.var_repo_path,
    )

    print("[vqvae] encoding and reconstructing", flush=True)
    tokens = encode_images(vae, images_m11, patch_nums=patch_nums, batch_size=args.batch_size)
    recon = decode_tokens(vae, tokens, patch_nums=patch_nums, batch_size=args.batch_size)
    image_paths["reconstructions"] = save_grid(
        recon,
        args.out_dir / "vqvae_reconstructions.png",
        title=f"Pretrained VQ-VAE reconstructions ({','.join(map(str, patch_nums))})",
        cols=4,
    )

    print("[imagegpt] training tiny causal transformer over VQ codes", flush=True)
    model, losses = train_gpt(
        tokens,
        vocab_size=int(vae.vocab_size),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        seed=args.seed,
        device=device,
    )
    image_paths["loss_curve"] = save_loss_curve(losses, args.out_dir / "imagegpt_loss_curve.png")

    print("[imagegpt] sampling and decoding", flush=True)
    sampled = sample_gpt(
        model,
        samples=args.samples,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    generated = decode_tokens(vae, sampled, patch_nums=patch_nums, batch_size=args.batch_size)
    image_paths["samples"] = save_grid(
        generated,
        args.out_dir / "imagegpt_samples.png",
        title="Tiny ImageGPT samples decoded by pretrained VQ-VAE",
        cols=4,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "tokens": tokens,
            "sampled_tokens": sampled,
            "patch_nums": patch_nums,
            "cfg": {
                "vocab_size": int(vae.vocab_size),
                "seq_len": int(tokens.shape[1]),
                "n_embd": int(args.n_embd),
                "n_head": int(args.n_head),
                "n_layer": int(args.n_layer),
                "dropout": float(args.dropout),
            },
        },
        args.out_dir / "tiny_imagegpt.pt",
    )
    np.save(args.out_dir / "vq_tokens.npy", tokens.numpy())

    usage = np.bincount(tokens.reshape(-1).numpy(), minlength=int(vae.vocab_size))
    summary = {
        "image_dir": str(args.image_dir),
        "image_names": image_names,
        "num_images": int(tokens.shape[0]),
        "patch_nums": list(patch_nums),
        "seq_len": int(tokens.shape[1]),
        "vocab_size": int(vae.vocab_size),
        "used_codes": int(np.count_nonzero(usage)),
        "used_code_ratio": float(np.count_nonzero(usage) / max(1, int(vae.vocab_size))),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "min_loss": float(np.nanmin(np.asarray(losses))) if losses else float("nan"),
        "samples": int(args.samples),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "outputs": {key: str(path) for key, path in image_paths.items()},
    }
    summary_path = args.out_dir / "vqvae_imagegpt_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(summary), fp, indent=2)
    image_paths["summary"] = summary_path

    url = maybe_log_wandb(args, summary, image_paths)
    print(json.dumps(to_jsonable(summary), indent=2))
    print("[outputs]")
    for key, path in image_paths.items():
        print(f"{key}: {path}")
    if url:
        print(f"[wandb] {url}")


if __name__ == "__main__":
    main()
