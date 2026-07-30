"""Encode real image folders into LlamaGen VQ tokens for AR uniformity probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pretrained_vq_ar_pipeline import (  # noqa: E402
    LLAMAGEN_PROFILES,
    llamagen_import_context,
    load_weight_payload,
    parse_class_labels,
    resolve_device,
    resolve_llamagen_repo,
    save_grid,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path, *, limit: int, seed: int) -> list[Path]:
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no images found under {root}")
    rng = np.random.default_rng(int(seed))
    if limit > 0 and len(paths) > limit:
        idx = np.sort(rng.choice(len(paths), size=int(limit), replace=False))
        paths = [paths[int(i)] for i in idx]
    return paths


def load_labels_file(path: Path) -> dict[str, int]:
    if not path:
        return {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k).replace("\\", "/"): int(v) for k, v in payload.items()}
        if isinstance(payload, list):
            out: dict[str, int] = {}
            for row in payload:
                if isinstance(row, dict) and "path" in row and "label" in row:
                    out[str(row["path"]).replace("\\", "/")] = int(row["label"])
            return out
    out = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            out[str(row["path"]).replace("\\", "/")] = int(row["label"])
    return out


def labels_for_images(
    paths: list[Path],
    *,
    root: Path,
    mode: str,
    class_labels: str,
    labels_file: Path | None,
    seed: int,
) -> list[int]:
    if mode == "class_labels":
        return parse_class_labels(class_labels, samples=len(paths), seed=seed)
    if mode == "labels_file":
        if labels_file is None:
            raise ValueError("--labels-file is required when --label-mode labels_file")
        lookup = load_labels_file(labels_file)
        labels = []
        for path in paths:
            rel = path.relative_to(root).as_posix()
            key_options = [rel, path.as_posix(), path.name]
            for key in key_options:
                if key in lookup:
                    labels.append(int(lookup[key]))
                    break
            else:
                raise KeyError(f"missing class label for {rel}")
        return labels
    if mode == "parent_index":
        classes = sorted({p.parent.name for p in paths})
        mapping = {name: idx for idx, name in enumerate(classes)}
        labels = [mapping[p.parent.name] for p in paths]
        if any(label < 0 or label >= 1000 for label in labels):
            raise ValueError("parent_index labels must be valid ImageNet class indices 0..999")
        return labels
    raise ValueError(f"unknown label mode: {mode}")


def pil_to_llamagen_tensor(image: Image.Image, *, image_size: int, center_crop_arr) -> torch.Tensor:
    image = image.convert("RGB")
    cropped = center_crop_arr(image, int(image_size))
    arr = np.asarray(cropped, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    profile = dict(LLAMAGEN_PROFILES[args.profile])
    repo_path = resolve_llamagen_repo(args.llamagen_repo or None)
    image_root = Path(args.image_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    paths = collect_images(image_root, limit=args.samples, seed=args.seed)
    labels = labels_for_images(
        paths,
        root=image_root,
        mode=args.label_mode,
        class_labels=args.class_labels,
        labels_file=Path(args.labels_file).resolve() if args.labels_file else None,
        seed=args.seed,
    )

    vq_path = hf_hub_download(repo_id=profile["repo_id"], filename=profile["vq_file"])
    image_size = int(profile["image_size"])
    latent_size = image_size // int(profile["downsample_size"])

    with llamagen_import_context(repo_path):
        from dataset.augmentation import center_crop_arr
        from tokenizer.tokenizer_image.vq_model import VQ_models

        vq_model = VQ_models[profile["vq_model"]](
            codebook_size=int(profile["codebook_size"]),
            codebook_embed_dim=int(profile["codebook_embed_dim"]),
        ).to(device)
        vq_model.load_state_dict(load_weight_payload(vq_path), strict=True)
        vq_model.eval()

        all_codes: list[torch.Tensor] = []
        recon_batches: list[torch.Tensor] = []
        source_batches: list[torch.Tensor] = []
        for start in range(0, len(paths), int(args.batch_size)):
            batch_paths = paths[start : start + int(args.batch_size)]
            batch = []
            for path in batch_paths:
                with Image.open(path) as image:
                    batch.append(pil_to_llamagen_tensor(image, image_size=image_size, center_crop_arr=center_crop_arr))
            x = torch.stack(batch, dim=0).to(device)
            _quant, _loss, info = vq_model.encode(x)
            indices = info[2].reshape(x.shape[0], latent_size * latent_size)
            all_codes.append(indices.detach().cpu())
            if len(recon_batches) * int(args.batch_size) < int(args.grid_samples):
                qzshape = [x.shape[0], int(profile["codebook_embed_dim"]), latent_size, latent_size]
                recon_batches.append(vq_model.decode_code(indices, qzshape).detach().cpu())
                source_batches.append(x.detach().cpu())

    tokens = torch.cat(all_codes, dim=0).long()
    tokens_path = out_dir / "llamagen_c2i_tokens.pt"
    torch.save(tokens, tokens_path)

    figures: dict[str, str] = {}
    if source_batches:
        source = torch.cat(source_batches, dim=0)[: int(args.grid_samples)]
        recon = torch.cat(recon_batches, dim=0)[: int(args.grid_samples)]
        source_path = save_grid(source, out_dir / "llamagen_c2i_sources.png", labels=labels[: source.shape[0]], title="VQ input images")
        recon_path = save_grid(
            recon,
            out_dir / "llamagen_c2i_reconstructions.png",
            labels=labels[: recon.shape[0]],
            title="LlamaGen VQ reconstructions",
        )
        figures["sources"] = str(source_path)
        figures["reconstructions"] = str(recon_path)

    records = [
        {"index": idx, "path": str(path), "relative_path": path.relative_to(image_root).as_posix(), "class_label": int(labels[idx])}
        for idx, path in enumerate(paths)
    ]
    records_path = out_dir / "llamagen_c2i_dataset_records.json"
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary = {
        "mode": "llamagen-c2i-encode-dataset",
        "profile": args.profile,
        "image_dir": str(image_root),
        "out_dir": str(out_dir),
        "device": str(device),
        "samples": len(paths),
        "image_size": image_size,
        "latent_size": latent_size,
        "sequence_length": int(tokens.shape[1]),
        "tokens": str(tokens_path),
        "class_labels": labels,
        "records": str(records_path),
        "figures": figures,
    }
    summary_path = out_dir / "llamagen_c2i_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
            config={k: v for k, v in summary.items() if isinstance(v, (str, int, float, bool))},
        )
        payload = {}
        for key, path in figures.items():
            payload[f"vq_ar_dataset/{key}"] = wandb.Image(path)
        if payload:
            wandb.log(payload)
        artifact = wandb.Artifact(f"{args.wandb_name}_outputs", type="dataset_tokens")
        artifact.add_file(str(tokens_path))
        artifact.add_file(str(summary_path))
        artifact.add_file(str(records_path))
        for path in figures.values():
            artifact.add_file(path)
        run.log_artifact(artifact)
        run.finish()

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(LLAMAGEN_PROFILES), default="c2i-B-256")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llamagen-repo", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-mode", choices=["class_labels", "labels_file", "parent_index"], default="class_labels")
    parser.add_argument("--class-labels", default="random")
    parser.add_argument("--labels-file", default="")
    parser.add_argument("--grid-samples", type=int, default=16)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="llamagen-c2i-dataset-encode")
    parser.add_argument("--wandb-tags", default="vq-ar,llamagen,dataset-tokenization")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
