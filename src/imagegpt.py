#!/usr/bin/env python3
"""
ImageGPT-style discrete-token image modeling for probing token polysemy.

Pipeline:
1) Fit a patch tokenizer via k-means over image patches (MiniBatchKMeans).
2) Encode images into sequences of discrete token IDs.
3) Train a causal GPT to model token sequences.
4) Probe "polysemy" for a token ID by:
   - visualizing real dataset patches assigned to that token
   - computing label entropy over occurrences
   - generating images while forcing a chosen token at a chosen position

This is intentionally lightweight and self-contained (no VQ-VAE dependency).
"""

from __future__ import annotations

from pathlib import Path

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

try:
    from sklearn.cluster import MiniBatchKMeans

    HAS_SK = True
except Exception:
    MiniBatchKMeans = None
    HAS_SK = False

try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

from data import IMAGENET_MEAN, IMAGENET_STD
from utils import denormalize_images, get_device, seed_everything as set_seed


def make_tf(img_size: int, train: bool) -> torchvision.transforms.Compose:
    tfs: List[torchvision.transforms.Transform] = []
    if train:
        tfs.extend(
            [
                torchvision.transforms.Resize((img_size, img_size)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        tfs.append(torchvision.transforms.Resize((img_size, img_size)))
    tfs.extend(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return torchvision.transforms.Compose(tfs)


def make_dataset(name: str, root: str, img_size: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    name = name.upper()
    if name == "STL10":
        train_ds = torchvision.datasets.STL10(root=root, split="train", download=True, transform=make_tf(img_size, True))
        test_ds = torchvision.datasets.STL10(root=root, split="test", download=True, transform=make_tf(img_size, False))
        return train_ds, test_ds, 10
    if name == "CIFAR10":
        train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=make_tf(img_size, True))
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=make_tf(img_size, False))
        return train_ds, test_ds, 10
    if name == "CIFAR100":
        train_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=make_tf(img_size, True))
        test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=make_tf(img_size, False))
        return train_ds, test_ds, 100
    raise ValueError(f"Unsupported dataset: {name}")


def denorm(imgs: torch.Tensor) -> torch.Tensor:
    return denormalize_images(imgs, "IMAGENET")


def save_image_grid(imgs01: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = torchvision.utils.make_grid(imgs01, nrow=nrow)
    torchvision.utils.save_image(grid, str(out_path))


# -----------------------------
# Tokenizer (k-means over patches)
# -----------------------------


@dataclass
class PatchKMeansTokenizer:
    img_size: int
    patch_size: int
    codebook_size: int
    centroids: np.ndarray  # [K, D] where D=3*patch*patch in normalized pixel space

    @property
    def grid(self) -> int:
        return self.img_size // self.patch_size

    @property
    def seq_len(self) -> int:
        return self.grid * self.grid

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [B,3,H,W] normalized. Returns token IDs: [B, L]
        """
        B, C, H, W = imgs.shape
        assert H == self.img_size and W == self.img_size
        ps = self.patch_size
        # [B, C, gh, ps, gw, ps] -> [B, gh*gw, C*ps*ps]
        gh = H // ps
        gw = W // ps
        patches = imgs.reshape(B, C, gh, ps, gw, ps).permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, C * ps * ps)
        x = patches.detach().cpu().numpy().astype(np.float32)
        # compute nearest centroid by L2
        c = self.centroids.astype(np.float32)
        # (B*L, D)
        x2 = x.reshape(-1, x.shape[-1])
        # ||x-c||^2 = x^2 + c^2 - 2x·c
        x_norm = np.sum(x2 * x2, axis=1, keepdims=True)
        c_norm = np.sum(c * c, axis=1, keepdims=True).T
        d2 = x_norm + c_norm - 2.0 * (x2 @ c.T)
        ids = np.argmin(d2, axis=1).astype(np.int64)
        return torch.from_numpy(ids).view(B, gh * gw)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B,L] -> imgs01: [B,3,H,W] in [0,1] (best-effort; denorm from ImageNet stats)
        """
        B, L = tokens.shape
        gh = self.grid
        assert L == gh * gh
        ps = self.patch_size
        c = torch.from_numpy(self.centroids).to(torch.float32)  # [K, D]
        tok = tokens.long().clamp(0, self.codebook_size - 1)
        patch = c[tok]  # [B, L, D]
        patch = patch.view(B, gh, gh, 3, ps, ps).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, gh * ps, gh * ps)
        # patch is in normalized space; denorm to [0,1]
        patch = patch.to(torch.float32)
        return denorm(patch)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "codebook_size": self.codebook_size,
            "centroids": self.centroids,
        }
        np.save(path, payload, allow_pickle=True)

    @staticmethod
    def load(path: Path) -> "PatchKMeansTokenizer":
        payload = np.load(path, allow_pickle=True).item()
        return PatchKMeansTokenizer(
            img_size=int(payload["img_size"]),
            patch_size=int(payload["patch_size"]),
            codebook_size=int(payload["codebook_size"]),
            centroids=np.array(payload["centroids"]),
        )


def fit_tokenizer(
    *,
    dataset: torch.utils.data.Dataset,
    img_size: int,
    patch_size: int,
    codebook_size: int,
    max_images: int,
    batch_size: int,
    seed: int,
) -> PatchKMeansTokenizer:
    """
    Fit k-means centroids over normalized patches.

    Uses scikit-learn MiniBatchKMeans if available, otherwise falls back to a small
    pure-PyTorch online mini-batch k-means implementation (slower but dependency-free).
    """
    set_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    ps = patch_size
    dim = 3 * ps * ps

    # Helper to extract flattened patches [N, D] on CPU
    def patches_from(imgs: torch.Tensor) -> torch.Tensor:
        imgs = imgs[:, :, :img_size, :img_size].contiguous()
        B, C, H, W = imgs.shape
        gh = H // ps
        gw = W // ps
        return (
            imgs.reshape(B, C, gh, ps, gw, ps)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B * gh * gw, dim)
            .to(torch.float32)
            .cpu()
        )

    if HAS_SK and MiniBatchKMeans is not None:
        kmeans = MiniBatchKMeans(n_clusters=codebook_size, batch_size=4096, random_state=seed, n_init="auto")
        seen = 0
        for imgs, _lbls in loader:
            x = patches_from(imgs).numpy().astype(np.float32)
            kmeans.partial_fit(x)
            seen += int(imgs.size(0))
            if max_images and seen >= max_images:
                break
        centroids = kmeans.cluster_centers_.astype(np.float32)
        return PatchKMeansTokenizer(
            img_size=img_size,
            patch_size=patch_size,
            codebook_size=codebook_size,
            centroids=centroids,
        )

    # ---- torch fallback ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[tokenizer] sklearn not found; using torch mini-batch k-means fallback", flush=True)

    # init centroids from first few batches
    init_buf: List[torch.Tensor] = []
    need = max(codebook_size * 4, 4096)
    for imgs, _lbls in loader:
        init_buf.append(patches_from(imgs))
        if sum(x.size(0) for x in init_buf) >= need:
            break
    if not init_buf:
        raise RuntimeError("No data to fit tokenizer.")
    init = torch.cat(init_buf, dim=0)
    # random subset for initialization
    perm = torch.randperm(init.size(0))[: max(codebook_size, 1)]
    centroids_t = init[perm].to(device)
    counts = torch.zeros(codebook_size, device=device, dtype=torch.float32)

    def assign(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [N,D] cpu -> device; returns [N] assignments
        x = x.to(device, non_blocking=True)
        # compute squared L2 distance to centroids; chunk for memory
        N = x.size(0)
        bs = 4096
        out = []
        for i in range(0, N, bs):
            xi = x[i : i + bs]
            # (xi-c)^2 = xi^2 + c^2 - 2 xi·c
            xi2 = (xi * xi).sum(dim=1, keepdim=True)
            c2 = (c * c).sum(dim=1, keepdim=True).T
            d2 = xi2 + c2 - 2.0 * (xi @ c.T)
            out.append(torch.argmin(d2, dim=1))
        return torch.cat(out, dim=0)

    seen_imgs = 0
    for imgs, _lbls in loader:
        x = patches_from(imgs)
        a = assign(x, centroids_t)  # [N]
        x_d = x.to(device, non_blocking=True)
        # batch centroid updates (EMA by cluster counts)
        for k in range(codebook_size):
            m = (a == k)
            if not torch.any(m):
                continue
            pts = x_d[m]
            n = float(pts.size(0))
            counts[k] += n
            # learning rate decays with counts
            lr_k = 1.0 / max(1.0, counts[k].item())
            centroids_t[k] = (1.0 - lr_k) * centroids_t[k] + lr_k * pts.mean(dim=0)
        seen_imgs += int(imgs.size(0))
        if max_images and seen_imgs >= max_images:
            break

    centroids = centroids_t.detach().cpu().numpy().astype(np.float32)
    return PatchKMeansTokenizer(
        img_size=img_size,
        patch_size=patch_size,
        codebook_size=codebook_size,
        centroids=centroids,
    )


# -----------------------------
# GPT over tokens
# -----------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, nh, T, hd]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, int(mlp_ratio * n_embd)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * n_embd), n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TokenGPT(nn.Module):
    def __init__(self, vocab: int, seq_len: int, n_embd: int, n_head: int, n_layer: int, dropout: float):
        super().__init__()
        self.vocab = vocab
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, mlp_ratio=4.0, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T == self.seq_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        return self.head(x)


@torch.no_grad()
def sample_gpt(
    model: TokenGPT,
    *,
    bos_seq: torch.Tensor,
    steps: int,
    temperature: float = 1.0,
    top_k: int = 0,
    force_pos: Optional[int] = None,
    force_token: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    x = bos_seq.clone()
    B, T = x.shape
    assert steps == T
    for t in range(T):
        if force_pos is not None and force_token is not None and t == int(force_pos):
            x[:, t] = int(force_token)
            continue
        logits = model(x)[:, t, :] / max(1e-6, float(temperature))
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, k=min(int(top_k), logits.shape[-1]))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        x[:, t] = torch.multinomial(probs, num_samples=1).squeeze(1)
    return x


# -----------------------------
# Polysemy probes
# -----------------------------


def label_entropy(labels: np.ndarray, num_classes: int) -> float:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_polysemy_probe(
    *,
    tokenizer: PatchKMeansTokenizer,
    model: TokenGPT,
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    outdir: Path,
    token_id: int,
    max_occ: int,
    gen_samples: int,
    force_pos: int,
    temperature: float,
    top_k: int,
    wandb_on: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Collect occurrences
    occ_patches: List[torch.Tensor] = []
    occ_lbls: List[int] = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    for imgs, lbls in loader:
        tok = tokenizer.encode_images(imgs)
        mask = tok == int(token_id)
        if mask.any():
            idxs = mask.nonzero(as_tuple=False)
            for b, p in idxs.tolist():
                if len(occ_patches) >= max_occ:
                    break
                # crop that patch
                bbox = _patch_bbox(p, tokenizer.grid, tokenizer.patch_size)
                img01 = denorm(imgs[b : b + 1])[0].cpu()
                x0, y0, x1, y1 = bbox
                patch = img01[:, y0:y1, x0:x1]
                patch = F.interpolate(patch.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)[0]
                occ_patches.append(patch)
                occ_lbls.append(int(lbls[b].item()))
            if len(occ_patches) >= max_occ:
                break
    if occ_lbls:
        ent = label_entropy(np.array(occ_lbls), num_classes=num_classes)
    else:
        ent = 0.0

    if occ_patches:
        grid_path = outdir / f"token_{token_id:05d}_occurrences.png"
        save_image_grid(torch.stack(occ_patches, dim=0), grid_path, nrow=8)
    else:
        grid_path = None

    # Forced-token generation
    device = next(model.parameters()).device
    bos = torch.zeros(gen_samples, tokenizer.seq_len, dtype=torch.long, device=device)
    forced = sample_gpt(
        model,
        bos_seq=bos,
        steps=tokenizer.seq_len,
        temperature=temperature,
        top_k=top_k,
        force_pos=force_pos,
        force_token=int(token_id),
    ).cpu()
    gen_imgs = tokenizer.decode_tokens(forced)
    gen_path = outdir / f"token_{token_id:05d}_forced_gen.png"
    save_image_grid(gen_imgs, gen_path, nrow=8)

    if wandb_on and wandb is not None:
        log = {
            "polysemy_token/id": int(token_id),
            "polysemy_token/label_entropy": float(ent),
            "polysemy_token/force_pos": int(force_pos),
            "polysemy_token/forced_gen": wandb.Image(str(gen_path)),
        }
        if grid_path is not None:
            log["polysemy_token/occurrences"] = wandb.Image(str(grid_path))
        wandb.log(log)


def _patch_bbox(p: int, grid: int, patch_size: int) -> Tuple[int, int, int, int]:
    r, c = divmod(int(p), int(grid))
    x0 = c * patch_size
    y0 = r * patch_size
    return x0, y0, x0 + patch_size, y0 + patch_size


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImageGPT-style discrete-token modeling + polysemy probes")
    p.add_argument("--dataset", type=str, default="STL10", choices=["STL10", "CIFAR10", "CIFAR100"])
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--codebook", type=int, default=1024)
    p.add_argument("--tokenizer-path", type=str, default="./runs/imagegpt/tokenizer.npy")
    p.add_argument("--outdir", type=str, default="./runs/imagegpt/run")
    p.add_argument("--seed", type=int, default=1337)

    sub = p.add_subparsers(dest="cmd", required=True)

    s_fit = sub.add_parser("fit-tokenizer")
    s_fit.add_argument("--max-images", type=int, default=5000)
    s_fit.add_argument("--batch-size", type=int, default=64)

    s_train = sub.add_parser("train-gpt")
    s_train.add_argument("--epochs", type=int, default=20)
    s_train.add_argument("--batch-size", type=int, default=64)
    s_train.add_argument("--lr", type=float, default=3e-4)
    s_train.add_argument("--dropout", type=float, default=0.1)
    s_train.add_argument("--n-embd", type=int, default=384)
    s_train.add_argument("--n-head", type=int, default=6)
    s_train.add_argument("--n-layer", type=int, default=6)
    s_train.add_argument("--wandb", action="store_true")
    s_train.add_argument("--project", type=str, default="imagegpt_polysemy")
    s_train.add_argument("--run-name", type=str, default=None)
    s_train.add_argument("--save-every", type=int, default=1)

    s_probe = sub.add_parser("probe-polysemy")
    s_probe.add_argument("--ckpt", type=str, required=True)
    s_probe.add_argument("--token-id", type=int, default=-1, help="If -1, pick a random token")
    s_probe.add_argument("--max-occ", type=int, default=64)
    s_probe.add_argument("--gen-samples", type=int, default=64)
    s_probe.add_argument("--force-pos", type=int, default=-1, help="If -1, force center patch")
    s_probe.add_argument("--temperature", type=float, default=1.0)
    s_probe.add_argument("--top-k", type=int, default=64)
    s_probe.add_argument("--wandb", action="store_true")
    s_probe.add_argument("--project", type=str, default="imagegpt_polysemy")
    s_probe.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


def train_gpt(
    *,
    tokenizer: PatchKMeansTokenizer,
    train_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    num_classes: int,
    outdir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    n_embd: int,
    n_head: int,
    n_layer: int,
    wandb_on: bool,
    project: str,
    run_name: Optional[str],
    save_every: int,
    seed: int,
) -> None:
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = TokenGPT(
        vocab=tokenizer.codebook_size,
        seq_len=tokenizer.seq_len,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    if wandb_on and wandb is not None:
        wandb.init(project=project, name=run_name)
        wandb.config.update(
            {
                "dataset": "TOKEN_" + str(num_classes),
                "img_size": tokenizer.img_size,
                "patch_size": tokenizer.patch_size,
                "codebook": tokenizer.codebook_size,
                "epochs": epochs,
                "lr": lr,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
            }
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    def run_eval() -> float:
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for imgs, _lbls in test_loader:
                tok = tokenizer.encode_images(imgs).to(device)
                logits = model(tok)
                loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), tok[:, 1:].reshape(-1))
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float("nan")

    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        for imgs, _lbls in train_loader:
            tok = tokenizer.encode_images(imgs).to(device)
            logits = model(tok)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), tok[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        eval_loss = run_eval()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[gpt] epoch {epoch:03d} train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}", flush=True)
        if wandb_on and wandb is not None:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": eval_loss})

        if (epoch % max(1, save_every) == 0) or (epoch == epochs - 1):
            ckpt = {
                "model": model.state_dict(),
                "tokenizer_path": str(outdir / "tokenizer.npy"),
                "cfg": {
                    "vocab": tokenizer.codebook_size,
                    "seq_len": tokenizer.seq_len,
                    "n_embd": n_embd,
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "dropout": dropout,
                },
            }
            torch.save(ckpt, outdir / f"gpt_epoch_{epoch:03d}.pt")

        # quick sample
        if (epoch % max(1, save_every) == 0) or (epoch == epochs - 1):
            with torch.no_grad():
                bos = torch.zeros(16, tokenizer.seq_len, dtype=torch.long, device=device)
                samp = sample_gpt(model, bos_seq=bos, steps=tokenizer.seq_len, temperature=1.0, top_k=64).cpu()
                imgs01 = tokenizer.decode_tokens(samp)
                sample_path = outdir / f"samples_epoch_{epoch:03d}.png"
                save_image_grid(imgs01, sample_path, nrow=8)
                if wandb_on and wandb is not None:
                    wandb.log({"samples": wandb.Image(str(sample_path)), "epoch": epoch})

    if wandb_on and wandb is not None:
        wandb.finish()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    tok_path = Path(args.tokenizer_path)
    train_ds, test_ds, num_classes = make_dataset(args.dataset, args.root, args.img_size)

    if args.cmd == "fit-tokenizer":
        tok = fit_tokenizer(
            dataset=train_ds,
            img_size=args.img_size,
            patch_size=args.patch_size,
            codebook_size=args.codebook,
            max_images=args.max_images,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(tok_path)
        print(f"[tokenizer] saved → {tok_path}")
        return

    if not tok_path.exists():
        raise RuntimeError(
            f"Tokenizer not found at {tok_path}. Run: python src/imagegpt.py fit-tokenizer ..."
        )
    tok = PatchKMeansTokenizer.load(tok_path)

    if args.cmd == "train-gpt":
        # keep a copy of tokenizer with the run
        outdir.mkdir(parents=True, exist_ok=True)
        tok.save(outdir / "tokenizer.npy")
        train_gpt(
            tokenizer=tok,
            train_ds=train_ds,
            test_ds=test_ds,
            num_classes=num_classes,
            outdir=outdir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            wandb_on=args.wandb,
            project=args.project,
            run_name=args.run_name,
            save_every=args.save_every,
            seed=args.seed,
        )
        return

    if args.cmd == "probe-polysemy":
        ckpt = torch.load(args.ckpt, map_location="cpu")
        cfg = ckpt["cfg"]
        model = TokenGPT(
            vocab=int(cfg["vocab"]),
            seq_len=int(cfg["seq_len"]),
            n_embd=int(cfg["n_embd"]),
            n_head=int(cfg["n_head"]),
            n_layer=int(cfg["n_layer"]),
            dropout=float(cfg["dropout"]),
        )
        model.load_state_dict(ckpt["model"])
        model.to(get_device())

        if args.wandb and wandb is not None:
            wandb.init(project=args.project, name=args.run_name)

        token_id = args.token_id
        if token_id < 0:
            token_id = random.randint(0, tok.codebook_size - 1)
        force_pos = args.force_pos
        if force_pos < 0:
            force_pos = tok.seq_len // 2

        run_polysemy_probe(
            tokenizer=tok,
            model=model,
            dataset=test_ds,
            num_classes=num_classes,
            outdir=outdir,
            token_id=int(token_id),
            max_occ=int(args.max_occ),
            gen_samples=int(args.gen_samples),
            force_pos=int(force_pos),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            wandb_on=bool(args.wandb),
        )

        if args.wandb and wandb is not None:
            wandb.finish()
        return


if __name__ == "__main__":
    main()
