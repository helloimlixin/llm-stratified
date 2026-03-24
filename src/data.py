"""Dataset and dataloader utilities for TinyViT training."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

__all__ = [
    "FlatImageDataset",
    "COCO2017MultiLabel",
    "VOCMultiLabel",
    "WithIndex",
    "build_dataset",
    "get_norm_stats",
    "make_loaders",
    "resolve_class_names",
]

# Normalization constants as tuples (for torchvision transforms)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Default image sizes per dataset
DEFAULT_IMG_SIZES = {
    "CIFAR10": 32, "CIFAR100": 32, "STL10": 96, "FOOD101": 224, "FLOWERS102": 224,
    "CELEBA": 64, "CELEBAHQ": 256, "SVHN": 32, "IMAGENET": 256, "FFHQ": 256,
    "VOC": 224, "VOC2007": 224, "VOC2012": 224, "COCO": 224, "COCO2017": 224,
    "FAKEDATA": 32,
}

# Known class names for specific datasets
KNOWN_CLASS_NAMES = {
    "CIFAR10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "FFHQ": ["face"],
    "CELEBAHQ": ["face"],
}

# VOC classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def get_norm_stats(dataset: str, device: torch.device | None = None, as_tensor: bool = True):
    """Get normalization mean/std for a dataset.

    Args:
        dataset: Dataset name (case-insensitive)
        device: Target device for tensors
        as_tensor: If True, return torch.Tensor; if False, return tuple

    Returns:
        Tuple of (mean, std) as tensors or tuples
    """
    is_cifar = dataset.upper() in ["CIFAR10", "CIFAR100", "SVHN"]
    mean = CIFAR_MEAN if is_cifar else IMAGENET_MEAN
    std = CIFAR_STD if is_cifar else IMAGENET_STD

    if as_tensor:
        mean_t = torch.tensor(mean)
        std_t = torch.tensor(std)
        if device is not None:
            mean_t = mean_t.to(device)
            std_t = std_t.to(device)
        return mean_t, std_t
    return mean, std


MANUAL_NO_DOWNLOAD_DATASETS = {
    "IMAGENET",
    "FFHQ",
    "VOC",
    "VOC2007",
    "VOC2012",
    "PASCALVOC",
    "VOC07",
    "VOC12",
    "COCO",
    "COCO2017",
}


def _make_transforms(norm_mean, norm_std, img_size: int, crop_pad: int = 4, heavy: bool = False):
    """Create train and test transforms."""
    train_tf = [
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)) if heavy else T.RandomCrop(img_size, padding=crop_pad),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_std),
    ]
    test_tf = [T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(norm_mean, norm_std)]
    return T.Compose(train_tf), T.Compose(test_tf)


def _voc_root_ok(root_dir: str, year: str) -> bool:
    voc_dir = Path(root_dir) / "VOCdevkit" / f"VOC{year}"
    return (voc_dir / "JPEGImages").is_dir() and (voc_dir / "Annotations").is_dir()


def _resolve_coco2017_paths(root_dir: str) -> tuple[Path, Path, Path, Path, Path]:
    """Locate COCO2017 train/val image dirs + annotation JSONs."""
    root_p = Path(root_dir)
    for base in [root_p, root_p / "coco", root_p / "COCO", root_p / "coco2017", root_p / "COCO2017"]:
        ann_dir = base / "annotations"
        ann_train, ann_val = ann_dir / "instances_train2017.json", ann_dir / "instances_val2017.json"
        for img_base in [base, base / "images"]:
            img_train, img_val = img_base / "train2017", img_base / "val2017"
            if all(p.exists() for p in [img_train, img_val, ann_train, ann_val]):
                return base, img_train, img_val, ann_train, ann_val
    raise RuntimeError(
        "COCO2017 not found. Expected: <root>/coco/{train2017,val2017,annotations/instances_*.json}\n"
        "Download from https://cocodataset.org/#download"
    )


class COCO2017MultiLabel(Dataset):
    """COCO instances -> multi-hot category vector for image-level classification."""

    def __init__(self, *, img_dir: Path, ann_file: Path, transform=None, classes=None, cat_id_to_idx=None):
        self.img_dir, self.transform = Path(img_dir), transform
        self._decode_failures: set[int] = set()
        self._warned_failures: set[int] = set()
        with open(ann_file, "r") as fp:
            data = json.load(fp)
        cats = data.get("categories", [])
        if cat_id_to_idx is None:
            cats_sorted = sorted(cats, key=lambda c: int(c.get("id", 0)))
            self.classes = [str(c.get("name", c.get("id"))) for c in cats_sorted]
            self.cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(cats_sorted) if "id" in c}
        else:
            self.classes = list(classes) if classes else [str(i) for i in range(len(cat_id_to_idx))]
            self.cat_id_to_idx = {int(k): int(v) for k, v in cat_id_to_idx.items()}
        images = data.get("images", [])
        self.items, self.orig_sizes = [], []
        id_to_pos = {}
        for im in images:
            try:
                id_to_pos[int(im["id"])] = len(self.items)
                self.items.append((int(im["id"]), str(im["file_name"])))
                self.orig_sizes.append((int(im.get("width", 0)), int(im.get("height", 0))))
            except Exception:
                continue
        self.cats_by_pos = [set() for _ in self.items]
        self.instances_by_pos = [[] for _ in self.items]
        for ann in data.get("annotations", []):
            try:
                pos = id_to_pos.get(int(ann["image_id"]))
                j = self.cat_id_to_idx.get(int(ann["category_id"]))
                if pos is None or j is None:
                    continue
                self.cats_by_pos[pos].add(j)
                bb = ann.get("bbox")
                if bb and len(bb) >= 4 and bb[2] > 0 and bb[3] > 0:
                    self.instances_by_pos[pos].append((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3], j))
            except Exception:
                continue

    def instances_after_eval_transform(self, idx: int, out_size: int):
        """Return instance boxes transformed for eval (Resize + CenterCrop)."""
        w0, h0 = self.orig_sizes[idx] if idx < len(self.orig_sizes) else (0, 0)
        if w0 <= 0 or h0 <= 0:
            return []
        scale = out_size / min(h0, w0)
        w1, h1 = int(round(w0 * scale)), int(round(h0 * scale))
        left, top = max(0, (w1 - out_size) // 2), max(0, (h1 - out_size) // 2)
        out = []
        for x0, y0, x1, y1, cat in self.instances_by_pos[idx]:
            x0c = max(0, min(out_size, x0 * scale - left))
            x1c = max(0, min(out_size, x1 * scale - left))
            y0c = max(0, min(out_size, y0 * scale - top))
            y1c = max(0, min(out_size, y1 * scale - top))
            if x1c > x0c and y1c > y0c:
                out.append((x0c, y0c, x1c, y1c, cat))
        return out

    def __len__(self) -> int:
        return len(self.items)

    def _load_rgb_image(self, idx: int):
        path = self.img_dir / self.items[idx][1]
        for attempt in range(3):
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise

    def _resolve_valid_index(self, idx: int) -> tuple[int, Image.Image]:
        total = len(self.items)
        if total <= 0:
            raise IndexError("COCO dataset is empty")

        last_error: Exception | None = None
        for offset in range(total):
            candidate = (int(idx) + offset) % total
            try:
                img = self._load_rgb_image(candidate)
                return candidate, img
            except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
                self._decode_failures.add(candidate)
                if candidate not in self._warned_failures:
                    print(
                        f"[data] skipping unreadable COCO image: {self.img_dir / self.items[candidate][1]} ({exc})",
                        flush=True,
                    )
                    self._warned_failures.add(candidate)
                last_error = exc
                continue

        raise RuntimeError(f"Unable to decode any COCO images under {self.img_dir}") from last_error

    def __getitem__(self, idx: int):
        resolved_idx, img = self._resolve_valid_index(int(idx))
        y = torch.zeros(len(self.classes), dtype=torch.float32)
        for j in self.cats_by_pos[resolved_idx]:
            y[j] = 1.0
        x = self.transform(img) if self.transform else img
        return x, y, resolved_idx


class VOCMultiLabel(Dataset):
    """VOC detection -> multi-hot class vector."""

    def __init__(self, base: Dataset):
        self.base = base
        self.class_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, target = self.base[idx]
        y = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        try:
            objs = target.get("annotation", {}).get("object", [])
            if isinstance(objs, dict):
                objs = [objs]
            for obj in objs:
                name = obj.get("name")
                if name in self.class_to_idx:
                    y[self.class_to_idx[name]] = 1.0
        except Exception:
            pass
        return img, y


class FlatImageDataset(Dataset):
    """Image dataset for a flat directory of images, with a deterministic split."""

    def __init__(self, root: str, transform=None, *, split: str = "train", val_fraction: float = 0.1):
        self.root = Path(root)
        self.transform = transform
        self.split = str(split).lower()
        self.val_fraction = float(val_fraction)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted(p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in exts)
        if not files:
            raise RuntimeError(f"No image files found in {self.root}")

        val_count = max(1, int(round(len(files) * self.val_fraction)))
        val_count = min(val_count, max(1, len(files) - 1))
        split_idx = len(files) - val_count
        if self.split in {"train", "trainval"}:
            self.files = files[:split_idx]
        elif self.split in {"val", "valid", "validation", "test"}:
            self.files = files[split_idx:]
        else:
            raise ValueError(f"Unsupported split for FlatImageDataset: {split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        x = self.transform(img) if self.transform else img
        return x, 0


def build_dataset(
    name: str = "CIFAR10",
    root: str = "./data",
    img_size: Optional[int] = None,
    split_celebA: str = "train",
    download: bool = True,
) -> Tuple[Dataset, Dataset, int, int, int, str]:
    """Construct a torchvision dataset pair plus metadata."""
    name = name.upper()
    img_size = img_size or DEFAULT_IMG_SIZES.get(name, 32)

    if name == "CIFAR10":
        train_tf, test_tf = _make_transforms(CIFAR_MEAN, CIFAR_STD, img_size)
        train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_tf)
        return train_ds, test_ds, 10, 3, img_size, "multiclass"

    if name == "CIFAR100":
        train_tf, test_tf = _make_transforms(CIFAR_MEAN, CIFAR_STD, img_size)
        train_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=test_tf)
        return train_ds, test_ds, 100, 3, img_size, "multiclass"

    if name == "STL10":
        train_tf, test_tf = _make_transforms(IMAGENET_MEAN, IMAGENET_STD, img_size, crop_pad=8, heavy=True)
        train_ds = torchvision.datasets.STL10(root=root, split="train", download=download, transform=train_tf)
        test_ds = torchvision.datasets.STL10(root=root, split="test", download=download, transform=test_tf)
        return train_ds, test_ds, 10, 3, img_size, "multiclass"

    if name == "FOOD101":
        train_tf, test_tf = _make_transforms(IMAGENET_MEAN, IMAGENET_STD, img_size, crop_pad=16, heavy=True)
        train_ds = torchvision.datasets.Food101(root=root, split="train", download=download, transform=train_tf)
        test_ds = torchvision.datasets.Food101(root=root, split="test", download=download, transform=test_tf)
        return train_ds, test_ds, 101, 3, img_size, "multiclass"

    if name == "FLOWERS102":
        train_tf, test_tf = _make_transforms(IMAGENET_MEAN, IMAGENET_STD, img_size, crop_pad=16, heavy=True)
        train_ds = torchvision.datasets.Flowers102(root=root, split="train", download=download, transform=train_tf)
        test_ds = torchvision.datasets.Flowers102(root=root, split="test", download=download, transform=test_tf)
        return train_ds, test_ds, 102, 3, img_size, "multiclass"

    if name == "SVHN":
        train_tf, test_tf = _make_transforms(CIFAR_MEAN, CIFAR_STD, img_size)
        train_ds = torchvision.datasets.SVHN(root=root, split="train", download=download, transform=train_tf)
        test_ds = torchvision.datasets.SVHN(root=root, split="test", download=download, transform=test_tf)
        return train_ds, test_ds, 10, 3, img_size, "multiclass"

    if name == "IMAGENET":
        train_tf = T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_tf = T.Compose(
            [
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        train_ds = torchvision.datasets.ImageNet(root=root, split="train", transform=train_tf)
        test_ds = torchvision.datasets.ImageNet(root=root, split="val", transform=test_tf)
        return train_ds, test_ds, 1000, 3, img_size, "multiclass"

    if name == "FFHQ":
        train_tf = T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_tf = T.Compose(
            [
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        train_root = os.path.join(root, "train") if os.path.isdir(os.path.join(root, "train")) else root
        val_root = os.path.join(root, "val") if os.path.isdir(os.path.join(root, "val")) else root
        train_ds = torchvision.datasets.ImageFolder(root=train_root, transform=train_tf)
        test_ds = torchvision.datasets.ImageFolder(root=val_root, transform=test_tf)
        return train_ds, test_ds, len(train_ds.classes), 3, img_size, "multiclass"

    if name == "CELEBAHQ":
        train_tf = T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_tf = T.Compose(
            [
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        train_ds = FlatImageDataset(root=root, split="train", transform=train_tf)
        test_ds = FlatImageDataset(root=root, split="test", transform=test_tf)
        return train_ds, test_ds, 1, 3, img_size, "multiclass"

    if name in {"FAKEDATA", "FAKE"}:
        train_tf, test_tf = _make_transforms(IMAGENET_MEAN, IMAGENET_STD, img_size)
        train_ds = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, img_size, img_size),
            num_classes=10,
            transform=train_tf,
        )
        test_ds = torchvision.datasets.FakeData(
            size=64,
            image_size=(3, img_size, img_size),
            num_classes=10,
            transform=test_tf,
        )
        return train_ds, test_ds, 10, 3, img_size, "multiclass"

    if name == "CELEBA":
        def celebA_transforms(train: bool = True):
            ops = [T.CenterCrop(178)]
            if train:
                ops += [
                    T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.2, 0.2, 0.2),
                ]
            else:
                ops += [T.Resize(img_size), T.CenterCrop(img_size)]
            return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        train_ds = torchvision.datasets.CelebA(
            root=root, split=split_celebA, download=download, transform=celebA_transforms(True)
        )
        test_ds = torchvision.datasets.CelebA(root=root, split="test", download=download, transform=celebA_transforms(False))
        return train_ds, test_ds, 40, 3, img_size, "multilabel"

    if name in {"COCO", "COCO2017"}:
        train_tf, test_tf = _make_transforms(IMAGENET_MEAN, IMAGENET_STD, img_size, crop_pad=16, heavy=True)
        _, train_img, val_img, ann_train, ann_val = _resolve_coco2017_paths(root)
        train_ds = COCO2017MultiLabel(img_dir=train_img, ann_file=ann_train, transform=train_tf)
        test_ds = COCO2017MultiLabel(
            img_dir=val_img,
            ann_file=ann_val,
            transform=test_tf,
            classes=train_ds.classes,
            cat_id_to_idx=train_ds.cat_id_to_idx,
        )
        return train_ds, test_ds, len(train_ds.classes), 3, img_size, "multilabel"

    if name in {"VOC", "VOC2007", "PASCALVOC", "VOC07"}:
        train_tf, test_tf = _make_transforms(IMAGENET_MEAN, IMAGENET_STD, img_size, crop_pad=16, heavy=True)
        if not _voc_root_ok(root, "2007"):
            raise RuntimeError(
                f"Pascal VOC 2007 not found. Expected: {root}/VOCdevkit/VOC2007/{{JPEGImages,Annotations}}/"
            )
        base_train = torchvision.datasets.VOCDetection(
            root=root, year="2007", image_set="trainval", download=False, transform=train_tf
        )
        base_test = torchvision.datasets.VOCDetection(
            root=root, year="2007", image_set="test", download=False, transform=test_tf
        )
        return VOCMultiLabel(base_train), VOCMultiLabel(base_test), 20, 3, img_size, "multilabel"

    if name in {"VOC2012", "VOC12"}:
        train_tf, test_tf = _make_transforms(IMAGENET_MEAN, IMAGENET_STD, img_size, crop_pad=16, heavy=True)
        if not _voc_root_ok(root, "2012"):
            raise RuntimeError(
                f"Pascal VOC 2012 not found. Expected: {root}/VOCdevkit/VOC2012/{{JPEGImages,Annotations}}/"
            )
        base_train = torchvision.datasets.VOCDetection(
            root=root, year="2012", image_set="train", download=False, transform=train_tf
        )
        base_test = torchvision.datasets.VOCDetection(
            root=root, year="2012", image_set="val", download=False, transform=test_tf
        )
        return VOCMultiLabel(base_train), VOCMultiLabel(base_test), 20, 3, img_size, "multilabel"

    raise ValueError(f"Unknown dataset: {name}")


def resolve_class_names(dataset: Dataset, dataset_name: str) -> Optional[List[str]]:
    name = dataset_name.upper()
    if name == "CELEBA" and hasattr(dataset, "attr_names"):
        try:
            return list(getattr(dataset, "attr_names"))
        except Exception:
            pass
    if name in KNOWN_CLASS_NAMES:
        return KNOWN_CLASS_NAMES[name]
    if name in {"VOC", "VOC2007", "VOC2012", "PASCALVOC", "VOC07", "VOC12"}:
        return VOC_CLASSES
    if hasattr(dataset, "classes"):
        return dataset.classes
    if hasattr(dataset, "dataset"):
        return resolve_class_names(dataset.dataset, name)
    return None


class WithIndex(Dataset):
    """Dataset wrapper that appends a stable base-dataset index as the 3rd return value."""

    def __init__(self, base: Dataset):
        self.dataset = base

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        out = self.dataset[int(idx)]
        img, y = out[0], out[1]
        if len(out) > 2:
            return img, y, int(out[2])
        gidx, base = int(idx), self.dataset
        while isinstance(base, Subset):
            gidx = int(base.indices[gidx])
            base = base.dataset
        return img, y, gidx

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


def make_loaders(
    dataset_name: str = "CIFAR10",
    root: str = "./data",
    img_size: Optional[int] = None,
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    num_workers: int = 4,
    subset_train: Optional[int] = None,
    subset_test: Optional[int] = None,
    device: Optional[torch.device] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, int, int, int, str]:
    """Instantiate training and evaluation data loaders."""
    name_u = dataset_name.upper()

    def _barrier():
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        except TypeError:
            dist.barrier()

    if distributed and dist.is_available() and dist.is_initialized():
        if name_u in MANUAL_NO_DOWNLOAD_DATASETS:
            train_ds, test_ds, num_classes, in_chans, img_size, task = build_dataset(
                dataset_name, root, img_size, download=False
            )
        else:
            if rank == 0:
                train_ds, test_ds, num_classes, in_chans, img_size, task = build_dataset(
                    dataset_name, root, img_size, download=True
                )
            _barrier()
            if rank != 0:
                train_ds, test_ds, num_classes, in_chans, img_size, task = build_dataset(
                    dataset_name, root, img_size, download=False
                )
            _barrier()
    else:
        train_ds, test_ds, num_classes, in_chans, img_size, task = build_dataset(
            dataset_name, root, img_size, download=True
        )

    if subset_train and subset_train > 0:
        train_ds = Subset(train_ds, list(range(min(subset_train, len(train_ds)))))
    if subset_test and subset_test > 0:
        test_ds = Subset(test_ds, list(range(min(subset_test, len(test_ds)))))

    if name_u in {"COCO", "COCO2017"}:
        train_ds, test_ds = WithIndex(train_ds), WithIndex(test_ds)

    pin = device is not None and device.type == "cuda"
    persistent_workers = num_workers > 0

    samples_per_rank = len(train_ds) // max(1, world_size) if distributed else len(train_ds)
    drop_last_train = samples_per_rank >= batch_size_train

    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last_train
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        shuffle_train = False
    else:
        train_sampler, test_sampler, shuffle_train = None, None, True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=shuffle_train,
        sampler=train_sampler,
        drop_last=drop_last_train,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_test,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers,
    )
    return train_loader, test_loader, num_classes, in_chans, img_size, task
