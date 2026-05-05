"""Main orchestration for per-epoch fiber analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from utils import to_serializable

from fiber.geometry import analyze_stratification, summarize_stratification
from fiber.hypothesis import (
    estimate_neighborhood_dimensions,
    format_hypothesis_summary_line,
    summarize_class_dimensions,
    summarize_hypothesis_metrics,
)
from fiber.sparse_probe import run_sparse_dictionary_probe
from fiber.plots import (
    HAS_MATPLOTLIB,
    _maybe_wandb_metric_table,
    _singular_token_mask,
    _wandb_image_or_none,
    add_heatmap_patch,
    build_embedding_scatter_figure,
    build_tsne_embedding_figure,
    project_embeddings_pca_3d,
    project_embeddings_tsne_3d,
    select_irregular_tokens,
    select_singular_tokens,
    save_polysemy_entropy_scatter_plot,
    save_polysemy_irregularity_plot,
    _require_matplotlib,
)
from fiber.polysemy import (
    _cluster_label_counts_from_loader,
    _cluster_label_entropy,
    _sample_patch_embeddings_from_loader,
    _stats_from_count_matrix,
    _torch_kmeans,
    _assign_centroids,
    compute_token_polysemy_entropy_scores,
    compute_token_polysemy_for_anchors,
    make_polysemy_entropy_triptychs,
    make_polysemy_gallery,
    select_polysemy_entropy_images,
)
from fiber.ablation import (
    _eval_ablation_controls,
    compute_masked_classification_effects,
)


def _finite_metric(value: Any) -> float:
    try:
        value_f = float(value)
    except Exception:
        return float("nan")
    return value_f if math.isfinite(value_f) else float("nan")


def _fmt_metric(value: Any, *, digits: int = 2, default: str = "n/a") -> str:
    value_f = _finite_metric(value)
    if not math.isfinite(value_f):
        return default
    return f"{value_f:.{digits}f}"


def _wandb_image_with_caption(wandb_module, image: Any, caption: str):
    try:
        return wandb_module.Image(image, caption=caption)
    except TypeError:
        return wandb_module.Image(image)


def _embedding_plot_caption(
    *,
    epoch: int,
    projection: str,
    fiber_summary: Dict[str, Any],
    hypothesis_summary: Dict[str, Any],
) -> str:
    mean_dim = _finite_metric(fiber_summary.get("mean_dim"))
    median_dim = _finite_metric(fiber_summary.get("median_dim"))
    irregular_ratio = _finite_metric(fiber_summary.get("irregular_ratio"))
    same_image = _finite_metric(hypothesis_summary.get("same_image_neighbor_ratio_mean"))
    same_chance = _finite_metric(hypothesis_summary.get("same_image_neighbor_chance"))
    local_chart = _finite_metric(hypothesis_summary.get("local_chart_score"))
    stage = str(hypothesis_summary.get("hypothesis_stage", "unclassified"))

    if math.isfinite(irregular_ratio) and irregular_ratio >= 0.45:
        regularity_text = "a large rejected-token fraction, so local dimension is not behaving like one smooth uniform manifold"
    elif math.isfinite(irregular_ratio) and irregular_ratio >= 0.20:
        regularity_text = "a visible but not dominant singular set, so the representation looks mixed rather than uniformly regular"
    else:
        regularity_text = "a relatively small rejected-token fraction, so the sampled neighborhoods look closer to locally regular"

    if math.isfinite(same_image) and math.isfinite(same_chance):
        if same_image >= max(0.30, 5.0 * same_chance):
            neighbor_text = "nearest-neighbor structure is strongly image-local"
        elif same_image >= 2.0 * same_chance:
            neighbor_text = "nearest-neighbor structure is somewhat more image-local than chance"
        else:
            neighbor_text = "nearest-neighbor structure is not much more image-local than chance"
    else:
        neighbor_text = "same-image neighbor structure is unavailable"

    return (
        f"Epoch {epoch} {projection} projection. Points are tokens and color encodes estimated local dimension; coherent color regions indicate smooth dimension fields, while speckled color changes indicate stratification or abrupt local geometry changes. "
        f"Conclusion: this run has mean/median dimension {_fmt_metric(mean_dim)}/{_fmt_metric(median_dim)} and irregular ratio {_fmt_metric(irregular_ratio, digits=3)}, which indicates {regularity_text}. "
        f"The hypothesis stage is {stage}; {neighbor_text}, and the local chart score is {_fmt_metric(local_chart, digits=3)}."
    )


def _polysemy_anchor_caption(*, epoch: int, polysemy_result: Dict[str, Any]) -> str:
    anchors = [item for item in polysemy_result.get("anchors", []) if item]
    entropies = [_finite_metric(item.get("label_entropy")) for item in anchors]
    entropies = [value for value in entropies if math.isfinite(value)]
    top_shares = [_finite_metric(item.get("top_label_share")) for item in anchors]
    top_shares = [value for value in top_shares if math.isfinite(value)]
    mean_entropy = float(np.mean(entropies)) if entropies else float("nan")
    mean_top_share = float(np.mean(top_shares)) if top_shares else float("nan")
    if math.isfinite(mean_entropy) and mean_entropy >= 1.0:
        conclusion = "many anchor neighborhoods mix labels, so token neighborhoods are semantically polysemous rather than class-pure"
    elif math.isfinite(mean_top_share) and mean_top_share >= 0.80:
        conclusion = "anchor neighborhoods are mostly dominated by one label, so this sample shows limited semantic mixing"
    else:
        conclusion = "semantic mixing is moderate and should be compared against the selected anchor policy"
    return (
        f"Epoch {epoch} polysemy anchor plot. Entropy increases when neighboring tokens come from multiple labels, while top-label share decreases when no single class dominates. "
        f"Conclusion: {conclusion}. Mean anchor entropy is {_fmt_metric(mean_entropy)} and mean top-label share is {_fmt_metric(mean_top_share, digits=3)} across {len(anchors)} anchors."
    )


def _polysemy_irregularity_caption(*, epoch: int, stats: Dict[str, Any]) -> str:
    pearson = _finite_metric(stats.get("pearson_r"))
    spearman = _finite_metric(stats.get("spearman_r"))
    reject_entropy = _finite_metric(stats.get("mean_entropy_reject"))
    ok_entropy = _finite_metric(stats.get("mean_entropy_non_reject"))
    if math.isfinite(spearman) and abs(spearman) >= 0.30:
        direction = "positive" if spearman > 0 else "negative"
        conclusion = f"polysemy entropy has a {direction} association with geometric irregularity"
    elif math.isfinite(reject_entropy) and math.isfinite(ok_entropy) and reject_entropy > ok_entropy:
        conclusion = "rejected neighborhoods have higher average entropy, but the rank correlation is weak"
    else:
        conclusion = "semantic label mixing is not strongly aligned with geometric irregularity in this sample"
    return (
        f"Epoch {epoch} entropy-irregularity plot. This checks whether semantically mixed token neighborhoods are also the neighborhoods that fail the local fiber-bundle test. "
        f"Conclusion: {conclusion}. Pearson r is {_fmt_metric(pearson)} and Spearman rho is {_fmt_metric(spearman)}; mean entropy is {_fmt_metric(reject_entropy)} for rejected tokens versus {_fmt_metric(ok_entropy)} for non-rejected tokens."
    )


def _irregular_samples_caption(*, epoch: int, item: Dict[str, Any]) -> str:
    return (
        f"Epoch {epoch} high-irregularity token. Conclusion: this patch is one of the strongest local fiber-test failures, "
        f"with dimension {_fmt_metric(item.get('dim'))}, irregularity {_fmt_metric(item.get('irregularity'))}, "
        f"and neighborhood dimension {_fmt_metric(item.get('neigh_dim'))}. Use these samples to inspect whether failures align with object boundaries, texture, occlusion, or mixed labels."
    )


def analyze_fiber_epoch(
    *,
    epoch: int,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    patch_indices: torch.Tensor | None = None,
    pred_labels: torch.Tensor,
    num_classes: int,
    class_names: List[str] | None,
    dataset: str,
    base_dir: Path,
    analysis_dir: Path,
    embed_dir: Path,
    vol_min: int,
    vol_max: int,
    ws: int,
    alpha: float,
    nstrat: int,
    neighborhood_size: int,
    polysemy: bool = False,
    polysemy_k: int = 48,
    polysemy_anchors: int = 12,
    polysemy_grid_cols: int = 8,
    vit_token_polysemy: bool = False,
    vit_token_polysemy_k: int = 256,
    vit_token_polysemy_topk: int = 16,
    vit_token_polysemy_ablate: bool = True,
    vit_token_polysemy_ablate_batches: int = 10,
    vit_token_polysemy_min_count: int = 50,
    vit_token_polysemy_ablate_reps: int = 5,
    sparse_probe: bool = False,
    sparse_probe_radius: float | None = None,
    sparse_probe_auto_neighbor_k: int = 32,
    sparse_probe_auto_radius_quantile: float = 0.5,
    sparse_probe_min_patches: int = 12,
    sparse_probe_max_anchors: int | None = None,
    sparse_probe_dictionary_size: int = 32,
    sparse_probe_residual_threshold: float = 0.15,
    sparse_probe_max_sparsity: int = 16,
    sparse_probe_algorithm: str = "omp",
    sparse_probe_iht_steps: int = 80,
    sparse_probe_iht_lr: float | None = None,
    sparse_probe_heatmap_images: int = 8,
    wandb_module=None,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    img_size: int | None = None,
    patch_size: int | None = None,
    val_loader: DataLoader | None = None,
) -> Dict[str, Any]:
    """Run fiber bundle diagnostics for one epoch worth of embeddings.

    ``images`` is the **unique** image buffer (one entry per source image).
    ``image_ids`` maps each token to its source image index.
    """

    # Save embeddings
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "images": images,
            "bboxes": bboxes,
            "image_ids": image_ids,
            "pred_labels": pred_labels,
        },
        embed_dir / f"epoch_{epoch:03d}.pt",
    )

    fiber_results, _sorted_dists, unsorted_dists = analyze_stratification(
        embeddings, vol_min=vol_min, vol_max=vol_max, ws=ws, alpha=alpha, nstrat=nstrat
    )
    neighborhood_dims = estimate_neighborhood_dimensions(fiber_results, bboxes, neighborhood_size)

    # Extract first-stratum dimensions once — reused throughout
    final_dims = np.array(
        [r["dimensions"][0] if r.get("dimensions") else np.nan for r in fiber_results]
    )

    # Per-image mean dimension
    mean_dim_by_image: Dict[int, float] = {}
    count_by_image: Dict[int, int] = {}
    img_id_np = image_ids.detach().cpu().numpy()
    for dim_val, iid in zip(final_dims, img_id_np):
        if math.isfinite(dim_val):
            k = int(iid)
            mean_dim_by_image[k] = mean_dim_by_image.get(k, 0) + dim_val
            count_by_image[k] = count_by_image.get(k, 0) + 1
    for k in mean_dim_by_image:
        mean_dim_by_image[k] /= max(1, count_by_image[k])

    fiber_summary = summarize_stratification(fiber_results, alpha=alpha)
    class_dim_means, class_dim_counts = summarize_class_dimensions(fiber_results, labels, num_classes)
    fiber_summary.update({
        "class_dim_means": class_dim_means,
        "class_dim_counts": class_dim_counts,
        "mean_neighborhood_dim": (
            float(np.mean([d for d in neighborhood_dims if math.isfinite(d)]))
            if neighborhood_dims
            else np.nan
        ),
        "neighborhood_size": neighborhood_size,
        "epoch": epoch,
    })

    # Reuse the unsorted distance matrix from the main geometry analysis.
    hypothesis_summary = summarize_hypothesis_metrics(
        embeddings=embeddings,
        fiber_results=fiber_results,
        image_ids=image_ids,
        bboxes=bboxes,
        neighborhood_dims=neighborhood_dims,
        neighborhood_size=neighborhood_size,
        alpha=alpha,
        precomputed_dists=unsorted_dists,
    )
    fiber_summary.update(hypothesis_summary)

    hypothesis_path = analysis_dir / f"epoch_{epoch:03d}_hypothesis_summary.json"
    with open(hypothesis_path, "w") as fp:
        json.dump(to_serializable(hypothesis_summary), fp, indent=2)
    print(format_hypothesis_summary_line(epoch=epoch, metrics=hypothesis_summary), flush=True)
    narrative = hypothesis_summary.get("hypothesis_narrative")
    if narrative:
        print(f"[hypothesis] {narrative}", flush=True)

    sparse_probe_result = None
    if sparse_probe:
        try:
            sparse_probe_result = run_sparse_dictionary_probe(
                epoch=epoch,
                embeddings=embeddings,
                images=images,
                image_ids=image_ids,
                bboxes=bboxes,
                fiber_results=fiber_results,
                dists=unsorted_dists,
                out_dir=analysis_dir,
                patch_size=patch_size,
                radius=sparse_probe_radius,
                auto_neighbor_k=sparse_probe_auto_neighbor_k,
                auto_radius_quantile=sparse_probe_auto_radius_quantile,
                min_patches=sparse_probe_min_patches,
                max_anchors=sparse_probe_max_anchors,
                dictionary_size=sparse_probe_dictionary_size,
                residual_threshold=sparse_probe_residual_threshold,
                max_sparsity=sparse_probe_max_sparsity,
                algorithm=sparse_probe_algorithm,
                iht_steps=sparse_probe_iht_steps,
                iht_lr=sparse_probe_iht_lr,
                heatmap_max_images=sparse_probe_heatmap_images,
            )
            sparse_summary = sparse_probe_result.get("summary", {}) if sparse_probe_result else {}
            for key in (
                "coding_algorithm",
                "radius",
                "candidate_tokens",
                "evaluated_tokens",
                "mean_patch_count",
                "mean_required_sparsity",
                "median_required_sparsity",
                "sparsity_std",
                "sparsity_q10",
                "sparsity_q90",
                "sparsity_iqr",
                "sparsity_range",
                "corr_sparsity_patch_count",
                "corr_sparsity_dimension",
                "corr_sparsity_irregularity",
            ):
                if key in sparse_summary:
                    fiber_summary[f"sparse_probe_{key}"] = sparse_summary[key]
            print(
                "[sparse_probe] "
                f"tokens={sparse_summary.get('evaluated_tokens', 0)}/"
                f"{sparse_summary.get('candidate_tokens', 0)} "
                f"algo={sparse_summary.get('coding_algorithm', sparse_probe_algorithm)} "
                f"eps={float(sparse_summary.get('radius', float('nan'))):.4g} "
                f"mean_sparsity={float(sparse_summary.get('mean_required_sparsity', float('nan'))):.3g} "
                f"range={float(sparse_summary.get('sparsity_range', float('nan'))):.3g}",
                flush=True,
            )
        except Exception as e:
            print(f"[sparse_probe] failed: {e}", flush=True)

    del unsorted_dists  # free O(N^2) memory

    with open(base_dir / f"fiber_epoch_{epoch:03d}.json", "w") as fp:
        json.dump(to_serializable(fiber_results), fp, indent=2)

    final_coords_3d = project_embeddings_pca_3d(embeddings)
    final_tsne_3d, tsne_idx = None, None
    try:
        result = project_embeddings_tsne_3d(embeddings)
        if result:
            final_tsne_3d, tsne_idx = result
    except Exception as e:
        print(f"[tsne] failed: {e}")

    # Polysemy analysis
    polysemy_result = None
    if polysemy:
        try:
            finite_idx = np.where(np.isfinite(final_dims) & (final_dims > 0))[0]
            sorted_idx = (
                finite_idx[np.argsort(final_dims[finite_idx])]
                if finite_idx.size
                else np.array([], dtype=np.int64)
            )
            singular_mask = _singular_token_mask(fiber_results, alpha)
            anchors = select_singular_tokens(
                fiber_results=fiber_results, alpha=alpha, top_k=polysemy_anchors
            )
            if not anchors:
                anchors = [
                    item["token_id"]
                    for item in select_irregular_tokens(
                        images=images,
                        image_ids=image_ids,
                        labels=labels,
                        fiber_results=fiber_results,
                        dataset=dataset,
                        bboxes=bboxes,
                        neighborhood_dims=neighborhood_dims,
                        class_names=class_names,
                        image_mean_dims=mean_dim_by_image,
                        pred_labels=pred_labels, top_k=polysemy_anchors,
                    )
                ]
            if sorted_idx.size and len(anchors) < polysemy_anchors:
                anchors.extend([int(sorted_idx[0]), int(sorted_idx[sorted_idx.size // 2]), int(sorted_idx[-1])])
            anchors = list(dict.fromkeys(anchors))[:polysemy_anchors]
            polysemy_result = compute_token_polysemy_for_anchors(
                embeddings=embeddings, labels=labels, pred_labels=pred_labels,
                images=images, image_ids=image_ids, bboxes=bboxes,
                dataset=dataset, anchor_ids=anchors, k=polysemy_k,
                grid_cols=polysemy_grid_cols, out_dir=analysis_dir,
                prefix=f"epoch_{epoch:03d}", class_names=class_names,
            )
            if polysemy_result is not None:
                singular_count = int(np.sum(singular_mask)) if singular_mask is not None else 0
                total_tokens = len(fiber_results)
                polysemy_result["singular_token_count"] = singular_count
                polysemy_result["singular_token_ratio"] = (
                    singular_count / total_tokens if total_tokens else float("nan")
                )
            top_entropy_sets = make_polysemy_gallery(
                polysemy_result, out_dir=analysis_dir, prefix=f"epoch_{epoch:03d}",
                top_k=min(polysemy_anchors, 8), cols=2,
            )
            if top_entropy_sets:
                polysemy_result["top_entropy_sets"] = top_entropy_sets
            save_polysemy_entropy_scatter_plot(
                polysemy_result, out_dir=analysis_dir, prefix=f"epoch_{epoch:03d}",
                annotate_top=min(polysemy_anchors, 6),
            )
            ent_scores, top_shares, top_labels = compute_token_polysemy_entropy_scores(
                embeddings=embeddings, labels=labels, num_classes=num_classes, k=polysemy_k,
            )
            irreg_path, irreg_stats = save_polysemy_irregularity_plot(
                entropies=ent_scores, fiber_results=fiber_results, out_dir=analysis_dir,
                prefix=f"epoch_{epoch:03d}", alpha=alpha,
            )
            if irreg_path:
                polysemy_result.setdefault("paths", {})["polysemy/entropy_irregularity"] = irreg_path
                polysemy_result["entropy_irregularity_stats"] = irreg_stats
            selection_mask = singular_mask if singular_mask is not None and np.any(singular_mask) else None
            selection = select_polysemy_entropy_images(
                image_ids=image_ids, entropies=ent_scores, bboxes=bboxes,
                top_k_images=9, token_mask=selection_mask,
            )
            selection_source = "singular_tokens" if selection_mask is not None else "all_tokens"
            if not selection and selection_mask is not None:
                selection = select_polysemy_entropy_images(
                    image_ids=image_ids, entropies=ent_scores, bboxes=bboxes, top_k_images=9,
                )
                selection_source = "all_tokens"
            if polysemy_result is not None:
                polysemy_result["entropy_triptych_source"] = selection_source
            entropy_triptych_sets: Dict[str, List[Dict[str, Any]]] = {}
            mask_effects: Dict[str, Dict[str, Any]] = {}
            for mask_mode in ("gray", "black", "blur"):
                triptych_sets = make_polysemy_entropy_triptychs(
                    images=images, image_ids=image_ids, bboxes=bboxes,
                    entropies=ent_scores, labels=labels, class_names=class_names,
                    top_shares=top_shares, top_labels=top_labels, dataset=dataset,
                    out_dir=analysis_dir, prefix=f"epoch_{epoch:03d}",
                    selection=selection, mask_mode=mask_mode,
                )
                if triptych_sets:
                    polysemy_result.setdefault("paths", {})[f"polysemy/entropy_triptychs_{mask_mode}"] = [
                        item["path"] for item in triptych_sets
                    ]
                    entropy_triptych_sets[mask_mode] = triptych_sets
                if model and device and selection:
                    per_image, agg = compute_masked_classification_effects(
                        model=model, device=device, images=images, image_ids=image_ids,
                        labels=labels, selection=selection, dataset=dataset,
                        mask_mode=mask_mode, class_names=class_names, num_classes=num_classes,
                    )
                    if per_image:
                        mask_effects[mask_mode] = {"per_image": per_image, "aggregate": agg}
                        if mask_mode in entropy_triptych_sets:
                            effect_map = {
                                (entry.get("image_id"), entry.get("token_id")): entry
                                for entry in per_image
                            }
                            for item in entropy_triptych_sets[mask_mode]:
                                eff = effect_map.get((item.get("image_id"), item.get("token_id")))
                                if eff:
                                    item.update(eff)
            if entropy_triptych_sets:
                polysemy_result["entropy_triptych_sets"] = entropy_triptych_sets
            if mask_effects:
                polysemy_result["mask_effects"] = mask_effects
        except Exception as e:
            print(f"[polysemy] failed: {e}")

    # ViT token polysemy
    vit_poly_metrics = None
    if vit_token_polysemy and model and device and img_size:
        try:
            analysis_loader = val_loader
            if val_loader and isinstance(getattr(val_loader, "sampler", None), DistributedSampler):
                analysis_loader = DataLoader(
                    val_loader.dataset, batch_size=val_loader.batch_size,
                    shuffle=False, num_workers=getattr(val_loader, "num_workers", 0), drop_last=False,
                )
            x = (
                _sample_patch_embeddings_from_loader(
                    model=model, loader=analysis_loader, device=device, max_tokens=50000,
                )
                if analysis_loader
                else embeddings
            )
            centroids = _torch_kmeans(x, vit_token_polysemy_k, iters=15, seed=0)

            if analysis_loader:
                count_mat = _cluster_label_counts_from_loader(
                    model=model, loader=analysis_loader, centroids=centroids,
                    device=device, num_classes=num_classes, max_batches=None,
                )
                stats = _stats_from_count_matrix(count_mat, smooth=1.0)
            else:
                assign_ids = _assign_centroids(embeddings, centroids).cpu().numpy()
                stats = _cluster_label_entropy(assign_ids, labels.cpu().numpy(), num_classes)

            min_ct = vit_token_polysemy_min_count
            filtered = [(cid, st) for cid, st in stats.items() if st.get("count", 0) >= min_ct]
            if not filtered:
                filtered = list(stats.items())
            ranked = sorted(
                filtered,
                key=lambda kv: (kv[1].get("label_entropy", 0), kv[1].get("count", 0)),
                reverse=True,
            )
            topk = [int(cid) for cid, _ in ranked[:vit_token_polysemy_topk]]

            vit_poly_metrics = {
                "top_clusters": topk,
                "mean_top_entropy": float(np.mean([stats[c]["label_entropy"] for c in topk if c in stats])),
            }

            if vit_token_polysemy_ablate and analysis_loader and patch_size:
                reps = vit_token_polysemy_ablate_reps
                results = [
                    _eval_ablation_controls(
                        model=model, loader=analysis_loader, centroids=centroids,
                        poly_cluster_ids=topk, patch_size=patch_size, img_size=img_size,
                        device=device, batches=vit_token_polysemy_ablate_batches,
                        seed=epoch * 10000 + r,
                    )
                    for r in range(reps)
                ]
                for key in [
                    "acc_drop_poly", "acc_drop_random_clusters", "acc_drop_random_patches",
                    "flip_rate_poly_on_correct", "flip_rate_random_clusters_on_correct",
                    "flip_rate_random_patches_on_correct",
                ]:
                    vals = [r[key] for r in results]
                    vit_poly_metrics[f"vit_polysemy/{key}_mean"] = float(np.mean(vals))
                    vit_poly_metrics[f"vit_polysemy/{key}_std"] = float(np.std(vals))
        except Exception as e:
            print(f"[vit_polysemy] failed: {e}")

    # Wandb logging
    if wandb_module:
        projection_caption = _embedding_plot_caption(
            epoch=epoch,
            projection="PCA",
            fiber_summary=fiber_summary,
            hypothesis_summary=hypothesis_summary,
        )
        log_dict: Dict[str, Any] = {
            "epoch": epoch,
            "media/epoch": epoch,
            "media/log_phase": "fiber_analysis",
            "media/media_index": epoch * 10,
            "fiber/mean_dim": fiber_summary["mean_dim"],
            "fiber/median_dim": fiber_summary.get("median_dim", np.nan),
            "fiber/mean_irregularity": fiber_summary.get("mean_irregularity", np.nan),
            "fiber/irregular_ratio": fiber_summary.get("irregular_ratio", np.nan),
            "fiber/mean_neighborhood_dim": fiber_summary.get("mean_neighborhood_dim", np.nan),
            "embeddings/pca_3d_caption": projection_caption,
        }
        for key, value in hypothesis_summary.items():
            if isinstance(value, str):
                log_dict[f"hypothesis/{key}"] = value
            elif isinstance(value, (int, float, np.integer, np.floating)):
                value_f = float(value)
                log_dict[f"hypothesis/{key}"] = value_f if math.isfinite(value_f) else np.nan

        metric_table = _maybe_wandb_metric_table(wandb_module, "hypothesis", hypothesis_summary)
        if metric_table is not None:
            log_dict["hypothesis/summary_table"] = metric_table

        if sparse_probe_result:
            sparse_summary = sparse_probe_result.get("summary", {})
            for key, value in sparse_summary.items():
                if isinstance(value, str):
                    log_dict[f"sparse_probe/{key}"] = value
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    value_f = float(value)
                    log_dict[f"sparse_probe/{key}"] = value_f if math.isfinite(value_f) else np.nan
            sparse_plot = sparse_summary.get("plot_path")
            sparse_caption = str(sparse_summary.get("interpretation") or "")
            if sparse_plot:
                try:
                    log_dict["sparse_probe/summary_plot"] = _wandb_image_with_caption(
                        wandb_module,
                        str(sparse_plot),
                        sparse_caption,
                    )
                    if sparse_caption:
                        log_dict["sparse_probe/summary_caption"] = sparse_caption
                except Exception as e:
                    print(f"[wandb] skipped sparse_probe plot: {e}")
            sparse_heatmap = sparse_summary.get("heatmap_path")
            if sparse_heatmap:
                try:
                    heatmap_caption = (
                        "Sparse complexity heatmap. Brighter patch overlays need more local dictionary atoms. "
                        + sparse_caption
                    ).strip()
                    log_dict["sparse_probe/image_heatmaps"] = _wandb_image_with_caption(
                        wandb_module,
                        str(sparse_heatmap),
                        heatmap_caption,
                    )
                    log_dict["sparse_probe/heatmap_caption"] = heatmap_caption
                except Exception as e:
                    print(f"[wandb] skipped sparse_probe heatmap: {e}")

        pca_image = _wandb_image_or_none(
            wandb_module=wandb_module, key="embeddings/pca_3d",
            build_fn=lambda: (
                (lambda fig: (
                    _wandb_image_with_caption(wandb_module, fig, projection_caption),
                    _require_matplotlib().close(fig),
                )[0])(build_embedding_scatter_figure(final_coords_3d, final_dims))
            ),
        )
        if pca_image is not None:
            log_dict["embeddings/pca_3d"] = pca_image

        if final_tsne_3d is not None:
            tsne_caption = _embedding_plot_caption(
                epoch=epoch,
                projection="t-SNE",
                fiber_summary=fiber_summary,
                hypothesis_summary=hypothesis_summary,
            )
            log_dict["embeddings/tsne_3d_caption"] = tsne_caption
            tsne_image = _wandb_image_or_none(
                wandb_module=wandb_module, key="embeddings/tsne_3d",
                build_fn=lambda: (
                    (lambda fig: (
                        _wandb_image_with_caption(wandb_module, fig, tsne_caption),
                        _require_matplotlib().close(fig),
                    )[0])(build_tsne_embedding_figure(
                        final_tsne_3d,
                        final_dims[tsne_idx] if tsne_idx is not None else final_dims,
                    ))
                ),
            )
            if tsne_image is not None:
                log_dict["embeddings/tsne_3d"] = tsne_image

        try:
            if polysemy_result and polysemy_result.get("anchors"):
                ents = [a.get("label_entropy", np.nan) for a in polysemy_result["anchors"] if a]
                if ents:
                    log_dict["polysemy/mean_label_entropy"] = float(np.nanmean(ents))
                if "singular_token_count" in polysemy_result:
                    log_dict["polysemy/singular_token_count"] = polysemy_result.get("singular_token_count")
                if "singular_token_ratio" in polysemy_result:
                    log_dict["polysemy/singular_token_ratio"] = polysemy_result.get("singular_token_ratio")
                if polysemy_result.get("entropy_triptych_source"):
                    log_dict["polysemy/entropy_triptych_source"] = polysemy_result.get("entropy_triptych_source")
                top_sets = polysemy_result.get("top_entropy_sets", [])
                if top_sets:
                    log_dict["polysemy/top_entropy_sets"] = []
                    for item in top_sets:
                        top_labels_info = item.get("top_labels", [])
                        labels_str = ", ".join(
                            [f"{t.get('name', t.get('id'))} {t.get('fraction', 0.0):.0%}" for t in top_labels_info]
                        ) if top_labels_info else "n/a"
                        poly_desc = "polysemy: diverse neighbor labels" if item.get("unique_labels", 0) > 1 else "polysemy: low"
                        caption = (
                            f"token {item.get('token_id', -1):05d} | "
                            f"H {item.get('label_entropy', 0.0):.2f} | "
                            f"uniq {item.get('unique_labels', 0)} | "
                            f"top {item.get('top_label_share', 0.0):.0%} | "
                            f"k {item.get('k', 0)} | "
                            f"labels {labels_str} | {poly_desc}"
                        )
                        log_dict["polysemy/top_entropy_sets"].append(
                            wandb_module.Image(str(item["path"]), caption=caption)
                        )
                scatter_path = polysemy_result.get("paths", {}).get("polysemy/entropy_scatter")
                if scatter_path:
                    scatter_caption = _polysemy_anchor_caption(epoch=epoch, polysemy_result=polysemy_result)
                    log_dict["polysemy/entropy_scatter"] = _wandb_image_with_caption(
                        wandb_module,
                        str(scatter_path),
                        scatter_caption,
                    )
                    log_dict["polysemy/entropy_scatter_caption"] = scatter_caption
                irreg_path_val = polysemy_result.get("paths", {}).get("polysemy/entropy_irregularity")
                irreg_stats_val = polysemy_result.get("entropy_irregularity_stats", {})
                if irreg_path_val:
                    caption = _polysemy_irregularity_caption(epoch=epoch, stats=irreg_stats_val)
                    log_dict["polysemy/entropy_irregularity"] = _wandb_image_with_caption(
                        wandb_module,
                        str(irreg_path_val),
                        caption,
                    )
                    log_dict["polysemy/entropy_irregularity_caption"] = caption
                if isinstance(irreg_stats_val, dict):
                    for k_stat in ("pearson_r", "pearson_p", "spearman_r", "spearman_p",
                                   "mean_entropy_reject", "mean_entropy_non_reject", "n_reject", "n_total", "alpha"):
                        if k_stat in irreg_stats_val:
                            log_dict[f"polysemy/entropy_irregularity/{k_stat}"] = irreg_stats_val[k_stat]
                triptych_sets = polysemy_result.get("entropy_triptych_sets", {})
                if isinstance(triptych_sets, dict):
                    for mask_mode, items in triptych_sets.items():
                        if not items:
                            continue
                        log_dict[f"polysemy/entropy_triptychs_{mask_mode}"] = [
                            wandb_module.Image(
                                str(item["path"]),
                                caption=(
                                    f"img {item.get('image_id', -1)} | "
                                    f"{item.get('label_text', 'label n/a')} | "
                                    f"token {item.get('token_id', -1)} | "
                                    f"max H {item.get('max_entropy', 0.0):.2f} | "
                                    f"{item.get('top_label_text', 'top label n/a')} | "
                                    f"pred {item.get('orig_pred_name', 'n/a')} {float(item.get('orig_pred_prob', float('nan'))):.2f} -> "
                                    f"{item.get('mask_pred_name', 'n/a')} {float(item.get('mask_pred_prob', float('nan'))):.2f} | "
                                    f"top1 drop {float(item.get('top1_drop', float('nan'))):.2f} | "
                                    f"true ({item.get('true_label_name', 'n/a')}) drop {float(item.get('true_drop', float('nan'))):.2f} | "
                                    f"mask {mask_mode}"
                                ),
                            )
                            for item in items
                        ]
                mask_effects_dict = polysemy_result.get("mask_effects", {})
                if isinstance(mask_effects_dict, dict):
                    for mask_mode, data in mask_effects_dict.items():
                        agg = data.get("aggregate", {})
                        if not isinstance(agg, dict):
                            continue
                        for k_agg in ("pred_change_rate", "mean_top1_drop", "mean_true_drop", "num_images"):
                            if k_agg in agg:
                                log_dict[f"polysemy/mask_{mask_mode}/{k_agg}"] = agg[k_agg]
            if vit_poly_metrics:
                log_dict.update({
                    k: v for k, v in vit_poly_metrics.items()
                    if isinstance(k, str) and k.startswith("vit_polysemy/")
                })

            irregular = select_irregular_tokens(
                images=images,
                image_ids=image_ids,
                labels=labels,
                fiber_results=fiber_results,
                dataset=dataset,
                bboxes=bboxes,
                neighborhood_dims=neighborhood_dims,
                class_names=class_names,
                image_mean_dims=mean_dim_by_image,
                pred_labels=pred_labels,
                top_k=12,
            )
            if irregular:
                neigh_max = max(
                    [item["neigh_dim"] for item in irregular if math.isfinite(item["neigh_dim"])],
                    default=1,
                )
                irregular_samples = _wandb_image_or_none(
                    wandb_module=wandb_module, key="embeddings/irregular_samples",
                    build_fn=lambda: [
                        wandb_module.Image(
                            add_heatmap_patch(
                                item["img"], item["bbox"], item["irregularity"],
                                neigh_value=item["neigh_dim"], neigh_max=neigh_max,
                                neighborhood_size=neighborhood_size,
                            ),
                            caption=_irregular_samples_caption(epoch=epoch, item=item),
                        )
                        for item in irregular
                    ],
                )
                if irregular_samples is not None:
                    log_dict["embeddings/irregular_samples"] = irregular_samples
            wandb_module.log(log_dict)
        except Exception as e:
            print(f"[wandb] logging failed: {e}")

    return {
        "fiber_summary": fiber_summary,
        "hypothesis_summary": hypothesis_summary,
        "hypothesis_summary_path": hypothesis_path,
        "sparse_probe_result": sparse_probe_result,
        "final_dims": final_dims,
        "final_coords_3d": final_coords_3d,
        "final_tsne_3d": final_tsne_3d,
    }


run_fiber_analysis_epoch = analyze_fiber_epoch
