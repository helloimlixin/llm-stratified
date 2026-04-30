"""Fiber analysis package with lazy top-level re-exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "HAS_MATPLOTLIB": "fiber.plots",
    "HAS_TSNE": "fiber.plots",
    "add_heatmap_patch": "fiber.plots",
    "analyze_fiber_epoch": "fiber.analysis",
    "analyze_stratification": "fiber.geometry",
    "analyze_stratification_from_sorted_distances": "fiber.geometry",
    "build_embedding_animation_frames": "fiber.animation",
    "build_embedding_scatter_figure": "fiber.plots",
    "build_tsne_embedding_figure": "fiber.plots",
    "collect_patch_tokens": "fiber.patch_tokens",
    "compute_class_dim_means": "fiber.hypothesis",
    "compute_masked_classification_effects": "fiber.ablation",
    "compute_neighborhood_dimensions": "fiber.hypothesis",
    "compute_stratified_manifold_hypothesis_metrics": "fiber.hypothesis",
    "compute_token_polysemy_entropy_scores": "fiber.polysemy",
    "compute_token_polysemy_for_anchors": "fiber.polysemy",
    "estimate_neighborhood_dimensions": "fiber.hypothesis",
    "extract_patch_vectors": "fiber.sparse_probe",
    "extract_patch_image": "fiber.plots",
    "format_hypothesis_log_line": "fiber.hypothesis",
    "format_hypothesis_summary_line": "fiber.hypothesis",
    "generate_embedding_animation": "fiber.animation",
    "make_embedding_figure_3d": "fiber.plots",
    "make_embedding_figure_tsne": "fiber.plots",
    "make_polysemy_entropy_scatter": "fiber.plots",
    "make_polysemy_entropy_triptychs": "fiber.polysemy",
    "make_polysemy_gallery": "fiber.polysemy",
    "make_polysemy_irregularity_plot": "fiber.plots",
    "matplotlib_supports_3d": "fiber.plots",
    "normalize_volume_range": "fiber.geometry",
    "plot_progress": "fiber.plots",
    "project_embeddings_2d": "fiber.plots",
    "project_embeddings_3d": "fiber.plots",
    "project_embeddings_pca_2d": "fiber.plots",
    "project_embeddings_pca_3d": "fiber.plots",
    "project_embeddings_tsne_3d": "fiber.plots",
    "run_sparse_dictionary_probe": "fiber.sparse_probe",
    "run_fiber_analysis_epoch": "fiber.analysis",
    "run_fiber_bundle_test": "fiber.geometry",
    "run_fiber_bundle_test_from_sorted_dists": "fiber.geometry",
    "save_polysemy_entropy_scatter_plot": "fiber.plots",
    "save_polysemy_irregularity_plot": "fiber.plots",
    "save_training_summary_plot": "fiber.plots",
    "select_irregular_images": "fiber.plots",
    "select_irregular_tokens": "fiber.plots",
    "select_polysemy_entropy_images": "fiber.polysemy",
    "select_fiber_anchors": "fiber.sparse_probe",
    "select_singular_token_indices": "fiber.plots",
    "select_singular_tokens": "fiber.plots",
    "sorted_distance_matrix": "fiber.geometry",
    "summarize_class_dimensions": "fiber.hypothesis",
    "summarize_hypothesis_metrics": "fiber.hypothesis",
    "summarize_stratification": "fiber.geometry",
    "summarize_stratifications": "fiber.geometry",
    "tsne_embeddings_3d": "fiber.plots",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
