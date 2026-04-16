"""Fiber bundle analysis package.

Re-exports the public API so that ``from fiber import X`` works for all
symbols previously available from ``fiber_bundle`` or ``stratified_geometry``.
"""

from fiber.geometry import (  # noqa: F401
    normalize_volume_range,
    run_fiber_bundle_test,
    run_fiber_bundle_test_from_sorted_dists,
    sorted_distance_matrix,
    summarize_stratifications,
)
from fiber.collection import collect_patch_tokens  # noqa: F401
from fiber.hypothesis import (  # noqa: F401
    compute_class_dim_means,
    compute_neighborhood_dimensions,
    compute_stratified_manifold_hypothesis_metrics,
    format_hypothesis_log_line,
)
from fiber.visualization import (  # noqa: F401
    HAS_MATPLOTLIB,
    HAS_TSNE,
    add_heatmap_patch,
    extract_patch_image,
    make_embedding_figure_3d,
    make_embedding_figure_tsne,
    make_polysemy_entropy_scatter,
    make_polysemy_irregularity_plot,
    plot_progress,
    project_embeddings_2d,
    project_embeddings_3d,
    select_irregular_images,
    select_singular_token_indices,
    tsne_embeddings_3d,
    matplotlib_supports_3d,
)
from fiber.animation import (  # noqa: F401
    build_embedding_animation_frames,
    generate_embedding_animation,
)
from fiber.polysemy import (  # noqa: F401
    compute_token_polysemy_entropy_scores,
    compute_token_polysemy_for_anchors,
    make_polysemy_entropy_triptychs,
    make_polysemy_gallery,
    select_polysemy_entropy_images,
)
from fiber.ablation import compute_masked_classification_effects  # noqa: F401
from fiber.orchestration import run_fiber_analysis_epoch  # noqa: F401
