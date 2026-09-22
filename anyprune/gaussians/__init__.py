"""
This package exposes functionality to operate seamlessly with Gaussians
produced by any method.
"""
from .gaussians import TILE_SIZE, Gaussians, depth_runs, view_matrices
from .importance import (
    BlendingWeights, MEASURED_SCORES, SCORES, blending_weights,
    importance_score, mini_splatting_score, radsplat_score, score_of,
)
from .pruning import SELECTIONS, Pruner
from .optimization import fine_tune, fine_tune_stages, scene_extent
from .sensitivity import Sensitivities, pup_score, sensitivity_scores, speedy_splat_score

__all__ = [
    "BlendingWeights",
    "Gaussians",
    "MEASURED_SCORES",
    "Pruner",
    "SCORES",
    "SELECTIONS",
    "Sensitivities",
    "TILE_SIZE",
    "blending_weights",
    "depth_runs",
    "importance_score",
    "mini_splatting_score",
    "pup_score",
    "radsplat_score",
    "fine_tune",
    "fine_tune_stages",
    "scene_extent",
    "sensitivity_scores",
    "speedy_splat_score",
    "score_of",
    "view_matrices",
]
