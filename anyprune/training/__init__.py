"""
This module exposes the features used to define a training step.
"""
from .balancing import BudgetHistogram, LossHistogram
from .learned_pruning import (
    LearnedRule, Pruned, QualityController, apply_rule, degradation_loss, gumbel_sigmoid,
    sparsity_loss,
)
from .losses import PhotometricLoss
from .reconstruction import Reconstruction, ViewSet, reconstruct
from .sampling import (
    fit_budget_fraction, plan_context_views, sample_budget_fraction,
    sample_num_context_views, sample_view_indices,
)


__all__ = [
    "BudgetHistogram",
    "LearnedRule",
    "LossHistogram",
    "PhotometricLoss",
    "Pruned",
    "QualityController",
    "apply_rule",
    "degradation_loss",
    "Reconstruction",
    "ViewSet",
    "fit_budget_fraction",
    "gumbel_sigmoid",
    "plan_context_views",
    "reconstruct",
    "sample_budget_fraction",
    "sample_num_context_views",
    "sample_view_indices",
    "sparsity_loss",
]
