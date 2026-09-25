"""
Fine-tune SplatFormer to adjust the subsampled Gaussians coming from a 
frozen feed-forward reconstructor.

A training step:
    - samples an even number of frames from one scene
    - hands half of them to the frozen reconstructor as context views
    - thins the Gaussians it predicts down to a budget, uniformly or by
      an importance score measured over those context views
    - refines the Gaussians it predicts with SplatFormer, which can
      read that score as one more input channel
    - lets a learned rule say which of the refined Gaussians survive
      (a sparsity term on their opacities, a mask head, or neither)
    - supervises with photometric loss on both context and test views
    - divides that loss by what a field of that size usually costs, so
      that a tight budget and a wide one pull equally hard

An evaluation step is performed once on the same feedforward 
reconstructor used during training and once with a different one.
The number of frames for evaluation is fixed.
"""
import gc
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import hydra
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from torch import Generator, Tensor
from tqdm import tqdm

from anyprune.datasets import DL3DVDataset, split_scenes
from anyprune.evaluation import psnr
from anyprune.gaussians import (
    Gaussians, Pruner, blending_weights, fine_tune, radsplat_score, scene_extent,
    sensitivity_scores,
)
from anyprune.models import RECONSTRUCTORS, SplatFormer, build_reconstructor
from anyprune.models.utils import (
    build_splatformer_optimizer, build_splatformer_scheduler,
)
from anyprune.training import (
    BudgetHistogram, LearnedRule, LossHistogram, PhotometricLoss, Pruned,
    QualityController, ViewSet, apply_rule, degradation_loss, fit_budget_fraction,
    RateController, lossless_loss, plan_context_views, psnr_drop, rate_term, reconstruct,
    sample_budget_fraction, sample_num_context_views, sparsity_loss,
)
from anyprune.utils import load_dotenv, out_of_memory, set_rng_seed
from anyprune.viz import RefinementBlock, plot_refinement


@dataclass
class StepResult:
    """What one training step leaves for the logging."""
    reconstruction: object
    # The views the loss was taken on, and where each half of the scene
    # sits in them when neither was thinned away (empty otherwise)
    views: ViewSet
    halves: Dict[str, slice]
    # The render the loss was taken on
    rendered: Tensor
    # The loss as the criterion reported it, and its terms
    loss: float
    terms: Dict[str, float]
    budget_fraction: float
    predicted: int
    loss_scale: float
    # What the learned rule made of the refined field, detached
    pruned: Pruned
    # What the hard cut cost against the unmasked refined render on the
    # step's views, in dB, and the weight the mask term was paid at:
    # only with a QualityController
    quality_gap: Optional[float] = None
    mask_weight: Optional[float] = None
    # How many renders the loss was backpropagated in (1: one graph),
    # and what the single graph was estimated at, in GiB
    stages: int = 1
    memory_estimate: float = 0.0


def thin(
    cfg,
    gaussians: Gaussians,
    kept: int,
    scored: ViewSet,
    pruner: Pruner,
    with_score: bool,
    generator: Optional[Generator] = None,
) -> Tuple[Gaussians, Optional[Tensor]]:
    """
    The `kept` Gaussians of a field `pruner` keeps, ranked over the
    views `scored`, and their RadSplat scores when `with_score` (None
    otherwise). A field already that small comes back whole.

    The field is measured once, over the whole prediction, before
    anything thins it: a measured rule ranks by that sweep and the score
    the refiner reads is the same sweep's, so that a Gaussian's score is
    what it carried in the whole field rather than in what was left of
    it.
    """
    if with_score and cfg.pruning.learned.score != "radsplat":
        # A learned rule reading another score than the one the pruner
        # ranks by: the field is cut by that same score when it has to
        # be, so that what the rule reads is what chose its input
        score = learned_score(cfg, gaussians, scored)
        if gaussians.num_gaussians <= kept:
            return gaussians, score
        index = torch.argsort(score, descending=True)[:kept]
        return gaussians[index], score[index]
    measured = None
    if pruner.measured or with_score:
        measured = blending_weights(
            gaussians, scored.poses, scored.intrinsics, scored.image_shape,
            batches_per_pass=cfg.pruning.batches_per_pass,
            max_intersections=cfg.device_max_intersections,
        )
    score = radsplat_score(measured) if with_score else None
    if gaussians.num_gaussians <= kept:
        return gaussians, score
    index = pruner.order(
        gaussians, scored.poses, scored.intrinsics, scored.image_shape,
        generator=generator, measured=measured,
    )[:kept]
    return gaussians[index], None if score is None else score[index]


def learned_score(cfg, gaussians: Gaussians, scored: ViewSet) -> Tensor:
    """
    The score a learned rule reads (pruning.learned.score), measured
    over the views 'scored': the RadSplat peak blending weight, or
    Speedy-Splat's sensitivity (anyprune.gaussians.sensitivity), which
    costs about twice the sweep.
    """
    which = cfg.pruning.learned.score
    assert which in ("radsplat", "speedy-splat"), (
        f"pruning.learned.score has to be 'radsplat' or 'speedy-splat', got {which!r}"
    )
    if which == "radsplat":
        return radsplat_score(blending_weights(
            gaussians, scored.poses, scored.intrinsics, scored.image_shape,
            batches_per_pass=cfg.pruning.batches_per_pass,
            max_intersections=cfg.device_max_intersections,
        ))
    return sensitivity_scores(
        gaussians, scored.poses, scored.intrinsics, scored.image_shape, fisher=False,
        batches_per_pass=cfg.pruning.batches_per_pass,
        max_intersections=cfg.device_max_intersections,
    ).speedy


def distillation_target(
    cfg, splatformer, rule: LearnedRule, gaussians: Gaussians, score: Optional[Tensor],
    context: ViewSet, generator: Generator,
) -> Optional[Tuple[Tensor, Gaussians, float]]:
    """
    What per-scene optimization would make of the field the refiner was
    handed (optim.distillation): the mask is read off a forward with no
    gradient, the *reconstructor's* Gaussians are cut to what it keeps,
    and those are optimized for distillation.steps steps on the context
    views, the way the benchmark post-optimizes a pruned field. Comes
    back as the (N,) keep mask, the optimized survivors, and the
    scene's extent the means are measured against; None when the rule
    keeps nothing.

    The target is the optimization of the *input*, not of the refiner's
    own output: chasing the optimization of its own prediction is a
    moving target, and the run of 2026-09-22 drifted off it (opacity
    and scale x126, 16 dB refined against 26 for the same recipe
    without the term).
    """
    with torch.no_grad():
        refined, logits = refine(cfg, splatformer, gaussians, score)
        pruned = apply_rule(rule, refined, logits, noisy=False)
        keep = pruned.hard
        if int(keep.sum().item()) == 0:
            return None
        survivors = gaussians[keep]
        survivors = replace(survivors, **{
            name: getattr(survivors, name).float()
            for name in ("means", "covariances", "harmonics", "opacities", "scales", "rotations")
        })
        del refined, logits, pruned
    seed = int(torch.randint(2 ** 31, (1,), generator=generator).item())
    optimized = fine_tune(
        survivors, context.poses, context.intrinsics, context.images,
        cfg.optim.distillation.steps, generator=Generator().manual_seed(seed), factorize=False,
    )
    return keep, optimized, scene_extent(survivors, context.poses)


def distillation_loss(
    cfg, refined: Gaussians, target: Tuple[Tensor, Gaussians, float]
) -> Tuple[Tensor, Dict[str, float]]:
    """
    How far the refiner's output sits from where optimization would
    have taken it, per surviving Gaussian: the mean L1 of the means (per
    unit of scene extent), of the log scales, of the opacity logits and
    of the harmonics, and one minus the cosine between the rotations,
    each at the weight optim.distillation gives it, times the whole
    term's weight. The target does not carry gradient.
    """
    keep, optimized, extent = target
    weights = cfg.optim.distillation
    means = (refined.means[keep].float() - optimized.means).abs().mean() / extent
    scales = (refined.scales[keep].float().clamp_min(1e-8).log() - optimized.scales.log()).abs().mean()
    q = torch.nn.functional.normalize(refined.rotations[keep].float(), dim=-1)
    rotations = (1.0 - (q * optimized.rotations).sum(dim=-1).abs()).mean()
    opacities = (
        torch.logit(refined.opacities[keep].float().clamp(1e-4, 1 - 1e-4))
        - torch.logit(optimized.opacities.clamp(1e-4, 1 - 1e-4))
    ).abs().mean()
    harmonics = (refined.harmonics[keep].float() - optimized.harmonics).abs().mean()
    terms = {
        "distill_means": means, "distill_scales": scales, "distill_rotations": rotations,
        "distill_opacities": opacities, "distill_harmonics": harmonics,
    }
    loss = weights.weight * (
        weights.means * means + weights.scales * scales + weights.rotations * rotations
        + weights.opacities * opacities + weights.harmonics * harmonics
    )
    logged = {name: value.item() for name, value in terms.items()}
    logged["distillation"] = loss.item()
    return loss, logged



def supervision_views(cfg, reconstruction, generator: Generator) -> Tuple[ViewSet, Dict[str, slice]]:
    """
    The views a step takes its loss on, and where the context views and
    the held-out ones sit in them: known when both halves are in and
    nothing had to be thinned, so that each half can be scored apart,
    and empty otherwise.
    """
    views = reconstruction.views(cfg.supervision.views)
    limit = cfg.supervision.max_views
    halves = {}
    if cfg.supervision.views == "all" and (limit is None or len(views) <= limit):
        split = len(reconstruction.context)
        halves = {"context": slice(0, split), "test": slice(split, len(views))}
    if limit is None:
        return views, halves
    return views.thin(limit, generator=generator), halves


# What the loss is rendered from, and what a staged backward hands the
# refiner's graph the gradient of
RENDER_FIELDS = ("means", "covariances", "harmonics", "opacities")


def estimate_single_graph_gib(cfg, num_gaussians: int, num_views: int, image_shape) -> float:
    """
    What a step that renders every supervised view in one graph is
    expected to peak at, in GiB: what is resident now, plus the
    refiner's graph and backward per million Gaussians, plus the
    render's per million Gaussians per view, the latter scaled by the
    pixels of a view against the resolution the constant was measured
    at (see device_memory_* in the machine config).
    """
    resident = torch.cuda.memory_allocated() / 2 ** 30
    pixels = image_shape[0] * image_shape[1] / cfg.device_memory_view_reference_pixels
    per_million = (
        cfg.device_memory_per_million_gaussians_gib
        + num_views * pixels * cfg.device_memory_per_million_gaussians_per_view_gib
    )
    return resident + per_million * num_gaussians / 1e6


def single_graph_fits(cfg, num_gaussians: int, num_views: int, image_shape) -> Tuple[bool, float, float]:
    """
    Whether the single graph is expected to fit under the card's memory
    with the configured margin to spare, with the estimate and what the
    card holds, both in GiB.
    """
    estimate = estimate_single_graph_gib(cfg, num_gaussians, num_views, image_shape)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    return estimate + cfg.device_memory_safety_margin_gib <= total, estimate, total


def reference_render(cfg, rule: LearnedRule, handed: Gaussians, refined: Gaussians, views: ViewSet) -> Optional[Tensor]:
    """
    The render the degradation term measures the rule's against, on
    these views, out of the graph: the field the refiner was handed or
    the refined field left whole (see LearnedRule). None with the term
    off.
    """
    if not (rule.by_degradation or rule.by_lossless):
        return None
    reference = (
        rule.lossless_reference if rule.by_lossless else rule.degradation_reference
    )
    field = refined if reference == "refined" else handed
    with torch.no_grad():
        rendered, _ = field.rasterize(
            views.poses, views.intrinsics, views.image_shape,
            views_per_pass=cfg.device_max_views_per_render,
        )
    return rendered


def chunks(views: ViewSet, size: int):
    """The views in groups of at most 'size', in order."""
    for first in range(0, len(views), size):
        yield ViewSet(
            views.images[first:first + size], views.poses[first:first + size],
            views.intrinsics[first:first + size],
        )


def training_step(
    cfg,
    splatformer: SplatFormer,
    reconstructor,
    criterion: PhotometricLoss,
    optimizer,
    scaler,
    scene: Dict[str, Tensor],
    context_idx: Tensor,
    test_idx: Tensor,
    budget_fraction: float,
    generator: Generator,
    pruner: Pruner,
    rule: LearnedRule,
    histogram: Optional[BudgetHistogram] = None,
    loss_histogram: Optional[LossHistogram] = None,
    budget_scale: float = 1.0,
    rates: Optional[RateController] = None,
    controller: Optional[QualityController] = None,
) -> StepResult:
    """
    Reconstruct, refine, score and take one optimizer step, returning
    what the logging needs. Raises on a card that ran out of memory.

    How much of the prediction is kept comes from `histogram` when there
    is one and from `budget_fraction` when there is not, scaled either
    way by `budget_scale`; which of the prediction is kept is `pruner`'s
    call, read over at most pruning.max_scoring_views of the context
    views. Which of the refined Gaussians then survive is `rule`'s call,
    paid for by its sparsity terms on top of the photometric loss.

    When there is a `loss_histogram`, what the step backpropagates is
    its loss over what a field of its size has been costing the run,
    while what it reports is the loss itself. The two histograms are
    independent: either one can be there without the other.
    """
    # Back on the card if the last step left it off (see below): no
    # move when it is already there
    reconstructor.to(scene["images"].device)
    reconstruction = reconstruct(
        reconstructor, scene, context_idx, test_idx,
        generator=generator, context_downscale=cfg.context_downscale,
    )
    # Thinned here rather than inside reconstruct(), which takes a count
    # and both ways of arriving at one need the prediction that only
    # exists once the reconstructor has run. It costs no memory the step
    # was not already spending: the whole field is predicted before
    # anything thins it.
    predicted = reconstruction.gaussians.num_gaussians
    if histogram is not None:
        # A size drawn from wherever the run is short of examples, and
        # the share is then whatever that size works out to: reported
        # rather than asked for.
        #
        # A retry lowers the ceiling rather than the size that came back
        # from it. Scaling the size would land the step in whichever
        # bucket the multiplication happened to reach, which is the one
        # thing this is here not to do; lowering the ceiling instead
        # re-asks the same question of the buckets that still fit.
        kept = histogram.choose(
            predicted,
            max(round(cfg.device_max_gaussians * budget_scale), 1),
            generator=generator,
        )
    else:
        # The share is carried onto what the card can hold rather than
        # the count being clipped at it, so that the step is handed a
        # share it can hold and reports the share it was handed.
        kept = round(
            fit_budget_fraction(
                budget_fraction * budget_scale, cfg.budget.min_fraction,
                cfg.budget.max_fraction, predicted, cfg.device_max_gaussians,
            ) * predicted
        )
    kept = max(kept, 1)
    budget_fraction = kept / predicted
    reconstruction.gaussians, score = thin(
        cfg, reconstruction.gaussians, kept, scoring_views(cfg, reconstruction.context),
        pruner, splatformer.reads_score, generator=generator,
    )
    if cfg.compensation.enabled:
        # Against the share the field actually came out at rather than
        # the one that was asked for: a step that kept more than the
        # reconstructor predicted kept all of it, and has nothing to put
        # back
        reconstruction.gaussians = reconstruction.gaussians.compensate(
            reconstruction.gaussians.num_gaussians / predicted,
            exponent=cfg.compensation.exponent,
        )
    views, halves = supervision_views(cfg, reconstruction, generator)

    # The reconstructor has done its part, and on a field wide enough
    # its weights are room the backward needs (see
    # device_offload_reconstructor_above): moved off the card for the
    # rest of the step. Not moved back here on a failure: a step that
    # ran out of memory still holds its graph while its frames are
    # live, and the move back would run out too, so the next step, or
    # run_validation(), puts it back once the memory is free.
    offload = (
        reconstruction.gaussians.num_gaussians >= cfg.device_offload_reconstructor_above
    )
    if offload:
        reconstructor.to("cpu")
    # Whether every supervised view can be rendered in one graph: the
    # refiner's graph is most of a step (10 GiB at 800k Gaussians) and
    # the render adds about 0.14 GiB per 448x448 view on top of it, so
    # on a wide field with many views the loss is instead rendered and
    # backpropagated a few views at a time into leaves standing in for
    # the refined field, and the refiner's graph backpropagated once
    # from the gradient they accumulate. The same gradient, by the
    # chain rule, at the memory of one render.
    fits, memory_estimate, total_memory = single_graph_fits(
        cfg, reconstruction.gaussians.num_gaussians, len(views), views.image_shape
    )
    stage_size = len(views) if fits else cfg.device_max_views_per_render
    stages = math.ceil(len(views) / stage_size)

    # With a controller, the mask term is paid at the weight the steps
    # before this one at this view count have set
    quality_gap = mask_weight = None
    if controller is not None:
        mask_weight = controller.weight(len(context_idx))
    # Read before this step is recorded, out in the loop, so that a
    # step is weighed against the sizes the run has already seen
    # and never against itself
    loss_scale = 1.0 if loss_histogram is None else loss_histogram.scale_of(
        reconstruction.gaussians.num_gaussians
    )

    # The distillation target first, off its own forward, so that its
    # optimization does not sit on top of the refiner's graph
    target = None
    if cfg.optim.distillation.steps > 0:
        target = distillation_target(
            cfg, splatformer, rule, reconstruction.gaussians, score, reconstruction.context, generator,
        )

    refined, logits = refine(cfg, splatformer, reconstruction.gaussians, score)
    if stages == 1:
        pruned = apply_rule(rule, refined, logits, noisy=True)
        rendered, _ = pruned.trained.rasterize(
            views.poses, views.intrinsics, views.image_shape,
            views_per_pass=cfg.device_max_views_per_render,
        )
        reference = reference_render(cfg, rule, reconstruction.gaussians, refined, views)
        if rule.by_lossless and rates is not None:
            # The constraint on a multiplier: the photometric loss with
            # the rate weighed at what the steps before this one at
            # this view count have settled on
            loss, terms = criterion(rendered, views.images)
            key = 0 if rule.rate_controller_shared else len(context_idx)
            mask_weight = rates.weight(key)
            rate = rate_term(rule, pruned.soft["mask"], refined.num_gaussians)
            loss = loss + mask_weight * rate
            quality_gap = psnr_drop(rendered, reference, views.images)
            terms.update({
                "lossless_rate": rate.item(), "lossless_drop_db": quality_gap,
                "rate_weight": rates.update(key, quality_gap),
            })
            del reference
        elif rule.by_lossless:
            # The rate against a quality budget: the photometric loss
            # is not what is minimized here, and is only added when
            # asked for (see anyprune.training.learned_pruning)
            loss, terms = lossless_loss(
                rule, rendered, reference, pruned.soft["mask"], views.images,
            )
            if cfg.pruning.learned.lossless.photometric_weight > 0.0:
                photometric, photometric_terms = criterion(rendered, views.images)
                loss = loss + cfg.pruning.learned.lossless.photometric_weight * photometric
                terms.update(photometric_terms)
            del reference
        else:
            loss, terms = criterion(rendered, views.images)
            if reference is not None:
                degradation = degradation_loss(rule, rendered, reference, views.images)
                loss = loss + degradation
                terms["degradation"] = degradation.item()
            del reference
            # One scalar and one backward for both: with non-reentrant
            # gradient checkpointing the graph is freed by the first
            sparsity, sparsity_terms = sparsity_loss(rule, pruned, mask_weight=mask_weight)
            loss = loss + sparsity
            terms.update(sparsity_terms)
        if target is not None:
            distill, distill_terms = distillation_loss(cfg, refined, target)
            loss = loss + distill
            terms.update(distill_terms)
        # The terms and the number that goes into the log are the ones
        # the criterion came back with, which are what a run that does
        # not normalize logs too and are the only ones comparable
        # across sizes
        reported = loss.item()
        if loss_scale != 1.0:
            loss = loss / loss_scale
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
    else:
        # Leaves in place of what the render reads of the refined
        # field, and of the mask logits: the rule is applied on them,
        # so that its ops are the one small graph the stages share
        leaves = {
            name: getattr(refined, name).detach().requires_grad_(True)
            for name in RENDER_FIELDS
        }
        logits_leaf = None if logits is None else logits.detach().requires_grad_(True)
        pruned = apply_rule(rule, replace(refined, **leaves), logits_leaf, noisy=True)
        optimizer.zero_grad(set_to_none=True)
        rendered_stages, terms, reported = [], {}, 0.0
        for stage in chunks(views, stage_size):
            # The mean over every view is the sum of each stage's mean
            # weighed by its share of the views
            share = len(stage) / len(views)
            stage_render, _ = pruned.trained.rasterize(
                stage.poses, stage.intrinsics, stage.image_shape, views_per_pass=len(stage),
            )
            reference = reference_render(cfg, rule, reconstruction.gaussians, refined, stage)
            if rule.by_lossless and rates is not None:
                stage_loss, stage_terms = criterion(stage_render, stage.images)
                rate = rate_term(rule, pruned.soft["mask"], refined.num_gaussians)
                stage_loss = stage_loss + rates.weight(
                    0 if rule.rate_controller_shared else len(context_idx)
                ) * rate
                stage_terms.update({
                    "lossless_rate": rate.item(),
                    "lossless_drop_db": psnr_drop(stage_render, reference, stage.images),
                })
                del reference
            elif rule.by_lossless:
                # Per stage, the whole rate against this stage's share
                # of the budget: the max cannot be taken over views the
                # backward has not reached yet, and a stage stands in
                # for the window the way its photometric loss does
                stage_loss, stage_terms = lossless_loss(
                    rule, stage_render, reference, pruned.soft["mask"], stage.images,
                )
                if cfg.pruning.learned.lossless.photometric_weight > 0.0:
                    photometric, photometric_terms = criterion(stage_render, stage.images)
                    stage_loss = stage_loss + cfg.pruning.learned.lossless.photometric_weight * photometric
                    stage_terms.update(photometric_terms)
                del reference
            else:
                stage_loss, stage_terms = criterion(stage_render, stage.images)
                if reference is not None:
                    degradation = degradation_loss(rule, stage_render, reference, stage.images)
                    stage_loss = stage_loss + degradation
                    stage_terms["degradation"] = degradation.item()
                    del reference
            # The shared graph above the leaves is kept for the next
            # stage; the render's own is freed with its outputs
            scaler.scale(stage_loss * share / loss_scale).backward(retain_graph=True)
            reported += stage_loss.item() * share
            for name, value in stage_terms.items():
                terms[name] = terms.get(name, 0.0) + value * share
            rendered_stages.append(stage_render.detach())
            del stage_render, stage_loss
        if rule.by_lossless and rates is not None:
            terms["rate_weight"] = rates.update(
                0 if rule.rate_controller_shared else len(context_idx), terms["lossless_drop_db"]
            )
        if not rule.by_lossless:
            sparsity, sparsity_terms = sparsity_loss(rule, pruned, mask_weight=mask_weight)
            scaler.scale(sparsity / loss_scale).backward()
            reported += sparsity.item()
            terms.update(sparsity_terms)
        rendered = torch.cat(rendered_stages, dim=0)
        del rendered_stages
        # The distillation term reads the refined field itself, so it
        # goes into the one backward through the refiner's graph
        extra_outputs, extra_grads = [], []
        if target is not None:
            distill, distill_terms = distillation_loss(cfg, refined, target)
            extra_outputs.append(scaler.scale(distill / loss_scale))
            extra_grads.append(torch.ones_like(extra_outputs[-1]))
            reported += distill.item()
            terms.update(distill_terms)
        # The refiner's graph, once, from what the stages accumulated
        torch.autograd.backward(
            [getattr(refined, name) for name in RENDER_FIELDS]
            + ([] if logits is None else [logits]) + extra_outputs,
            [leaves[name].grad for name in RENDER_FIELDS]
            + ([] if logits is None else [logits_leaf.grad]) + extra_grads,
        )
    if cfg.optim.grad_clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            splatformer.parameters(), cfg.optim.grad_clip_norm
        )
    scaler.step(optimizer)
    scaler.update()
    # What the cut cost on these views against the field the refiner
    # was handed, for the controller: read here, with the graph freed
    # by the backward, since the whole render of a dense field is a
    # peak the step cannot afford on top of its activations. Not
    # against the refined field left whole: rendered hard, the refiner
    # is free to write anything into the Gaussians it cuts, and did,
    # so that the whole refined render sat 1-9 dB *under* the cut and
    # the controller never moved (2026-09-20).
    if controller is not None:
        with torch.no_grad():
            whole, _ = reconstruction.gaussians.rasterize(
                views.poses, views.intrinsics, views.image_shape,
                views_per_pass=cfg.device_max_views_per_render,
            )
            quality_gap = (
                psnr(whole.float(), views.images) - psnr(rendered.detach().float(), views.images)
            ).mean().item()
            del whole
        controller.update(len(context_idx), quality_gap)
    if offload:
        reconstructor.to(reconstruction.gaussians.device)
    return StepResult(
        reconstruction, views, halves, rendered.detach(), reported, terms,
        budget_fraction, predicted, loss_scale, pruned.detach(),
        quality_gap, mask_weight, stages, memory_estimate,
    )


def scoring_views(cfg, context):
    """
    The context views a score is measured over: all of them, or at most
    pruning.max_scoring_views drawn evenly along the capture, which
    bounds what a step spends ranking a dense prediction.
    """
    limit = cfg.pruning.max_scoring_views
    if limit is None or len(context) <= limit:
        return context
    return context[torch.linspace(0, len(context) - 1, limit).round().long()]


def draw_pruner(pruners: Sequence[Pruner], generator: Generator) -> Pruner:
    """The rule a step thins with, one of the configured ones at random."""
    if len(pruners) == 1:
        return pruners[0]
    return pruners[torch.randint(len(pruners), (1,), generator=generator).item()]


def recover_from_oom(optimizer, scaler):
    """
    Put the optimizer and the loss scaler back into a state a fresh
    attempt can start from, then collect and hand the memory back.

    The scaler is closed out with update(), which refuses when there is
    no iteration open; that refusal is ignored.
    """
    optimizer.zero_grad(set_to_none=True)
    try:
        scaler.update()
    except (AssertionError, RuntimeError):
        pass
    gc.collect()
    torch.cuda.empty_cache()


def plan_views(
    cfg,
    dataset: DL3DVDataset,
    scene_idx: int,
    num_context_views: int,
    generator: Optional[Generator] = None,
):
    """
    Sample a run of views from one scene, returning the frames to read
    and the two halves they split into, without reading anything.

    A view count the scene is too short for is lowered to what the
    scene does hold at this stride.
    """
    return plan_context_views(
        dataset.num_frames(scene_idx), num_context_views,
        stride=cfg.view_stride, generator=generator,
    )


def read_scene(dataset: DL3DVDataset, scene_idx: int, frames: Tensor):
    """Read the named frames of a scene off disk, leaving them on the host."""
    return dataset.get_frames(scene_idx, frames)


def load_scene(
    cfg,
    dataset: DL3DVDataset,
    scene_idx: int,
    device: torch.device,
    num_context_views: int,
    generator: Optional[Generator] = None,
):
    """
    Sample a run of views from one scene and read just those frames,
    returning the scene alongside the two halves it splits into.
    """
    frames, context_idx, test_idx = plan_views(
        cfg, dataset, scene_idx, num_context_views, generator
    )
    scene = read_scene(dataset, scene_idx, frames)
    scene = {name: value.to(device) for name, value in scene.items()}
    return scene, context_idx, test_idx


@dataclass
class StepPlan:
    """
    Everything one training step draws, before any of it is read or
    reconstructed. The budget is a share of the prediction rather than a
    count.
    """
    scene_idx: int
    frames: Tensor
    context_idx: Tensor
    test_idx: Tensor
    budget_fraction: float


def draw_step(
    cfg,
    dataset: DL3DVDataset,
    scenes: Sequence[int],
    generator: Generator,
) -> StepPlan:
    """
    Draw the scene a step trains on, the views it takes off it and the
    share of the prediction it keeps.
    """
    scene_idx = scenes[torch.randint(len(scenes), (1,), generator=generator).item()]
    num_context_views = sample_num_context_views(
        cfg.context_views, generator=generator
    )
    frames, context_idx, test_idx = plan_views(
        cfg, dataset, scene_idx, num_context_views, generator
    )
    return StepPlan(
        scene_idx=scene_idx,
        frames=frames,
        context_idx=context_idx,
        test_idx=test_idx,
        budget_fraction=sample_budget_fraction(
            cfg.budget.min_fraction, cfg.budget.max_fraction, generator=generator
        ),
    )


class ScenePrefetcher:
    """
    Read the frames of the next step on one worker thread while the GPU
    is still working on this one. Only the reading moves: every draw
    stays on the main thread, in the order draw_step() makes them.
    """

    def __init__(self, dataset: DL3DVDataset, device: torch.device):
        self.dataset = dataset
        self.device = device
        self.workers = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def submit(self, plan: StepPlan):
        """Start reading the frames a plan names."""
        assert self.pending is None, "A read is already in flight"
        self.pending = (
            plan,
            self.workers.submit(read_scene, self.dataset, plan.scene_idx, plan.frames),
        )

    def take(self):
        """
        Wait for the read in flight and move it onto the GPU, returning
        it with the plan it belongs to.
        """
        assert self.pending is not None, "Nothing was submitted to read"
        plan, reading = self.pending
        # The read itself is on the host and cannot fail for want of
        # room; the copy onto the card can, and a card that is full at
        # this moment is one a collection may yet empty. So the read is
        # only forgotten once its copy has landed, which leaves this
        # safe to call again after a failure rather than losing the step
        # the caller was about to take.
        scene = reading.result()
        scene = {name: value.to(self.device) for name, value in scene.items()}
        self.pending = None
        return plan, scene

    def close(self):
        self.workers.shutdown()


def refine(
    cfg,
    splatformer: SplatFormer,
    gaussians: Gaussians,
    score: Optional[Tensor] = None,
    enable_amp: Optional[bool] = None,
) -> Tuple[Gaussians, Optional[Tensor]]:
    """
    Run SplatFormer over a set of Gaussians, in half precision if asked,
    returning the refined set and the mask logits when it has a mask
    head.
    """
    with torch.cuda.amp.autocast(
        enabled=cfg.optim.enable_amp if enable_amp is None else enable_amp
    ):
        return splatformer.refine(gaussians, score)


def control_at(cfg, gaussians: Gaussians, score: Tensor, count: int) -> Gaussians:
    """
    What a learned rule is measured against at the count it chose: the
    top of the RadSplat score, with the optical depth of the thinning
    put back. Without it a rule that simply keeps more would look better
    for free.
    """
    control = gaussians
    if gaussians.num_gaussians > count:
        control = gaussians[torch.topk(score, count, sorted=False).indices]
    return control.compensate(
        control.num_gaussians / gaussians.num_gaussians,
        exponent=cfg.compensation.exponent,
    )


def refinement_figure(
    cfg,
    reconstruction,
    thinned,
    refined,
    test_renders: Dict[str, Tensor],
    full_test: Optional[Tensor],
    scene_idx: int,
    budget: int,
    num_context_views: int,
) -> Figure:
    """
    The figure a validation pass logs: what one scene's thinned
    Gaussians and the refined ones make of the views the reconstructor
    saw and of the ones held out from it, against the whole prediction
    they were thinned out of and the ground truth of both.

    The whole prediction goes in with its own PSNR and its own count,
    and is left out when the budget was wider than the prediction or its
    render of the held-out views did not fit. The held-out renders of
    the thinned and refined Gaussians, and of the whole prediction, are
    the ones the pass already took for its metrics, handed in rather
    than drawn again; the context views are rendered here. Everything is
    moved to the host as it is built.
    """
    context, test = reconstruction.context, reconstruction.test
    predicted = reconstruction.gaussians
    thinning = full_test is not None and (
        predicted.num_gaussians > thinned.num_gaussians
        or refined.num_gaussians < thinned.num_gaussians
    )

    def render(gaussians, views) -> Tensor:
        return gaussians.rasterize(
            views.poses, views.intrinsics, views.image_shape,
            views_per_pass=cfg.device_max_views_per_render,
        )[0].cpu()

    context_renders = {
        name: render(gaussians, context)
        for name, gaussians in (("input", thinned), ("refined", refined))
    }
    full_context = render(predicted, context) if thinning else None
    blocks = (
        RefinementBlock(
            name="self-reconstruction, on the views the reconstructor saw",
            tag="ctx",
            truth=context.images.cpu(),
            downsampled=context_renders["input"],
            refined=context_renders["refined"],
            full=full_context,
        ),
        RefinementBlock(
            name="novel view synthesis, on the views held out from it",
            tag="test",
            truth=test.images.cpu(),
            downsampled=test_renders["psnr_input"],
            refined=test_renders["psnr_refined"],
            full=full_test if thinning else None,
        ),
    )
    return plot_refinement(
        blocks,
        # What is delivered: what was handed over, less what the learned
        # rule dropped
        num_gaussians=refined.num_gaussians,
        # What the reconstructor was handed, not what the figure draws:
        # a pass renders at most validation.scored_views of each half,
        # and the count that explains the size of the field is the one
        # the reconstruction was made from
        num_context_views=num_context_views,
        title=f"DL3DV scene {scene_idx} at a {budget // 1000}k Gaussian budget",
        num_shown=cfg.validation.image_views,
        num_input_gaussians=predicted.num_gaussians if thinning else None,
    )


@torch.no_grad()
def validate(
    cfg,
    splatformer: SplatFormer,
    reconstructor,
    dataset: DL3DVDataset,
    scene_indices: Sequence[int],
    num_context_views: int,
    pruner: Pruner,
    rule: LearnedRule,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Figure]]:
    """
    Score every stage a field goes through on the held-out views of
    every validation scene, at each of the fixed budgets in
    validation.gaussian_budgets, and return their mean PSNRs and the
    gains between them, so that a pass says where the PSNR at a budget
    comes from:
        psnr_full      the whole prediction, before any thinning
        psnr_thinned   the budget's subsample of it, as the rule left it
        psnr_input     the same after opacity compensation, which is what
                       the network is handed (the thinned field itself
                       when compensation is off)
        psnr_refined   what the network made of it, less what the
                       learned rule dropped
    with psnr_compensation_gain what compensation put back over the raw
    subsample, psnr_gain what refinement added over its input, and
    psnr_vs_full what the refined field is worth against the whole
    prediction it was thinned out of.

    With a learned rule on, a pass also says what the rule kept
    (kept_fraction of the prediction, rule_fraction of what it was
    handed, the mean soft keep of each term as soft_<term>) and how the
    refined scales moved (scale_ratio, refined over input, which is
    where an opacity term goes to be paid when it is not paid in
    Gaussians); and it measures psnr_control, the compensated RadSplat
    top-k at exactly the count the rule kept, with psnr_vs_control the
    rule's gap to it. That control is what makes the rule's number mean
    something: a rule that simply keeps more would look better for free.

    A scene is reconstructed once, ranked once by `pruner` over its
    context views, and then thinned down to each budget in turn as a
    prefix of that order. Both the views and the thinning come off a
    generator seeded by the scene alone.

    A scene that does not fit is dropped instead of ending the run. A
    budget's means are then taken over the scenes that did fit, and a
    budget no scene fit at is left out of the metrics entirely.

    Alongside the metrics come the figures of the first
    validation.num_image_scenes scenes at each of the
    validation.image_budgets budgets, keyed by the two, leaving out any
    of those pairings that did not fit.
    """
    was_training = splatformer.training
    splatformer.eval()
    device = next(splatformer.parameters()).device
    scores = {
        budget: {
            "psnr_thinned": [], "psnr_input": [], "psnr_refined": [],
            "psnr_full": [], "psnr_vs_full": [],
            "psnr_control": [], "psnr_vs_control": [],
            "kept_fraction": [], "rule_fraction": [], "scale_ratio": [],
            "soft_opacity": [], "soft_mask": [],
        }
        for budget in cfg.validation.gaussian_budgets
    }
    # The score is read by the refiner, or by the control the rule is
    # measured against
    with_score = splatformer.reads_score or rule.enabled
    figures: Dict[Tuple[int, int], Figure] = {}
    figure_scenes = set(scene_indices[:cfg.validation.num_image_scenes])

    for scene_idx in tqdm(
        scene_indices, desc="Validating", unit="scene", leave=False
    ):
        generator = Generator().manual_seed(cfg.split_seed + scene_idx)
        scene, context_idx, test_idx = load_scene(
            cfg, dataset, scene_idx, device, num_context_views, generator
        )
        try:
            reconstruction = reconstruct(
                reconstructor, scene, context_idx, test_idx,
                generator=generator, context_downscale=cfg.context_downscale,
            )
        except RuntimeError as error:
            if not out_of_memory(error):
                raise
            del scene
            torch.cuda.empty_cache()
            tqdm.write(f"  scene {scene_idx} did not fit, skipped")
            continue
        # Off their own generators, seeded by the scene alone, so that
        # the views a scene is scored on are the same at every pass of
        # the run and the same for both reconstructors
        reconstruction.test = reconstruction.test.thin(
            cfg.validation.scored_views,
            generator=Generator().manual_seed(cfg.split_seed + scene_idx),
        )
        reconstruction.context = reconstruction.context.thin(
            cfg.validation.scored_views,
            generator=Generator().manual_seed(cfg.split_seed + scene_idx),
        )
        views = reconstruction.test
        try:
            scored = scoring_views(cfg, reconstruction.context)
            if with_score and cfg.pruning.learned.score != "radsplat":
                score = learned_score(cfg, reconstruction.gaussians, scored)
                order = torch.argsort(score, descending=True)
            else:
                measured = None
                if pruner.measured or with_score:
                    measured = blending_weights(
                        reconstruction.gaussians,
                        scored.poses, scored.intrinsics, scored.image_shape,
                        batches_per_pass=cfg.pruning.batches_per_pass,
                        max_intersections=cfg.device_max_intersections,
                    )
                order = pruner.order(
                    reconstruction.gaussians,
                    scored.poses, scored.intrinsics, scored.image_shape,
                    generator=Generator().manual_seed(cfg.split_seed + scene_idx),
                    measured=measured,
                )
                score = radsplat_score(measured) if with_score else None
                del measured
        except RuntimeError as error:
            if not out_of_memory(error):
                raise
            error.__traceback__ = None
            del reconstruction
            torch.cuda.empty_cache()
            tqdm.write(f"  scene {scene_idx} did not fit being ranked, skipped")
            continue
        # The whole prediction, rendered once per scene rather than once
        # per budget: it is the ceiling every budget below is spending
        # against, and it is the same render whichever of them is being
        # measured. It is also the widest render of the pass, wider than
        # any budget by construction, so it gets a guard of its own: a
        # scene whose whole field does not fit still has its input and
        # refined numbers, it just has nothing to be put against.
        drawing_scene = scene_idx in figure_scenes
        full_test = full_render = psnr_full = None
        try:
            full_render, _ = reconstruction.gaussians.rasterize(
                views.poses, views.intrinsics, views.image_shape,
                views_per_pass=cfg.device_max_views_per_render,
            )
            psnr_full = psnr(full_render, views.images).mean().item()
            if drawing_scene:
                full_test = full_render.cpu()
        except RuntimeError as error:
            if not out_of_memory(error):
                raise
            tqdm.write(
                f"  scene {scene_idx} did not fit rendered whole, "
                f"measured without a ceiling"
            )
        finally:
            del full_render
            torch.cuda.empty_cache()

        for budget in cfg.validation.gaussian_budgets:
            raw = thinned = refined = rendered = control = None
            try:
                raw = reconstruction.gaussians[order[:budget]]
                kept_score = None if score is None else score[order[:budget]]
                thinned = raw
                if cfg.compensation.enabled:
                    # Measured as the input too, so that psnr_gain stays
                    # what the network adds over what it was handed,
                    # and the raw subsample is kept to be measured on
                    # its own, so that the pass also says what
                    # compensation put back
                    thinned = raw.compensate(
                        raw.num_gaussians
                        / reconstruction.gaussians.num_gaussians,
                        exponent=cfg.compensation.exponent,
                    )
                # In single precision whatever training runs in: spconv
                # takes a different path once a module leaves training
                # mode, and it has no half-precision kernel to offer there
                refined, logits = refine(
                    cfg, splatformer, thinned, kept_score, enable_amp=False
                )
                # What inference delivers: the plain sigmoid's level set,
                # with no Gumbel noise
                pruned = apply_rule(rule, refined, logits, noisy=False)
                refined = pruned.kept
                # Held back until every render is in, so that a budget
                # that overflows halfway through does not leave the
                # PSNRs averaged over different sets of scenes, which
                # would put a gain between them that no scene measured
                measured, kept = {}, {}
                drawing = drawing_scene and budget in cfg.validation.image_budgets
                stages = [("psnr_input", thinned), ("psnr_refined", refined)]
                if thinned is not raw:
                    stages.insert(0, ("psnr_thinned", raw))
                if rule.enabled:
                    control = control_at(
                        cfg, reconstruction.gaussians, score, refined.num_gaussians
                    )
                    stages.append(("psnr_control", control))
                for name, gaussians in stages:
                    rendered, _ = gaussians.rasterize(
                        views.poses, views.intrinsics, views.image_shape,
                        views_per_pass=cfg.device_max_views_per_render,
                    )
                    measured[name] = psnr(rendered, views.images).mean().item()
                    if drawing:
                        kept[name] = rendered.cpu()
                # With nothing to compensate the raw subsample is the
                # input, and is not rendered twice to say so
                measured.setdefault("psnr_thinned", measured["psnr_input"])
                if rule.enabled:
                    measured.update({
                        "psnr_vs_control": measured["psnr_refined"] - measured["psnr_control"],
                        "kept_fraction": refined.num_gaussians / reconstruction.gaussians.num_gaussians,
                        "rule_fraction": refined.num_gaussians / thinned.num_gaussians,
                        "scale_ratio": (
                            pruned.refined.scales.float().mean() / thinned.scales.float().mean()
                        ).item(),
                        **{
                            f"soft_{name}": value.mean().item()
                            for name, value in pruned.soft.items()
                        },
                    })
                for name, value in measured.items():
                    scores[budget][name].append(value)
                if psnr_full is not None:
                    # Per scene rather than between the means: the
                    # scenes whose whole field fit are a subset of the
                    # ones measured here, and a difference of means over
                    # two different sets is a gap no scene saw
                    scores[budget]["psnr_full"].append(psnr_full)
                    scores[budget]["psnr_vs_full"].append(
                        measured["psnr_refined"] - psnr_full
                    )
                if drawing:
                    # Its own try: the scores above are already in, so a
                    # figure that does not fit is a figure missing from
                    # the pass and not a scene missing from the metrics
                    try:
                        figures[(scene_idx, budget)] = refinement_figure(
                            cfg, reconstruction, thinned, refined, kept,
                            full_test, scene_idx, budget, num_context_views,
                        )
                    except RuntimeError as error:
                        if not out_of_memory(error):
                            raise
                        error.__traceback__ = None
                        tqdm.write(
                            f"  scene {scene_idx} was measured at {budget:,} "
                            f"Gaussians but did not fit being drawn"
                        )
            except RuntimeError as error:
                if not out_of_memory(error):
                    raise
                error.__traceback__ = None
                tqdm.write(
                    f"  scene {scene_idx} did not fit at {budget:,} Gaussians, skipped"
                )
            finally:
                del raw, thinned, refined, rendered, control
                torch.cuda.empty_cache()
        del reconstruction, full_test, order, score
        torch.cuda.empty_cache()

    splatformer.train(was_training)
    metrics = {}
    for budget, budget_scores in scores.items():
        num_scenes = len(budget_scores["psnr_input"])
        if num_scenes == 0:
            continue
        # Each mean over the scenes that produced that number: every
        # scene measured here has an input and a refined PSNR, but only
        # the ones whose whole field also fit have the two against it
        means = {
            name: sum(values) / len(values)
            for name, values in budget_scores.items() if values
        }
        means["psnr_compensation_gain"] = means["psnr_input"] - means["psnr_thinned"]
        means["psnr_gain"] = means["psnr_refined"] - means["psnr_input"]
        means["num_scenes"] = num_scenes
        metrics[budget] = means
    return metrics, figures


def run_validation(
    cfg,
    splatformer: SplatFormer,
    training_reconstructor,
    dataset: DL3DVDataset,
    scene_indices: Sequence[int],
    step: int,
    pruners: Sequence[Pruner],
    rule: LearnedRule,
):
    """
    Validate against every reconstructor named in
    reconstructor.validation, thinned by each rule of `pruners` in turn,
    and log the result under `val`.

    One reconstructor is on the card at a time: a held-out one is built
    for its pass and thrown away again, and the one being trained
    against is parked on the host while it is up.
    """
    device = next(splatformer.parameters()).device
    # The gradients of the step just taken are still allocated, and are
    # not read again: the next step's zero_grad() would be what frees
    # them, well after the pass that needs the room
    splatformer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    def held() -> str:
        """What is on the card right now, live and reserved."""
        return (
            f"{torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB live, "
            f"{torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB reserved"
        )

    tqdm.write(f"  before validating: {held()}")
    for name in cfg.reconstructor.validation:
        if name == cfg.reconstructor.training:
            # Off the card if the last training step ran out of memory
            # with it moved off (see training_step)
            reconstructor = training_reconstructor.to(device)
        else:
            tqdm.write(f"  building {name} for validation...")
            training_reconstructor.to("cpu")
            torch.cuda.empty_cache()
            reconstructor = build_reconstructor(
                name, cfg.anysplat_checkpoint, cfg.yonosplat_checkpoint,
            ).to(device)

        logged = {}
        seen = "trained on" if name == cfg.reconstructor.training else "held out"
        for rank, pruner in enumerate(pruners):
            for position, num_context_views in enumerate(cfg.validation.context_views):
                # The first rule and the first view count keep the plain
                # metric names, which is what every run before this
                # logged and what scripts/compare_runs.py reads; the rest
                # are logged under their own rule and view count beside it
                scope = ("" if rank == 0 else f"{pruner.slug}/") + (
                    "" if position == 0 else f"{num_context_views}v/"
                )
                thinned_by = "" if len(pruners) == 1 else f" by {pruner.name}"
                metrics, figures = validate(
                    cfg, splatformer, reconstructor, dataset, scene_indices,
                    num_context_views, pruner, rule,
                )
                for budget, scores in metrics.items():
                    fit = (
                        "" if scores["num_scenes"] == len(scene_indices)
                        else f" (over the {scores['num_scenes']} scenes that fit)"
                    )
                    # The stages in the order the field goes through
                    # them, each gain beside the stage that earned it
                    whole = (
                        "" if "psnr_full" not in scores
                        else f"{scores['psnr_full']:.2f} dB whole, "
                    )
                    compensated = (
                        "" if not cfg.compensation.enabled
                        else (
                            f"{scores['psnr_input']:.2f} dB compensated "
                            f"({scores['psnr_compensation_gain']:+.2f}), "
                        )
                    )
                    against = (
                        "" if "psnr_full" not in scores
                        else f", {scores['psnr_vs_full']:+.2f} dB against whole"
                    )
                    learned = (
                        "" if not rule.enabled else (
                            f"; the rule kept {scores['kept_fraction']:.1%} of the "
                            f"prediction ({scores['rule_fraction']:.1%} of what it was "
                            f"handed, scales x{scores['scale_ratio']:.2f}), against "
                            f"{scores['psnr_control']:.2f} dB for the compensated top-k "
                            f"at that count ({scores['psnr_vs_control']:+.2f})"
                        )
                    )
                    tqdm.write(
                        f"  [val] {name} ({seen}) from {num_context_views} views "
                        f"at {budget // 1000}k Gaussians{thinned_by}: {whole}"
                        f"{scores['psnr_thinned']:.2f} dB thinned, {compensated}"
                        f"{scores['psnr_refined']:.2f} dB refined "
                        f"({scores['psnr_gain']:+.2f}){against}{learned}{fit}"
                    )
                    logged.update({
                        f"val/{name}/{scope}{budget // 1000}k/{key}": scores[key]
                        for key in cfg.validation.logged_metrics if key in scores
                    })
                for (scene_idx, image_budget), figure in figures.items():
                    # A key of its own per scene, rule, view count and
                    # budget, so that the slider over a run walks one of
                    # them through the run rather than walking the scenes
                    # and budgets at one step
                    logged[
                        f"val/{name}/views/scene_{scene_idx}/"
                        f"{scope}{image_budget // 1000}k"
                    ] = wandb.Image(figure)
                    # wandb has taken its copy by now, and pyplot holds
                    # onto every figure it made until it is told not to
                    plt.close(figure)
                torch.cuda.empty_cache()
        wandb.log(logged, step=step)

        if name != cfg.reconstructor.training:
            del reconstructor
            torch.cuda.empty_cache()
            training_reconstructor.to(device)
        tqdm.write(f"  after {name}: {held()}")

    # A pass renders more Gaussians at once than any training step is
    # allowed to, and the step that follows it is the one most likely to
    # find the card full. What the pass cached is of no use to it.
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    # The whole configuration a run used, resolved, at the top of its
    # log: a sweep's logs are read long after the overrides that made
    # them have scrolled away
    print(OmegaConf.to_yaml(cfg))
    unswept = (
        set(cfg.validation.image_budgets)
        - set(cfg.validation.gaussian_budgets)
    )
    assert not unswept, (
        f"The figures are drawn off the budget sweep, so every budget they "
        f"are drawn at has to be one of {cfg.validation.gaussian_budgets}: "
        + ", ".join(f"{budget:,}" for budget in sorted(unswept)) + " is not"
    )
    unknown = (
        {cfg.reconstructor.training} | set(cfg.reconstructor.validation)
    ) - set(RECONSTRUCTORS)
    assert not unknown, (
        f"Every reconstructor a run names has to be one of "
        f"{RECONSTRUCTORS}: " + ", ".join(sorted(unknown)) + " is not"
    )
    pruners = [
        Pruner(**OmegaConf.to_container(rule, resolve=True))
        for rule in cfg.pruning.rules
    ]
    assert pruners, "There is nothing to thin with: no pruning rule is configured"
    slugs = [pruner.slug for pruner in pruners]
    assert len(set(slugs)) == len(slugs), (
        f"Two pruning rules would be logged under the same key: {slugs}"
    )
    load_dotenv()
    set_rng_seed(cfg.seed, deterministic=cfg.deterministic)
    # TF32 matmuls, which every model here is happy with and none of them
    # would run at a sensible speed without. AnySplat's vendored croco
    # turns them on as an import side effect, so this is really about not
    # depending on which submodule happened to be imported first.
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print("Initializing dataset...")
    dataset = DL3DVDataset(cfg.dl3dv_root_dir, cfg.dl3dv_images_subdir)
    splits = split_scenes(len(dataset), cfg.splits, cfg.split_seed)
    validation_scenes = splits["val"][:cfg.validation.num_scenes]
    print(
        f"DL3DV initialized with {len(dataset)} scenes: "
        + ", ".join(f"{len(scenes)} {name}" for name, scenes in splits.items())
        + f", validating on {len(validation_scenes)} of them."
    )

    print(
        f"Initializing frozen {cfg.reconstructor.training} from pre-trained "
        f"checkpoint..."
    )
    reconstructor = build_reconstructor(
        cfg.reconstructor.training, cfg.anysplat_checkpoint,
        cfg.yonosplat_checkpoint,
    ).to(device)

    start = (
        "zeroed output heads" if cfg.splatformer.zero_output_heads
        else "its own output heads"
    )
    additions = [
        name for name, wanted in (
            ("the importance score as an input channel", cfg.splatformer.importance_input),
            ("a mask head", cfg.splatformer.mask_head),
        ) if wanted
    ]
    print(
        f"Initializing SplatFormer from {cfg.splatformer.checkpoint} "
        f"with {start}" + (
            "" if not additions else ", " + " and ".join(additions)
        ) + "..."
    )
    splatformer = SplatFormer(
        str(cfg.splatformer.checkpoint),
        quiet=True,
        zero_output_heads=cfg.splatformer.zero_output_heads,
        gradient_checkpointing=cfg.splatformer.gradient_checkpointing,
        batch_statistics=cfg.splatformer.batch_statistics,
        importance_input=cfg.splatformer.importance_input,
        mask_head=cfg.splatformer.mask_head,
        importance_reference=cfg.pruning.learned.importance_reference,
        importance_eps=cfg.pruning.learned.importance_eps,
    ).to(device)
    splatformer.train()
    trainable = sum(p.numel() for p in splatformer.parameters() if p.requires_grad)
    print(f"SplatFormer has {trainable / 1e6:.1f}M trainable parameters.")
    rule = LearnedRule(
        opacity_weight=cfg.optim.opacity_loss_weight,
        opacity_threshold=cfg.pruning.learned.opacity_threshold,
        mask=cfg.splatformer.mask_head,
        mask_weight=cfg.optim.mask_loss_weight,
        mask_tau=cfg.pruning.learned.mask_tau,
        mask_straight_through=cfg.pruning.learned.mask_straight_through,
        quality_margin=cfg.pruning.learned.quality_margin_db,
        quality_rate=cfg.pruning.learned.quality_rate,
        quality_max_multiplier=cfg.pruning.learned.quality_max_multiplier,
        degradation_weight=cfg.pruning.learned.degradation.weight,
        degradation_reference=cfg.pruning.learned.degradation.reference,
        degradation_against_truth=cfg.pruning.learned.degradation.against_truth,
        degradation_power=cfg.pruning.learned.degradation.power,
        lossless_enabled=cfg.pruning.learned.lossless.enabled,
        lossless_quality=cfg.pruning.learned.lossless.quality,
        lossless_tolerance_db=cfg.pruning.learned.lossless.tolerance_db,
        lossless_tolerance_reference=cfg.pruning.learned.lossless.tolerance_reference,
        lossless_quality_floor_db=cfg.pruning.learned.lossless.quality_floor_db,
        lossless_quality_scale=cfg.pruning.learned.lossless.quality_scale,
        lossless_hinge=cfg.pruning.learned.lossless.hinge,
        lossless_hinge_softness_db=cfg.pruning.learned.lossless.hinge_softness_db,
        rate_controller=cfg.pruning.learned.lossless.controller,
        rate_controller_step=cfg.pruning.learned.lossless.controller_step,
        rate_controller_max_step=cfg.pruning.learned.lossless.controller_max_step,
        rate_weight_initial=cfg.pruning.learned.lossless.weight_initial,
        rate_weight_minimum=cfg.pruning.learned.lossless.weight_minimum,
        rate_weight_maximum=cfg.pruning.learned.lossless.weight_maximum,
        rate_reference_gaussians=cfg.pruning.learned.lossless.reference_gaussians,
        rate_size_exponent=cfg.pruning.learned.lossless.size_exponent,
        rate_controller_shared=cfg.pruning.learned.lossless.controller_shared,
        lossless_reference=cfg.pruning.learned.lossless.reference,
        compensate=cfg.pruning.learned.compensate,
        compensation_exponent=cfg.compensation.exponent,
    )
    controller = (
        QualityController(rule) if rule.mask and rule.quality_margin is not None else None
    )
    rates = RateController(rule) if rule.by_lossless and rule.rate_controller else None

    criterion = PhotometricLoss(
        l1_weight=cfg.optim.l1_loss_weight,
        l2_weight=cfg.optim.l2_loss_weight,
        lpips_weight=cfg.optim.lpips_loss_weight,
    )
    optimizer = build_splatformer_optimizer(
        splatformer.model,
        lr_dict=OmegaConf.to_container(cfg.optim.learning_rates, resolve=True),
    )
    scheduler = build_splatformer_scheduler(
        optimizer, schedule=cfg.optim.lr_schedule, total_step=cfg.optim.total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.optim.enable_amp)

    run = wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=cfg.wandb.run_name,
        notes=cfg.wandb.notes,
        settings=wandb.Settings(x_disable_stats=not cfg.wandb.system_metrics),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_dir = Path(cfg.log.output_dir) / (
        run.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing checkpoints to {checkpoint_dir}")

    # Every step's randomness, the scene it draws and the views and
    # Gaussians it samples, comes off this one generator
    generator = Generator().manual_seed(cfg.seed)

    oom_retries = 0

    staged_before = False
    histogram = (
        BudgetHistogram(
            bucket_size=cfg.budget.balance.bucket_size,
            max_gaussians=cfg.budget.balance.max_gaussians,
        ) if cfg.budget.balance.enabled else None
    )
    # Its own switch and its own buckets: a run can weigh its losses by
    # size whether the sizes themselves were balanced, drawn as a share,
    # or held at one number
    loss_histogram = (
        LossHistogram(
            bucket_size=cfg.optim.loss_balance.bucket_size,
            max_gaussians=cfg.optim.loss_balance.max_gaussians,
            momentum=cfg.optim.loss_balance.momentum,
            max_ratio=cfg.optim.loss_balance.max_ratio,
        ) if cfg.optim.loss_balance.enabled else None
    )
    if histogram is not None:
        keeps = (
            f"A step keeps whichever of the first "
            f"{histogram.max_gaussians:,} Gaussians' "
            f"{histogram.num_buckets} buckets of "
            f"{histogram.bucket_size:,} the run has trained on least and can "
            f"still reach by thinning"
        )
    else:
        keeps = (
            f"A step keeps {cfg.budget.min_fraction:.0%} to "
            f"{cfg.budget.max_fraction:.0%} of what the reconstructor predicts"
        )
    counts = [str(views) for views in cfg.context_views]
    drawn = counts[0] if len(counts) == 1 else (
        ", ".join(counts[:-1]) + " or " + counts[-1]
    )
    if any(pruner.measured for pruner in pruners):
        over = (
            "the context views" if cfg.pruning.max_scoring_views is None
            else f"at most {cfg.pruning.max_scoring_views} of the context views"
        )
        keeps += ", chosen " + (
            "by " if len(pruners) == 1 else "by one of "
        ) + ", ".join(
            f"{pruner.name!r} ({pruner.selection} out of the {pruner.score} score)"
            if pruner.measured else f"{pruner.name!r} (a uniform draw)"
            for pruner in pruners
        ) + f" measured over {over},"
    print(
        keeps
        + f" from {drawn} context views, held under "
        f"{cfg.device_max_gaussians:,} Gaussians, with the loss taken on "
        + (
            f"all of the '{cfg.supervision.views}' views"
            if cfg.supervision.max_views is None else
            f"at most {cfg.supervision.max_views} of the '{cfg.supervision.views}' views"
        )
        + f", in one graph while its estimate stays "
        f"{cfg.device_memory_safety_margin_gib:g} GiB under the card's "
        f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GiB and in "
        f"stages of {cfg.device_max_views_per_render} views otherwise. "
        f"Validating from "
        + " and ".join(str(views) for views in cfg.validation.context_views)
        + " context views at "
        + ", ".join(
            f"{budget // 1000}k" for budget in cfg.validation.gaussian_budgets
        )
        + " Gaussians."
    )
    if cfg.compensation.enabled:
        print(
            f"A thinned field has its opacities raised to "
            f"1 - (1 - a)^({cfg.compensation.exponent:g}/f) before anything "
            f"reads it, which puts back the optical depth that keeping a "
            f"share f of it took out."
        )
    if loss_histogram is not None:
        print(
            f"A step's loss is divided by what its own bucket of "
            f"{loss_histogram.bucket_size:,} Gaussians has been costing the "
            f"run, over the mean over every step, held inside a factor of "
            f"{loss_histogram.max_ratio:g} either way."
        )
    if rule.enabled:
        print(
            f"Which of the refined Gaussians survive is the network's call, "
            f"through {rule.describe()}. Validation measures it against the "
            f"compensated RadSplat top-k at the count it kept."
        )

    # The frames of a step are read while the step before it is still on
    # the GPU, so the first read has to be started before the loop
    prefetcher = ScenePrefetcher(dataset, device)
    draw = lambda: draw_step(cfg, dataset, splits["train"], generator)
    prefetcher.submit(draw())

    progress = tqdm(range(cfg.optim.total_steps), desc="Training", unit="step")
    for step in progress:
        if step % cfg.validation.interval == 0 and not (
            step == 0 and cfg.validation.skip_initial
        ):
            tqdm.write(f"Validating at step {step}...")
            run_validation(
                cfg, splatformer, reconstructor, dataset, validation_scenes,
                step, pruners, rule,
            )

        no_room = False
        try:
            plan, scene = prefetcher.take()
        except RuntimeError as error:
            # Moving a step's frames onto the card is the smallest
            # allocation a step makes, and it is made before any of the
            # room a step needs is asked for, so a failure here is not
            # this step being too big for the card: it is the card still
            # holding what something before it left behind
            if not out_of_memory(error):
                raise
            no_room = True
        if no_room:
            tqdm.write(f"Step {step}: no room to read a scene in, clearing")
            recover_from_oom(optimizer, scaler)
            plan, scene = prefetcher.take()
        if step + 1 < cfg.optim.total_steps:
            prefetcher.submit(draw())
        scene_idx = plan.scene_idx
        # What the plan drew can be more than the scene holds, in which
        # case plan_views() lowered it, so the count that goes into the
        # log is the one the step actually reconstructed from
        context_views = len(plan.context_idx)
        context_idx, test_idx = plan.context_idx, plan.test_idx
        budget_fraction = plan.budget_fraction
        pruner = draw_pruner(pruners, generator)

        step_result = None
        # What a retry after an out-of-memory shrinks the step by,
        # applied to the count the step would otherwise have kept
        # whether a draw or the histogram decided it
        budget_scale = 1.0
        for attempt in range(cfg.optim.oom_max_retries + 1):
            ran_out = False
            try:
                step_result = training_step(
                    cfg, splatformer, reconstructor, criterion, optimizer,
                    scaler, scene, context_idx, test_idx, budget_fraction,
                    generator, pruner, rule, histogram=histogram,
                    loss_histogram=loss_histogram, budget_scale=budget_scale,
                    controller=controller,
                    rates=rates,
                )
                break
            except RuntimeError as error:
                if not out_of_memory(error):
                    raise
                ran_out = True
            # Recovered out here, and not in the handler above, because
            # for as long as a handler is running the interpreter holds
            # the exception it is handling, which holds the frames of
            # the step that failed, which hold that step's autograd
            # graph: several gigabytes of it, on a card that has just
            # said it has none left. Collecting in there frees nothing,
            # since nothing is unreferenced yet, and the step retries
            # into the same wall. Leaving the handler drops the frames.
            if ran_out:
                recover_from_oom(optimizer, scaler)
                oom_retries += 1
                budget_scale *= cfg.optim.oom_retry_factor
                if attempt < cfg.optim.oom_max_retries:
                    tqdm.write(
                        f"Step {step}: out of memory, retrying scene {scene_idx} "
                        f"on {budget_scale:.0%} of the budget "
                        f"({attempt + 1}/{cfg.optim.oom_max_retries})"
                    )
        if step_result is None:
            tqdm.write(f"Step {step}: skipped, scene {scene_idx} did not fit")
            continue
        if step_result.stages > 1 and not staged_before:
            staged_before = True
            tqdm.write(
                f"Step {step}: one graph of {step_result.reconstruction.gaussians.num_gaussians:,} "
                f"Gaussians over {len(step_result.views)} views is estimated at "
                f"{step_result.memory_estimate:.1f} GiB, within "
                f"{cfg.device_memory_safety_margin_gib:g} GiB of the card; the loss is "
                f"rendered and backpropagated in {step_result.stages} stages of "
                f"{cfg.device_max_views_per_render} views from here on whenever that is so"
            )
        reconstruction, views, rendered = (
            step_result.reconstruction, step_result.views, step_result.rendered
        )
        loss, terms, loss_scale = step_result.loss, step_result.terms, step_result.loss_scale
        budget_fraction, predicted_gaussians = step_result.budget_fraction, step_result.predicted
        pruned, halves = step_result.pruned, step_result.halves
        # Counted here rather than where the size was chosen, so that a
        # step lands in the bucket it trained on: one that ran out of
        # memory and was retried thinner is an example of the smaller
        # size, and one that never fit at all is not an example at all
        if histogram is not None:
            histogram.record(reconstruction.gaussians.num_gaussians)
        # Recorded here for the same reason: the loss of a step that was
        # retried thinner is what that thinner size cost, and a step that
        # never fit cost nothing that the sizes it was asked for explain
        if loss_histogram is not None:
            loss_histogram.record(reconstruction.gaussians.num_gaussians, loss)
        scheduler.step()

        if step % cfg.log.interval == 0:
            with torch.no_grad():
                baseline, _ = reconstruction.gaussians.rasterize(
                    views.poses, views.intrinsics, views.image_shape,
                    views_per_pass=cfg.device_max_views_per_render,
                )
                # What the rule would deliver, rendered on the same
                # views: the render the loss saw is the soft one
                survivors = rendered
                if rule.enabled:
                    survivors, _ = pruned.kept.rasterize(
                        views.poses, views.intrinsics, views.image_shape,
                        views_per_pass=cfg.device_max_views_per_render,
                    )
            # The input render is what the gain is measured against, so
            # it is taken every time it is logged and charted only as
            # that difference
            per_view = {
                name: psnr(images, views.images)
                for name, images in (
                    ("input", baseline), ("refined", rendered), ("pruned", survivors)
                )
            }
            psnr_input = per_view["input"].mean().item()
            psnr_refined = per_view["refined"].mean().item()
            metrics = {
                f"train/{name}": value for name, value in terms.items()
            }
            # Each half of the scene on its own, when both are in whole:
            # self-reconstruction on the views the reconstructor saw and
            # novel view synthesis on the ones held out from it
            metrics.update({
                f"train/psnr_{name}_{half}": values[part].mean().item()
                for name, values in per_view.items()
                for half, part in halves.items()
            })
            metrics.update({
                "train/num_gaussians": reconstruction.gaussians.num_gaussians,
                "train/predicted_gaussians": predicted_gaussians,
                "train/budget_fraction": budget_fraction,
                "train/context_views": context_views,
                "train/pruning_rule": pruners.index(pruner),
                "train/supervision_views": len(views),
                # In how many renders the loss was backpropagated, and
                # what one graph of it was estimated at (see
                # training_step)
                "train/backward_stages": step_result.stages,
                "train/memory_estimate_gib": step_result.memory_estimate,
                "train/loss": loss,
                "train/oom_retries": oom_retries,
                "train/memory_live_gib": torch.cuda.memory_allocated() / 2 ** 30,
                "train/memory_peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "train/psnr_refined": psnr_refined,
                "train/psnr_gain": psnr_refined - psnr_input,
            })
            if loss_histogram is not None:
                metrics.update({
                    # What the size this step landed on costs against the
                    # run's mean, and the loss that came of dividing by it:
                    # a scale that stays at 1 is a run whose budgets are
                    # not pulling unevenly in the first place
                    "train/loss_scale": loss_scale,
                    "train/loss_balanced": loss / loss_scale,
                })
            if rule.enabled:
                num_gaussians = reconstruction.gaussians.num_gaussians
                metrics.update({
                    "train/psnr_pruned": per_view["pruned"].mean().item(),
                    # Of what the refiner was handed, and of the prediction
                    "train/hard_fraction": pruned.num_kept / num_gaussians,
                    "train/kept_fraction": pruned.num_kept / predicted_gaussians,
                    # Whether a sparsity term is being paid in size rather
                    # than in Gaussians: the same coverage at a lower
                    # opacity needs a wider footprint
                    "train/scale_ratio": (
                        pruned.refined.scales.float().mean()
                        / reconstruction.gaussians.scales.float().mean()
                    ).item(),
                    **{
                        f"train/soft_{name}": value.mean().item()
                        for name, value in pruned.soft.items()
                    },
                })
                if step_result.quality_gap is not None:
                    metrics.update({
                        # What the cut cost on the step's views, and the
                        # weight the controller paid the mask term at
                        # (negative: a bonus for keeping)
                        "train/quality_gap_db": step_result.quality_gap,
                        "train/mask_weight": step_result.mask_weight,
                    })
            if histogram is not None:
                metrics.update({
                    "train/budget_bucket": histogram.bucket_of(
                        reconstruction.gaussians.num_gaussians
                    ),
                    # Falls over a run as the counts level out, so this
                    # is what says whether the balancing is keeping up
                    # with what the reconstructor is handing it
                    "train/budget_spread": histogram.spread,
                })
            wandb.log(metrics, step=step)
            progress.set_postfix(
                loss=f"{loss:.4f}",
                psnr_gain=f"{metrics['train/psnr_gain']:+.2f}",
            )
            kept = "" if not rule.enabled else (
                f", {metrics['train/psnr_pruned']:.2f} dB keeping "
                f"{metrics['train/hard_fraction']:.1%} of them "
                f"(scales x{metrics['train/scale_ratio']:.2f})"
            )
            tqdm.write(
                f"Step {step}: loss {loss:.4f}, "
                f"PSNR {psnr_input:.2f} -> "
                f"{psnr_refined:.2f} dB "
                f"over {reconstruction.gaussians.num_gaussians:,} Gaussians "
                f"({budget_fraction:.1%} of the "
                f"{predicted_gaussians:,} predicted){kept} "
                f"from {context_views} context views"
                + ("" if len(pruners) == 1 else f" by {pruner.name}")
                + f", supervised on {len(views)}"
                + (
                    "" if step_result.stages == 1 else
                    f" in {step_result.stages} stages ({step_result.memory_estimate:.1f} GiB estimated)"
                )
                + f", peak "
                f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB"
            )

        if (step + 1) % cfg.log.save_interval == 0:
            path = checkpoint_dir / f"model_{step + 1:08d}.pth"
            torch.save(splatformer.model.state_dict(), path)
            tqdm.write(f"Wrote {path}")

        del step_result, reconstruction, views, rendered, pruned

    progress.close()
    prefetcher.close()
    if histogram is not None:
        print(
            f"Trained on {histogram.total} fields, spread "
            f"{histogram.spread:.2f} over {histogram.num_buckets} buckets "
            f"of {histogram.bucket_size:,} Gaussians:\n"
            + histogram.summary()
        )
    if loss_histogram is not None and loss_histogram.total > 0:
        print(
            f"The mean loss of each of the {loss_histogram.num_buckets} "
            f"buckets of {loss_histogram.bucket_size:,} Gaussians, against a "
            f"mean of {loss_histogram.mean:.4f} over the run:\n"
            + loss_histogram.summary()
        )
    run_validation(
        cfg, splatformer, reconstructor, dataset, validation_scenes,
        cfg.optim.total_steps, pruners, rule,
    )
    torch.save(splatformer.model.state_dict(), checkpoint_dir / "model_final.pth")
    wandb.finish()


if __name__ == "__main__":
    main()
