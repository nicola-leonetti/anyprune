"""
Fine-tune SplatFormer to adjust the subsampled Gaussians coming from a 
frozen feed-forward reconstructor.

A training step:
    - samples an even run of frames from one scene
    - hands half of them to the frozen reconstructor as context views
    - refines the Gaussians it predicts with SplatFormer 
    - supervises with photometric loss on both context and test views

An evaluation step is performed once on the same feedforward 
reconstructor used during training and once with a different one.
The number of frames for evaluation is fixed.
"""
import gc
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from torch import Generator, Tensor
from tqdm import tqdm

from anyprune.datasets import DL3DVDataset
from anyprune.evaluation import psnr
from anyprune.models import FrozenAnySplat, FrozenYoNoSplat, SplatFormer
from anyprune.models.utils import (
    build_splatformer_optimizer, build_splatformer_scheduler,
)
from anyprune.training import (
    BudgetHistogram, PhotometricLoss, fit_budget_fraction, reconstruct,
    sample_budget_fraction, sample_num_context_views, sample_view_indices,
)
from anyprune.utils import set_rng_seed
from anyprune.viz import RefinementBlock, plot_refinement


ROOT = Path(__file__).resolve().parent.parent


# The feed-forward reconstructors a run can name, as
# reconstructor.training and reconstructor.validation in
# configs/train.yaml
RECONSTRUCTORS = ("AnySplat", "AnySplat-voxelized", "YoNoSplat")


def build_reconstructor(cfg, name: str) -> nn.Module:
    """
    Build one of RECONSTRUCTORS from the checkpoint the config names.

    Built on demand rather than kept in a table, since each of these is
    a few GB of weights and a run only ever has one of them on the card
    at a time.
    """
    assert name in RECONSTRUCTORS, (
        f"Every reconstructor a run names has to be one of "
        f"{RECONSTRUCTORS}: {name} is not"
    )
    if name == "YoNoSplat":
        return FrozenYoNoSplat(cfg.yonosplat_checkpoint, quiet=True)
    # The same weights read two ways: the voxelized one fuses the
    # per-pixel Gaussians onto a grid inside the encoder, which is what
    # the released checkpoint configures itself for
    return FrozenAnySplat(
        cfg.anysplat_checkpoint,
        quiet=True,
        voxelize=name == "AnySplat-voxelized",
    )


def load_dotenv():
    """
    Read the .env file at the root of the repository, which is where the
    README asks for the Weights & Biases API key, without pulling in a
    dependency to do it.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def out_of_memory(error: BaseException) -> bool:
    """
    Whether an exception is the card running out of memory.

    torch raises OutOfMemoryError for an allocation it makes itself, but
    the libraries underneath do not all go through it, and each says so
    in its own words:

    - spconv, which is most of SplatFormer's backbone, allocates through
      cumm and reports a plain RuntimeError naming the failure;
    - cuDNN, which is where the reconstructor's convolutions run, cannot
      allocate a workspace and reports instead that it could not find an
      engine to run the computation at all;
    - cuBLAS reports an allocation status.

    All of them are the same event as far as a step is concerned, and a
    run that only knows the first dies on the others instead of retrying
    the step on a smaller share of the prediction. The cuDNN wording is
    not exclusively about memory - an unsupported configuration says the
    same thing - but a configuration that has run for hundreds of steps
    does not become unsupported, so on this path it means the card is
    full.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return any(
        wording in message for wording in (
            "out of memory",
            "unable to find an engine to execute this computation",
            "cublas_status_alloc_failed",
        )
    )


def split_scenes(cfg, num_scenes: int) -> Dict[str, List[int]]:
    """
    Divide the scenes between the splits, in the proportions the config
    names and always the same way.
    """
    order = torch.randperm(
        num_scenes, generator=Generator().manual_seed(cfg.split_seed)
    ).tolist()
    splits, first = {}, 0
    for name, fraction in cfg.splits.items():
        last = first + round(fraction * num_scenes)
        splits[name] = order[first:last]
        first = last
    # Rounding can leave a scene over, which goes to the training split
    splits["train"].extend(order[first:])
    return splits


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
    histogram: Optional[BudgetHistogram] = None,
    budget_scale: float = 1.0,
):
    """
    Reconstruct, refine, score and take one optimizer step, returning
    what the logging needs. Raises on a card that ran out of memory,
    which the caller retries on a smaller budget.

    How much of the prediction is kept comes from `histogram` when there
    is one, which sends the step wherever the run is short of examples,
    and from `budget_fraction` when there is not. `budget_scale` is what
    a retry after an out-of-memory shrinks the step by, and applies
    whichever of the two decided it.
    """
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
    reconstruction.gaussians = reconstruction.gaussians.subsample(
        kept, generator=generator
    )
    views = reconstruction.views(cfg.supervision.views).thin(
        cfg.supervision.max_views, generator=generator
    )

    refined = refine(cfg, splatformer, reconstruction.gaussians)
    rendered, _ = refined.rasterize(
        views.poses, views.intrinsics, views.image_shape,
        views_per_pass=cfg.device_max_views_per_render,
    )
    loss, terms = criterion(rendered, views.images)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if cfg.optim.grad_clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            splatformer.parameters(), cfg.optim.grad_clip_norm
        )
    scaler.step(optimizer)
    scaler.update()
    return (
        reconstruction, views, rendered.detach(), loss.item(), terms,
        budget_fraction, predicted,
    )


def recover_from_oom(optimizer, scaler):
    """
    Put the optimizer and the loss scaler back into a state a fresh
    attempt can start from, and hand the memory back.

    The collection is what actually hands it back. A step that runs out
    of memory does so deep inside its own call stack, and the exception
    carrying that failure up holds a traceback, which holds those
    frames, which hold the step's locals: the reconstruction, the
    refined Gaussians, the renders, and the whole autograd graph behind
    them. Frames and tracebacks reference each other, so none of it is
    freed by reference counting when the exception goes out of scope -
    it waits for the cyclic collector, which knows how many container
    objects are alive and nothing about the gigabytes of CUDA memory
    hanging off them, and so has no reason to run. Emptying the cache
    without collecting first returns nothing, because none of it is
    free yet.

    Which is what turns one step that did not fit into a run that dies:
    the graph of the first failure stays resident, so the next step has
    less to work with, fails in turn, and leaves its own graph behind.

    The scaler tracks where in the step each optimizer is: an OOM
    between unscale_() and step() leaves it believing this iteration was
    already unscaled, and the next attempt's unscale_() would refuse.
    update() is what closes an iteration out. It refuses in turn if
    there is nothing to close, which is the ordinary case here since
    most of a step's memory is allocated well before unscale_() is
    reached, so that refusal is the signal that no cleanup was needed.
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

    A view count the scene is too short for is lowered to what it does
    hold, since half the views are held out and a run of 2V frames at
    this stride has to fit inside the capture: at sixty-four context
    views that is 128 frames, and two of DL3DV's 541 scenes are shorter
    than that. Lowering it costs those scenes the top of the view range
    and keeps them in the run; asserting would end the run on whichever
    step first drew one of them.

    Kept apart from read_scene() below so that the drawing can stay on
    the main thread while the reading is handed to a worker.
    """
    num_frames = dataset.num_frames(scene_idx)
    # An even number of views, so that the two halves are the same size
    holds = 2 * (((num_frames - 1) // cfg.view_stride + 1) // 2)
    num_views = min(2 * num_context_views, holds)
    assert num_views >= 2, (
        f"Scene {scene_idx} has {num_frames} frames, too few for a context "
        f"view and a held-out one at stride {cfg.view_stride}"
    )
    return sample_view_indices(
        num_frames, num_views, stride=cfg.view_stride, generator=generator,
    )


def read_scene(dataset: DL3DVDataset, scene_idx: int, frames: Tensor):
    """
    Read the named frames of a scene off disk, leaving them on the host.

    Nothing here touches CUDA or draws a random number, which is what
    makes it safe to run on the prefetch thread.
    """
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
    reconstructed.

    The budget is a share rather than a count because the count is not
    known yet: it depends on how many Gaussians the reconstructor makes
    of the views drawn here, which is only settled once it has run.
    """
    scene_idx: int
    num_context_views: int
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
    share of the prediction it keeps, in the order they have always been
    drawn.
    """
    scene_idx = scenes[torch.randint(len(scenes), (1,), generator=generator).item()]
    num_context_views = sample_num_context_views(
        cfg.context_views.min, cfg.context_views.max, generator=generator
    )
    frames, context_idx, test_idx = plan_views(
        cfg, dataset, scene_idx, num_context_views, generator
    )
    return StepPlan(
        scene_idx=scene_idx,
        num_context_views=num_context_views,
        frames=frames,
        context_idx=context_idx,
        test_idx=test_idx,
        budget_fraction=sample_budget_fraction(
            cfg.budget.min_fraction, cfg.budget.max_fraction, generator=generator
        ),
    )


class ScenePrefetcher:
    """
    Read the frames of the next step while the GPU is still working on
    this one.

    Reading a step's frames costs about 50 ms of decoding, all of it
    with the GPU idle, against a step of roughly 1.5 s. One worker is
    enough to hide it: the read of step n + 1 starts as soon as step n's
    plan has been drawn and is collected a step later, by which time it
    has long finished.

    Only the reading moves. Every draw stays on the main thread, in the
    order draw_step() makes them, so a run is still reproducible from
    its seed alone. What does change is where those draws fall relative
    to the Gaussian subsampling inside a step, which comes off the same
    generator: a step is now planned before the step ahead of it
    subsamples rather than after. A prefetching run therefore does not
    repeat a run from before this, step for step, though it draws from
    the same distribution.
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
    cfg, splatformer: SplatFormer, gaussians, enable_amp: Optional[bool] = None
):
    """
    Run SplatFormer over a set of Gaussians, in half precision if asked.
    The rasterizer downstream is left in single precision, which is what
    the wrapper hands back whatever it ran in.
    """
    with torch.cuda.amp.autocast(
        enabled=cfg.optim.enable_amp if enable_amp is None else enable_amp
    ):
        return splatformer(gaussians)


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

    The whole prediction is the ceiling the thinning is spending
    against, so it goes in the figure with its own PSNR and its own
    count: the gap between it and the thinned row is what the budget
    cost, and the gap refinement closes is only worth reading next to
    it. It is left out when the budget was wider than the prediction,
    since the thinning then returns the same Gaussians and the row would
    be the row below it drawn twice, and when its render of the held-out
    views did not fit, since there is then nothing to draw the block
    below from.

    The held-out renders of the thinned and refined Gaussians, and of
    the whole prediction, are the ones the pass already took for its
    metrics, handed in rather than drawn again. What the figure costs
    over a pass without it, for one scene at one budget: two renders of
    the context views, and one more of the whole prediction over them,
    where there was any thinning to show.

    Everything is moved to the host as it is built. What is held onto
    while the figure is drawn would otherwise be GPU memory the widest
    budget of the next scene is about to want.
    """
    context, test = reconstruction.context, reconstruction.test
    predicted = reconstruction.gaussians
    thinning = (
        predicted.num_gaussians > thinned.num_gaussians and full_test is not None
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
        num_gaussians=thinned.num_gaussians,
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
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Figure]]:
    """
    Score the refined Gaussians against the input ones and against the
    whole prediction on the held-out views of every validation scene, at
    each of the fixed budgets in validation.gaussian_budgets, and return
    the three mean PSNRs and the two gains for each: psnr_gain, what
    refinement won back over the thinned Gaussians, and psnr_vs_full,
    what the refined Gaussians are worth against the whole prediction
    they were thinned out of.

    A scene is reconstructed once and then thinned down to each budget
    in turn, so the levels are read off the same prediction. Both the
    views and the thinning come off a generator seeded by the scene, so
    a scene is always validated on the same frames and the same
    Gaussians however far into the run we are and whichever
    reconstructor produced them.

    A scene that does not fit is dropped rather than allowed to end the
    run, the same treatment a training step gets, and for a sharper
    reason: the budgets swept here run to several times what a step is
    allowed to draw, so the widest of them are the first thing to
    overflow on a card the rest of the run fits on. A budget's means are
    then taken over the scenes that did fit, and a budget no scene fit
    at is left out of the metrics entirely rather than reported as a
    number standing on nothing.

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
            "psnr_input": [], "psnr_refined": [],
            "psnr_full": [], "psnr_vs_full": [],
        }
        for budget in cfg.validation.gaussian_budgets
    }
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
            thinned = refined = rendered = None
            try:
                thinned = reconstruction.gaussians.subsample(
                    budget,
                    generator=Generator().manual_seed(
                        cfg.split_seed + scene_idx
                    ),
                )
                # In single precision whatever training runs in: spconv
                # takes a different path once a module leaves training
                # mode, and it has no half-precision kernel to offer there
                refined = refine(cfg, splatformer, thinned, enable_amp=False)
                # Held back until both renders are in, so that a budget
                # that overflows halfway through does not leave the two
                # PSNRs averaged over different sets of scenes, which
                # would put a gain between them that no scene measured
                measured, kept = {}, {}
                drawing = drawing_scene and budget in cfg.validation.image_budgets
                for name, gaussians in (
                    ("psnr_input", thinned), ("psnr_refined", refined)
                ):
                    rendered, _ = gaussians.rasterize(
                        views.poses, views.intrinsics, views.image_shape,
                        views_per_pass=cfg.device_max_views_per_render,
                    )
                    measured[name] = psnr(rendered, views.images).mean().item()
                    if drawing:
                        kept[name] = rendered.cpu()
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
                del thinned, refined, rendered
                torch.cuda.empty_cache()
        del reconstruction, full_test
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
    prefix: str = "val",
):
    """
    Validate against every reconstructor named in
    reconstructor.validation and log the result.

    `prefix` is what the metrics are logged under, so that a pass taken
    on something other than the live weights is charted beside them
    rather than over them.

    One reconstructor is on the card at a time: a held-out one is built
    for its pass and thrown away again, and the one being trained
    against is parked on the host while it is up, since two feed-forward
    reconstructors and a point transformer do not comfortably share a
    single consumer GPU. AnySplat and YoNoSplat are about 6 GB of
    weights between them, which is where a 16 GB card's validation pass
    went before this: leaving both resident is what put the widest
    budgets out of reach, not what they are rendering.

    The weights make the trip over PCIe twice per held-out reconstructor
    per pass, a second or two of a pass that takes minutes, and they are
    frozen, so nothing is lost by moving them.
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
            reconstructor = training_reconstructor
        else:
            tqdm.write(f"  building {name} for validation...")
            training_reconstructor.to("cpu")
            torch.cuda.empty_cache()
            reconstructor = build_reconstructor(cfg, name).to(device)

        logged = {}
        seen = "trained on" if name == cfg.reconstructor.training else "held out"
        for position, num_context_views in enumerate(cfg.validation.context_views):
            # The first view count keeps the plain metric names, which
            # is what every run before this logged and what
            # scripts/compare_runs.py reads; the rest are logged under
            # their own view count beside it
            scope = "" if position == 0 else f"{num_context_views}v/"
            metrics, figures = validate(
                cfg, splatformer, reconstructor, dataset, scene_indices,
                num_context_views,
            )
            for budget, scores in metrics.items():
                fit = (
                    "" if scores["num_scenes"] == len(scene_indices)
                    else f" (over the {scores['num_scenes']} scenes that fit)"
                )
                whole = (
                    "" if "psnr_full" not in scores
                    else (
                        f", {scores['psnr_full']:.2f} dB whole "
                        f"({scores['psnr_vs_full']:+.2f} dB against it)"
                    )
                )
                tqdm.write(
                    f"  [{prefix}] {name} ({seen}) from {num_context_views} views at "
                    f"{budget // 1000}k Gaussians: "
                    f"{scores['psnr_input']:.2f} dB in, "
                    f"{scores['psnr_refined']:.2f} dB out, "
                    f"{scores['psnr_gain']:+.2f} dB{whole}{fit}"
                )
                logged.update({
                    f"{prefix}/{name}/{scope}{budget // 1000}k/{key}": scores[key]
                    for key in cfg.validation.logged_metrics if key in scores
                })
            for (scene_idx, image_budget), figure in figures.items():
                # A key of its own per scene, view count and budget, so
                # that the slider over a run walks one of them through
                # the run rather than walking the scenes and budgets at
                # one step
                logged[
                    f"{prefix}/{name}/views/scene_{scene_idx}/"
                    f"{scope}{image_budget // 1000}k"
                ] = wandb.Image(figure)
                # wandb has taken its copy by now, and pyplot holds onto
                # every figure it made until it is told not to
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
    splits = split_scenes(cfg, len(dataset))
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
    reconstructor = build_reconstructor(cfg, cfg.reconstructor.training).to(device)

    start = (
        "zeroed output heads" if cfg.splatformer.zero_output_heads
        else "its own output heads"
    )
    print(
        f"Initializing SplatFormer from {cfg.splatformer.checkpoint} "
        f"with {start}..."
    )
    splatformer = SplatFormer(
        str(cfg.splatformer.checkpoint),
        quiet=True,
        zero_output_heads=cfg.splatformer.zero_output_heads,
        gradient_checkpointing=cfg.splatformer.gradient_checkpointing,
    ).to(device)
    splatformer.train()
    trainable = sum(p.numel() for p in splatformer.parameters() if p.requires_grad)
    print(f"SplatFormer has {trainable / 1e6:.1f}M trainable parameters.")

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
    histogram = (
        BudgetHistogram(
            bucket_size=cfg.budget.balance.bucket_size,
            max_gaussians=cfg.budget.balance.max_gaussians,
        ) if cfg.budget.balance.enabled else None
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
    print(
        keeps
        + f" from {cfg.context_views.min} to "
        f"{cfg.context_views.max} context views, held under "
        f"{cfg.device_max_gaussians:,} Gaussians, with the loss taken on at "
        f"most {cfg.supervision.max_views} of the '{cfg.supervision.views}' views. "
        f"Validating from "
        + " and ".join(str(views) for views in cfg.validation.context_views)
        + " context views at "
        + ", ".join(
            f"{budget // 1000}k" for budget in cfg.validation.gaussian_budgets
        )
        + " Gaussians."
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
                step,
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
                    generator, histogram=histogram, budget_scale=budget_scale,
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
        (
            reconstruction, views, rendered, loss, terms, budget_fraction,
            predicted_gaussians,
        ) = step_result
        # Counted here rather than where the size was chosen, so that a
        # step lands in the bucket it trained on: one that ran out of
        # memory and was retried thinner is an example of the smaller
        # size, and one that never fit at all is not an example at all
        if histogram is not None:
            histogram.record(reconstruction.gaussians.num_gaussians)
        scheduler.step()

        if step % cfg.log.interval == 0:
            with torch.no_grad():
                baseline, _ = reconstruction.gaussians.rasterize(
                    views.poses, views.intrinsics, views.image_shape,
                    views_per_pass=cfg.device_max_views_per_render,
                )
            # The input render is what the gain is measured against, so
            # it is taken every time it is logged and charted only as
            # that difference
            psnr_input = psnr(baseline, views.images).mean().item()
            psnr_refined = psnr(rendered, views.images).mean().item()
            metrics = {
                f"train/{name}": value for name, value in terms.items()
            }
            metrics.update({
                "train/num_gaussians": reconstruction.gaussians.num_gaussians,
                "train/predicted_gaussians": predicted_gaussians,
                "train/budget_fraction": budget_fraction,
                "train/context_views": context_views,
                "train/supervision_views": len(views),
                "train/loss": loss,
                "train/oom_retries": oom_retries,
                "train/memory_live_gib": torch.cuda.memory_allocated() / 2 ** 30,
                "train/memory_peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "train/psnr_refined": psnr_refined,
                "train/psnr_gain": psnr_refined - psnr_input,
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
            tqdm.write(
                f"Step {step}: loss {loss:.4f}, "
                f"PSNR {psnr_input:.2f} -> "
                f"{psnr_refined:.2f} dB "
                f"over {reconstruction.gaussians.num_gaussians:,} Gaussians "
                f"({budget_fraction:.1%} of the "
                f"{predicted_gaussians:,} predicted) "
                f"from {context_views} context views, "
                f"supervised on {len(views)}, peak "
                f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB"
            )

        if (step + 1) % cfg.log.save_interval == 0:
            path = checkpoint_dir / f"model_{step + 1:08d}.pth"
            torch.save(splatformer.model.state_dict(), path)
            tqdm.write(f"Wrote {path}")

        del step_result, reconstruction, views, rendered

    progress.close()
    prefetcher.close()
    if histogram is not None:
        print(
            f"Trained on {histogram.total} fields, spread "
            f"{histogram.spread:.2f} over {histogram.num_buckets} buckets "
            f"of {histogram.bucket_size:,} Gaussians:\n"
            + histogram.summary()
        )
    run_validation(
        cfg, splatformer, reconstructor, dataset, validation_scenes,
        cfg.optim.total_steps,
    )
    torch.save(splatformer.model.state_dict(), checkpoint_dir / "model_final.pth")
    wandb.finish()


if __name__ == "__main__":
    main()
