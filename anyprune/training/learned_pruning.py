"""
Pruning as something the refiner learns, rather than a budget it is
handed: which of the refined Gaussians survive is read off the
network's own output, and a sparsity term in the loss pays it to let go
of what the render does not need.

Two rules, each its own switch, which can be on together:

  opacity  A term on the refined opacities,

               loss += opacity_weight * mean_i(opacity_i)

           and at inference every Gaussian whose refined opacity is at
           or below opacity_threshold is dropped. No new parameters:
           the refiner already has an opacity head, and what the
           rasterizer blends with is what the term reads. The mean is
           over the count, so that the weight means the same thing on a
           field of 100k Gaussians and on one of 500k.

  mask     The LP-3DGS route (arXiv 2405.18784): the refiner's mask
           head (see anyprune.models.wrappers.splatformer) gives every
           Gaussian a logit, which a Gumbel-Sigmoid turns into a soft
           keep in [0, 1],

               keep_i = sigmoid((logit_i + g0 - g1) / tau)
               loss  += mask_weight * mean_i(keep_i)

           with the soft mask multiplying the refined opacities in the
           field the loss renders, and its 0.5 level set (logit > 0)
           dropping Gaussians at inference. The difference of two
           Gumbels is logistic noise, added before the temperature
           divides, which pushes the mask towards 0 and 1 while keeping
           it differentiable.

           With 'mask_straight_through' the loss renders the hard mask
           instead - the very cut inference makes, logit > 0 - and the
           gradient goes through the noisy soft one (the value is the
           hard mask, the derivative the Gumbel-Sigmoid's). Without it
           the mask can park Gaussians just under the cut, where they
           are gone at inference but still lend a third of their
           opacity to the training render; on a dense field enough of
           them together cover every surface, the loss stays low, and
           the delivered field loses 10 dB (2026-09-19, 8-16 context
           views). The straight-through render makes the loss pay for
           the cut. The cut has to be the noiseless one: rendering the
           cut of the *noisy* mask instead, every logit went to about
           -1, where nothing survives at inference but the noise keeps
           a fresh random quarter of a dense field every step, which
           renders well enough (2026-09-19, collapsed to 0.1% kept).

           Rendered hard, a Gaussian that is cut gets no gradient from
           the render at all (the rasterizer culls it at alpha < 1/255),
           only the sparsity term's push further down: the mask can
           shrink but never grow, and did, to 2% of a 16-view field
           (2026-09-20). The QualityController below is the way back
           in: the weight of the mask term is lowered, down to a keep
           bonus, by a multiplier that grows while the hard cut costs
           more than a margin in dB against the render of the field the
           refiner was handed, on the step's views, and decays while it
           costs less. One multiplier per context view count, since a
           dense field and a sparse one settle at different fractions.
           The reference is the input, not the refined field left
           whole: rendered hard, the refiner writes anything into what
           it cuts, and the whole refined render came out 1-9 dB under
           the cut (2026-09-20), which told the controller nothing.

           Against the input the controller engaged, and the mask
           went to 100% at every count within 100 steps and stayed
           there: the bonus drove the logits past the point where the
           sigmoid still has a gradient, an absorbing state at the
           other end. At 16 views it could not have settled anyway,
           since the refiner with everything kept renders 1-2 dB under
           its input, further than the margin. Anything built on the
           straight-through render needs logits that cannot saturate
           (bounded, or regularized towards zero) and a margin set
           against what the refiner can reach, not the input; as of
           2026-09-20 neither is done, and the soft mask is the
           default.

Neither rule drops anything during training - the opacity rule makes a
Gaussian transparent, the mask rule fades it - so the field keeps its
size and the gradient reaches everything. The saving shows at
inference, where the hard rule drops what training made invisible. With
both rules on, a Gaussian survives only if both keep it.

The photometric loss pulls on the mask from every pixel, the ones whose
error is the reconstructor's or the refiner's included, which no mask
can mend; on a dense field most of the error is that kind. The
degradation term (degradation_weight > 0) charges the mask for the
damage it did and nothing else: the masked render against a reference
render of the same views that has nothing masked out, per pixel,

    d_p = |masked_p - reference_p|                       (against the reference)
    d_p = relu(|masked_p - truth_p| - |reference_p - truth_p|)   (against the truth)

    loss += degradation_weight * mean_p(d_p ^ degradation_power)

with the reference held out of the gradient (it would otherwise be
taught to get worse, since that too closes the gap). The reference is
the field the refiner was handed ('input', what the benchmark measures
a pruning against) or the refined field left whole ('refined'). A
power of 2 weighs the holes pruning leaves - a few percent of pixels
gone very wrong - the way an L1 mean does not. Against the reference
the term needs no ground truth, so it can be taken on any camera.

With 'compensate', what the rule keeps has the optical depth of the
thinning put back the way Gaussians.compensate() does for a budget: a
ray meets a share f as many Gaussians, so their opacities are raised to
1 - (1 - a)^(exponent / f). At inference f is the share the hard rule
kept; during training it is the mean soft keep, held out of the
gradient, and the raise is applied to the faded opacities the loss
renders, so that the refiner is trained on what it will be delivered
with.
"""
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import math
import torch
from torch import Tensor

from ..gaussians import Gaussians


@dataclass(frozen=True)
class LearnedRule:
    """
    Which of the rules above a run trains with, and their constants.
    'mask' says whether the refiner has a mask head; the other two
    switches are the weights themselves, off at zero.
    """
    opacity_weight: float = 0.0
    opacity_threshold: float = 0.005
    mask: bool = False
    mask_weight: float = 0.0
    mask_tau: float = 0.5
    mask_straight_through: bool = False
    # The controller's margin in dB (None: off), its rate per dB per
    # step, and the most the multiplier can grow to
    quality_margin: Optional[float] = None
    quality_rate: float = 0.002
    quality_max_multiplier: float = 0.05
    # The degradation term (see the module docstring): its weight, off
    # at zero; whose render it is measured against; whether the
    # per-pixel damage is read against that render or as the error
    # increase against the truth; and the power the damage is raised to
    degradation_weight: float = 0.0
    degradation_reference: str = "input"
    degradation_against_truth: bool = False
    degradation_power: float = 2.0
    lossless_enabled: bool = False
    lossless_quality: str = "deviation"
    lossless_tolerance_db: float = 0.1
    lossless_tolerance_reference: str = "truth"
    lossless_quality_floor_db: float = 40.0
    lossless_quality_scale: float = 8.0
    lossless_hinge: bool = True
    lossless_hinge_softness_db: float = 0.1
    rate_controller: bool = False
    rate_controller_step: float = 0.002
    rate_controller_max_step: float = 0.005
    rate_weight_initial: float = 0.005
    rate_weight_minimum: float = -0.05
    rate_weight_maximum: float = 0.05
    rate_reference_gaussians: float = 100_352.0
    rate_size_exponent: float = 0.0
    rate_controller_shared: bool = False
    lossless_reference: str = "input"
    compensate: bool = False
    compensation_exponent: float = 1.0

    def __post_init__(self):
        assert self.opacity_weight >= 0.0 and self.mask_weight >= 0.0, (
            f"A sparsity weight cannot be negative: got {self.opacity_weight} "
            f"and {self.mask_weight}"
        )
        assert 0.0 < self.opacity_threshold < 1.0, (
            f"The opacity cut has to be inside (0, 1): got {self.opacity_threshold}"
        )
        assert self.mask_tau > 0.0, f"The temperature has to be positive: got {self.mask_tau}"
        assert self.quality_margin is None or self.quality_margin >= 0.0, (
            f"The quality margin is a loss in dB, at least zero: got {self.quality_margin}"
        )
        assert self.quality_rate > 0.0 and self.quality_max_multiplier > 0.0, (
            f"Need a positive controller rate and ceiling, got {self.quality_rate} "
            f"and {self.quality_max_multiplier}"
        )
        assert self.compensation_exponent > 0.0, (
            f"Need a positive compensation exponent, got {self.compensation_exponent}"
        )
        assert self.degradation_weight >= 0.0 and self.degradation_power > 0.0, (
            f"The degradation weight cannot be negative and its power has to be positive: "
            f"got {self.degradation_weight} and {self.degradation_power}"
        )
        assert self.degradation_reference in ("input", "refined"), (
            f"The degradation reference is 'input' or 'refined', got {self.degradation_reference!r}"
        )
        assert self.lossless_reference in ("input", "refined"), (
            "The lossless objective is measured against the field the refiner was handed "
            f"('input') or against the refined field left whole ('refined'), got {self.lossless_reference!r}"
        )
        assert self.lossless_quality in ("deviation", "drop"), (
            "The lossless objective charges the 'deviation' from the reference render or "
            f"the PSNR 'drop' against the truth, got {self.lossless_quality!r}"
        )
        assert self.lossless_tolerance_reference in ("truth", "fixed"), (
            "The tolerance of the lossless objective is read against 'truth' or at a "
            f"'fixed' floor, got {self.lossless_tolerance_reference!r}"
        )
        assert self.lossless_tolerance_db >= 0.0 and self.lossless_quality_scale > 0.0, (
            f"Need a non-negative tolerance and a positive scale, got "
            f"{self.lossless_tolerance_db} and {self.lossless_quality_scale}"
        )

    @property
    def by_opacity(self) -> bool:
        return self.opacity_weight > 0.0

    @property
    def enabled(self) -> bool:
        """Whether anything of a refined field is dropped at all."""
        return self.mask or self.by_opacity

    @property
    def by_lossless(self) -> bool:
        """Whether the rate is traded against a quality budget rather than weighed."""
        return self.lossless_enabled

    @property
    def by_degradation(self) -> bool:
        """Whether the loss charges the rule for the damage it does."""
        return self.enabled and self.degradation_weight > 0.0

    def describe(self) -> str:
        parts = []
        if self.by_opacity:
            parts.append(
                f"a sparsity term of {self.opacity_weight:g} on the mean refined "
                f"opacity, with everything at or below {self.opacity_threshold:g} "
                f"dropped at inference"
            )
        if self.mask:
            parts.append(
                f"a Gumbel-Sigmoid mask head at tau={self.mask_tau:g} weighed at "
                f"{self.mask_weight:g}, with logits below zero dropped at inference"
                + (", rendered hard in training with the gradient through the soft mask"
                   if self.mask_straight_through else "")
                + (f", its weight lowered by a multiplier (at most {self.quality_max_multiplier:g}, "
                   f"{self.quality_rate:g} per dB per step) while the cut costs more than "
                   f"{self.quality_margin:g} dB against the render of its input"
                   if self.quality_margin is not None else "")
            )
        described = " and ".join(parts) if parts else "no learned rule"
        if self.by_degradation:
            described += (
                f", charged {self.degradation_weight:g} per unit of the mean "
                f"{'error increase against the truth' if self.degradation_against_truth else 'difference'}"
                f"^{self.degradation_power:g} of its render against the "
                f"{'refined field left whole' if self.degradation_reference == 'refined' else 'input'}"
            )
        if self.enabled and self.compensate:
            described += (
                f", with the opacities of what it keeps raised to "
                f"1 - (1 - a)^({self.compensation_exponent:g}/f) for the share f kept"
            )
        return described


class QualityController:
    """
    The multiplier that lowers the mask term's weight while the hard cut
    costs more than 'rule.quality_margin' dB, one per key (the context
    view count of the step). See the module docstring.
    """

    def __init__(self, rule: LearnedRule):
        assert rule.mask and rule.quality_margin is not None, (
            "The controller needs a mask head and a margin to hold it to"
        )
        self.rule = rule
        self.multiplier: Dict[int, float] = {}

    def weight(self, key: int) -> float:
        """What the mask term is weighed at on a step of this key."""
        return self.rule.mask_weight - self.multiplier.get(key, 0.0)

    def update(self, key: int, gap: float) -> float:
        """
        Move the key's multiplier by how far the cut's cost 'gap' (dB)
        stands from the margin, and return it.
        """
        moved = self.multiplier.get(key, 0.0) + self.rule.quality_rate * (gap - self.rule.quality_margin)
        self.multiplier[key] = min(max(moved, 0.0), self.rule.quality_max_multiplier)
        return self.multiplier[key]


@dataclass
class Pruned:
    """
    What one forward of the refiner says about a field under the rule.
    """
    # The refiner's output, untouched
    refined: Gaussians
    # What the photometric loss renders: the refined field itself, or
    # with the soft mask on its opacities
    trained: Gaussians
    # (N,) in [0, 1] per rule that is on: the per-Gaussian keep term
    # the sparsity loss averages ('opacity', 'mask')
    soft: Dict[str, Tensor]
    # (N,) bool: what inference keeps
    hard: Tensor
    # The exponent the kept share compensates the survivors with, or
    # None when they are delivered as refined
    compensation: Optional[float] = None

    @property
    def num_kept(self) -> int:
        return int(self.hard.sum().item())

    def detach(self) -> "Pruned":
        """The same, off the autograd graph, to be kept past the step."""
        def free(gaussians: Gaussians) -> Gaussians:
            return replace(gaussians, **{
                name: getattr(gaussians, name).detach()
                for name in ("means", "covariances", "harmonics", "opacities", "scales", "rotations")
            })
        refined = free(self.refined)
        trained = refined if self.trained is self.refined else free(self.trained)
        return Pruned(
            refined, trained,
            {name: value.detach() for name, value in self.soft.items()}, self.hard,
            self.compensation,
        )

    @property
    def kept(self) -> Gaussians:
        """
        The refined Gaussians inference keeps, never none of them: a
        field that lost everything keeps the one the rule was least
        sure about dropping.
        """
        keep = self.hard
        if not bool(keep.any()):
            least = torch.stack(list(self.soft.values())).sum(dim=0) if self.soft else keep
            keep = least.argmax().reshape(1)
        kept = self.refined[keep]
        if self.compensation is not None:
            kept = kept.compensate(
                kept.num_gaussians / self.refined.num_gaussians, exponent=self.compensation
            )
        return kept


def gumbel_sigmoid(logits: Tensor, tau: float, noisy: bool = True) -> Tensor:
    """
    The soft mask in [0, 1] a set of logits stands for: with 'noisy'
    the Gumbel-Sigmoid of LP-3DGS, without it the plain sigmoid, whose
    0.5 level set is the hard mask. Always in single precision: the
    noise saturates in half at this temperature.
    """
    with torch.cuda.amp.autocast(enabled=False):
        logits = logits.float()
        if noisy:
            uniform = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            logits = logits + uniform.log() - torch.log1p(-uniform)
        return torch.sigmoid(logits / tau)


def apply_rule(
    rule: LearnedRule, refined: Gaussians, logits: Optional[Tensor], noisy: bool
) -> Pruned:
    """
    Read the rule off a refined field and the mask logits that came
    with it (None without a mask head). 'noisy' is the training-time
    Gumbel noise; inference reads the plain sigmoid.

    With no rule on, everything is kept and the field trains as it is.
    """
    assert rule.mask == (logits is not None), (
        "The rule and the refiner disagree on whether there is a mask head"
    )
    soft: Dict[str, Tensor] = {}
    hard = torch.ones(refined.num_gaussians, dtype=torch.bool, device=refined.device)
    trained = refined
    if rule.by_opacity:
        opacities = refined.opacities.float()
        soft["opacity"] = opacities
        hard &= opacities > rule.opacity_threshold
    if rule.mask:
        mask = gumbel_sigmoid(logits, rule.mask_tau, noisy=noisy)
        soft["mask"] = mask
        hard &= logits > 0
        keep = mask
        if rule.mask_straight_through:
            # Inference's own cut as the value, the (noisy) soft mask
            # as what the gradient flows through
            keep = (logits > 0).to(mask.dtype) + mask - mask.detach()
        trained = replace(refined, opacities=refined.opacities * keep)
    compensation = None
    if rule.enabled and rule.compensate:
        compensation = rule.compensation_exponent
        # The share the loss compensates for is what the rule keeps of
        # the field - the soft mask where there is one, the opacity cut
        # where there is not - as a number rather than as part of the
        # graph: what the gradient moves is the opacities, not the share
        keep = hard.float()
        if rule.mask:
            keep = soft["mask"] * (keep if rule.by_opacity else 1.0)
        fraction = keep.mean().detach().clamp_min(1e-6)
        opacities = trained.opacities.clamp(0.0, 1.0)
        trained = replace(
            trained, opacities=1.0 - (1.0 - opacities) ** (compensation / fraction)
        )
    return Pruned(refined, trained, soft, hard, compensation)


def degradation_loss(
    rule: LearnedRule, rendered: Tensor, reference: Tensor, truth: Optional[Tensor] = None
) -> Tensor:
    """
    What the rule pays for the damage its render did against a
    reference render of the same views (see the module docstring), a
    mean over the pixels of (V, 3, H, W) images. 'truth' is read only
    against the truth. The reference is detached here, whatever it came
    from.
    """
    assert rendered.shape == reference.shape, (
        f"Rendered {tuple(rendered.shape)} against a reference of {tuple(reference.shape)}"
    )
    reference = reference.detach()
    if rule.degradation_against_truth:
        assert truth is not None, "Reading the damage against the truth needs the truth"
        damage = torch.relu((rendered - truth).abs() - (reference - truth).abs())
    else:
        damage = (rendered - reference).abs()
    return rule.degradation_weight * damage.pow(rule.degradation_power).mean()


def lossless_loss(
    rule: LearnedRule, rendered: Tensor, reference: Tensor, keep: Tensor,
    truth: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    The rate against a quality budget rather than weighed against it:
    the larger of what the cut costs and what it keeps, so that the
    mask is only ever pushed down while the render it makes stays
    within 'lossless_tolerance_db' of the reference render (detached;
    the field the refiner was handed).

        quality = scale * relu(excess in dB)
        rate    = mean keep
        loss    = max(quality, rate)

    Inside the budget the quality term is zero, so the loss is the rate
    and the only gradient pushes the mask down; outside it the quality
    term climbs past the rate and the only gradient pushes the render
    back. The equilibrium sits where the two cross, a rate/scale of a
    dB outside the budget, so the scale is how hard the constraint is
    held (10 leaves a twentieth of a dB at a half-kept field).

    Both terms are written in dB because the errors themselves are not
    comparable to a share: an error a hundred times the budget is a
    loss of a hundred against a rate below one, and the step is then
    whatever the gradient clipping leaves of it, the same for every
    violation. In dB the term is a few units at a few dB out.

    'lossless_quality' says what the excess is measured on:

      - 'deviation' (no truth in the gradient): the mean squared
        difference from the reference render against a budget, which is
        either the share of the reference's own error against the truth
        that costs tolerance_db of PSNR - (10^(tol/10) - 1) times it,
        the error a render may add to drop the reference's PSNR by the
        tolerance when the two are uncorrelated, the truth read as a
        scale only and never differentiated - or, with
        'lossless_tolerance_reference' at 'fixed', the error of a
        render lossless_quality_floor_db from the reference, which
        needs no truth at all. Note that this charges the refiner for
        *improving* on its input as much as for damaging it: every
        deviation eats the budget.
      - 'drop' (the truth in the gradient): the PSNR the render loses
        against the truth compared to the reference, less the
        tolerance, so that a render better than the reference is free
        and only damage is charged.
    """
    reference = reference.detach()
    assert rendered.shape == reference.shape, (
        f"Rendered {tuple(rendered.shape)} against a reference of {tuple(reference.shape)}"
    )
    floor = torch.finfo(torch.float32).tiny
    if rule.lossless_quality == "drop":
        assert truth is not None, "The PSNR drop is read against the truth"
        ours = (rendered - truth).pow(2).mean()
        theirs = (reference - truth).pow(2).mean().detach()
        excess = 10.0 * torch.log10(ours.clamp_min(floor) / theirs.clamp_min(floor)) - rule.lossless_tolerance_db
        measured = {"lossless_drop_db": excess.item() + rule.lossless_tolerance_db}
    else:
        error = (rendered - reference).pow(2).mean()
        if rule.lossless_tolerance_reference == "truth":
            assert truth is not None, "The budget against the truth needs the truth"
            budget = (reference - truth).pow(2).mean().detach() * (
                10.0 ** (rule.lossless_tolerance_db / 10.0) - 1.0
            )
        else:
            budget = torch.full_like(error, 10.0 ** (-rule.lossless_quality_floor_db / 10.0))
        excess = 10.0 * torch.log10(error.clamp_min(floor) / budget.clamp_min(floor))
        measured = {
            "lossless_psnr_vs_reference": -10.0 * math.log10(max(error.item(), 1e-20)),
            "lossless_budget_used_db": excess.item(),
        }
    if not rule.lossless_hinge:
        quality = excess
    elif rule.lossless_hinge_softness_db > 0.0:
        # Softened, so that the quality term starts pushing back just
        # before the boundary rather than switching on at it: a hard
        # hinge leaves the rate as the only gradient until the budget
        # is spent, and the step that discovers the boundary has
        # already crossed it (2026-09-22: the mask collapsed to a
        # twentieth of the field in eighty steps that way).
        softness = rule.lossless_hinge_softness_db
        quality = softness * torch.nn.functional.softplus(excess / softness)
    else:
        quality = torch.relu(excess)
    quality = rule.lossless_quality_scale * quality
    rate = keep.mean()
    loss = torch.maximum(quality, rate)
    return loss, {
        "lossless": loss.item(), "lossless_quality": quality.item(),
        "lossless_rate": rate.item(), "lossless_excess_db": excess.item(),
        **measured,
    }


class RateController:
    """
    The same constraint as lossless_loss, held with a multiplier
    instead of a maximum: the loss is the photometric one plus
    'weight' per unit of mean keep, and the weight is raised while the
    render is inside the budget and lowered while it is outside,

        w <- w + rate * (tolerance - drop in dB)

    clamped to [minimum, maximum], one weight per context view count
    since the slack differs by an order of magnitude between them.

    This is dual ascent on 'prune as much as possible while losing at
    most tolerance_db', and it is what the maximum cannot do from a
    starting point that already violates the constraint: there the
    quality branch is hundreds of times the photometric loss, gradient
    clipping makes every step maximal, and the run walks off (the mask
    collapsed to a seventh of the field on 2026-09-22 that way). Here
    the gradient a step takes is always the size of the photometric
    loss, and only the pressure on the rate moves, slowly.
    """
    def __init__(self, rule: "LearnedRule"):
        self.rule = rule
        self.weights: Dict[int, float] = {}

    def weight(self, key: int) -> float:
        return self.weights.setdefault(key, self.rule.rate_weight_initial)

    def update(self, key: int, drop_db: float) -> float:
        """
        The weight after a step that lost 'drop_db' at this view count,
        moved by the slack it left and bounded per step so that one bad
        scene cannot swing the pressure.

        The move is additive and the weight may go negative, which is a
        bonus per unit kept rather than a price: a multiplier that can
        only fall to zero leaves nothing pushing the mask back up, and
        the photometric loss alone does not do it - the render is
        nearly flat in the mask over a wide range (probed 2026-09-22:
        PSNR against the truth moves by 0.01 dB between a mask kept at
        61% and at 68%), so with no pressure the mask drifts, and it
        drifted down through three dB of loss.
        """
        weight = self.weight(key)
        step = self.rule.rate_controller_step * (self.rule.lossless_tolerance_db - drop_db)
        step = max(min(step, self.rule.rate_controller_max_step), -self.rule.rate_controller_max_step)
        weight = weight + step
        weight = max(min(weight, self.rule.rate_weight_maximum), self.rule.rate_weight_minimum)
        self.weights[key] = weight
        return weight


def rate_term(rule: LearnedRule, keep: Tensor, num_gaussians: int) -> Tensor:
    """
    What the loss charges per unit kept, as a share of the field, with
    the pressure scaled by how large the field is:

        rate = mean keep * (reference / N) ** exponent

    At exponent 0 it is the plain kept share, which pushes as hard on a
    field of a million as on one of a hundred thousand; above it the
    pressure falls with the size, so that a dense field - which is the
    one a mask damages most, and the one whose view count the refiner
    was trained on least - is pruned more carefully. The reference is
    the size the exponent leaves untouched (2 context views of
    YoNoSplat, 100,352).

    Unlike a weight per context view count, this is a function of the
    field the step is holding, so it says something about a count the
    run never trained on: the 24, 32 and 64-view fields the benchmark
    ends on.
    """
    rate = keep.mean()
    if rule.rate_size_exponent == 0.0:
        return rate
    scale = (rule.rate_reference_gaussians / max(num_gaussians, 1)) ** rule.rate_size_exponent
    return rate * scale


def psnr_drop(rendered: Tensor, reference: Tensor, truth: Tensor) -> float:
    """
    How many dB of PSNR against the truth the render loses against the
    reference render, positive when it is the worse of the two.
    """
    ours = (rendered.detach().float() - truth).pow(2).mean().item()
    theirs = (reference.detach().float() - truth).pow(2).mean().item()
    return 10.0 * math.log10(max(ours, 1e-20) / max(theirs, 1e-20))


def sparsity_loss(
    rule: LearnedRule, pruned: Pruned, mask_weight: Optional[float] = None
) -> Tuple[Tensor, Dict[str, float]]:
    """
    What the rule adds to the photometric loss, and each term of it for
    logging, named after the rule it belongs to. Zero, and no terms,
    with no rule on. 'mask_weight' is the controller's weight for the
    mask term on this step, in place of the rule's; it can be negative.
    """
    terms = {}
    if rule.by_opacity:
        terms["opacity"] = rule.opacity_weight * pruned.soft["opacity"].mean()
    if rule.mask:
        weight = rule.mask_weight if mask_weight is None else mask_weight
        terms["mask"] = weight * pruned.soft["mask"].mean()
    total = sum(terms.values()) if terms else torch.zeros((), device=pruned.refined.device)
    return total, {name: term.item() for name, term in terms.items()}


__all__ = [
    "LearnedRule",
    "Pruned",
    "QualityController",
    "apply_rule",
    "degradation_loss",
    "RateController",
    "gumbel_sigmoid",
    "lossless_loss",
    "psnr_drop",
    "sparsity_loss",
]
