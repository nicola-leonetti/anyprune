"""
Image features of the context views, lifted onto the Gaussians of a
field reconstructed from them, for the refiner to read next to the
Gaussians' own parameters.

ReSplat (Xu et al., 2025) conditions its Gaussians on dense features of
the input views: monocular (Depth Anything V2, a DINOv2 encoder), CNN
and multi-view features at its initialization, lifted to the 3D points
they were unprojected to; and, in its recurrent update, ResNet-18
features of the error between a render of the current Gaussians and the
input views. A refiner handed a field from any reconstructor has no
pixel each Gaussian came from, so here a Gaussian reads what it covers:
the blending-weighted mean, over every context view and every pixel it
contributes to, of the feature maps of those views,

    f_i = sum_{v, p} w_vip F_vp / sum_{v, p} w_vip

with w_vip the weight the Gaussian carries in pixel p of view v,
T_vip * alpha_vip. That is what a render's backward computes for the
colour of every Gaussian, given the feature map as the gradient of the
image, so one rasterizer backward per view and channel slice measures
it (splat_back), or, cheaper, gathering the map at every intersection
the rasterizer composites (splat_back_swept), which is what lift() does.
It is occlusion-aware (a Gaussian hidden in a view gets nothing from
it) and needs no correspondence between Gaussians and pixels. The
weight itself, log(1 + sum w), goes along as one more channel: a
Gaussian no view sees reads zeros and says so.

The sources, any of them, in this order:

    dino          DINOv2 ViT-B/14 patch tokens (semantic, 1/14)
    resnet        ResNet-18 at 1/2, 1/4 and 1/8 (ReSplat's update features)
    rgb           the views themselves
    rgb-error     render minus view (ReSplat's image error)
    resnet-error  ResNet-18 of the render minus of the view (its feature error)

The two backbones are frozen ImageNet/LVD weights, and their channels
are cut down by a fixed PCA fitted once on training frames
(fit_projections), whitened, so that a field of 800k Gaussians carries
64 channels of each rather than 768 or 256.
"""
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import gsplat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..gaussians import Gaussians
from ..gaussians.gaussians import view_matrices

SOURCES = ("dino", "resnet", "rgb", "rgb-error", "resnet-error")
# The backbone each source runs, if any
_BACKBONE = {"dino": "dino", "resnet": "resnet", "resnet-error": "resnet"}
# Channels a backbone's features are cut down to
COMPONENTS = 64
# Where the PCA of each backbone lives
PROJECTIONS_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "image_features"
# The resolution the backbones read the views at (a multiple of DINOv2's
# 14-pixel patch: 32 x 32 tokens) and the one the features are lifted at:
# at 112 a 16-view field's features are those of 224 (median cosine 0.995,
# the same Gaussians unseen) in a quarter of the intersections (2.2 s
# against 7.4, 2026-10-08)
BACKBONE_RESOLUTION = 448
LIFT_RESOLUTION = 112
# How many channels one rasterizer kernel carries back, and how many
# intersections a swept splat gathers at once
_CHANNELS_PER_PASS = 32
_INTERSECTIONS_PER_SLICE = 1_000_000
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINO_HUB = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


def projection_path(backbone: str) -> Path:
    return PROJECTIONS_DIR / f"{backbone}_pca{COMPONENTS}.pt"


class _ResNetFeatures(nn.Module):
    """ResNet-18's features at 1/2, 1/4 and 1/8 (64 + 64 + 128 channels), all brought to 1/2."""

    def __init__(self):
        super().__init__()
        import torchvision
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2

    def forward(self, x: Tensor) -> Tensor:
        half = self.stem(x)
        quarter = self.layer1(self.pool(half))
        eighth = self.layer2(quarter)
        size = half.shape[-2:]
        return torch.cat([
            half,
            F.interpolate(quarter, size=size, mode="bilinear", align_corners=False),
            F.interpolate(eighth, size=size, mode="bilinear", align_corners=False),
        ], dim=1)


class _DinoFeatures(nn.Module):
    """DINOv2 ViT-B/14's last normalized patch tokens, as a (768, H/14, W/14) map."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(str(_DINO_HUB), "dinov2_vitb14", source="local", pretrained=True)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        rows, columns = x.shape[-2] // 14, x.shape[-1] // 14
        return tokens.transpose(1, 2).reshape(x.shape[0], -1, rows, columns)


def build_backbone(name: str) -> nn.Module:
    return {"dino": _DinoFeatures, "resnet": _ResNetFeatures}[name]()


def splat_back(
    gaussians: Gaussians,
    maps: Tensor,
    poses: Tensor,
    intrinsics: Tensor,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    views_per_pass: int = 2,
) -> Tuple[Tensor, Tensor]:
    """
    (N, C) the blending-weighted sum of 'maps' (V, C, h, w) over every
    pixel of every view each Gaussian contributes to, and (N,) the sum
    of its weights, with 'intrinsics' in pixels of the maps: the colour
    gradient of a render whose image gradient is the map.
    """
    viewmats = view_matrices(poses.reshape(-1, 4, 4).to(gaussians.means))
    intrinsics = intrinsics.reshape(-1, 3, 3).to(gaussians.means)
    num_views, channels, height, width = maps.shape
    means, covariances = gaussians.means.float(), gaussians.covariances.float()
    opacities = gaussians.opacities.float()
    # The map with a channel of ones, whose sum is the weight
    maps = torch.cat([maps.float(), maps.new_ones(num_views, 1, height, width, dtype=torch.float32)], dim=1)
    total = torch.zeros(gaussians.num_gaussians, channels + 1, device=means.device)
    # Every channel in one call, which projects and sorts the field once
    # and rasterizes the channels in slices
    colours = torch.zeros(gaussians.num_gaussians, channels + 1, device=means.device, requires_grad=True)
    for first in range(0, num_views, views_per_pass):
        group = slice(first, first + views_per_pass)
        with torch.enable_grad():
            rendered, _, _ = gsplat.rasterization(
                means=means, quats=None, scales=None, opacities=opacities, colors=colours,
                viewmats=viewmats[group], Ks=intrinsics[group],
                width=width, height=height, near_plane=near_plane, far_plane=far_plane,
                sh_degree=None, covars=covariances, channel_chunk=_CHANNELS_PER_PASS,
            )
            gradient, = torch.autograd.grad(rendered, colours, grad_outputs=maps[group].permute(0, 2, 3, 1))
        total += gradient
        del rendered, gradient
    return total[:, :-1], total[:, -1]


@torch.no_grad()
def splat_back_swept(
    gaussians: Gaussians,
    maps: Tensor,
    poses: Tensor,
    intrinsics: Tensor,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
) -> Tuple[Tensor, Tensor]:
    """
    What splat_back() measures, from the weights the front-to-back sweep
    of anyprune.gaussians.importance reads off the rasterizer, gathered
    and summed per intersection: the rasterizer's backward recomputes
    every Gaussian's gradient once per slice of channels, which at 16
    views and 200 channels costs ten times this.
    """
    from ..gaussians.importance import sweep
    num_views, channels, height, width = maps.shape
    # (V, h * w, C), a row per ray
    rows = maps.flatten(2).transpose(1, 2).contiguous()
    total = torch.zeros(gaussians.num_gaussians, channels, device=maps.device)
    weight = torch.zeros(gaussians.num_gaussians, device=maps.device)
    for walked in sweep(gaussians, poses, intrinsics, (height, width), near_plane, far_plane):
        weight.index_add_(0, walked.gaussian_ids, walked.weights.float())
        for first in range(0, walked.gaussian_ids.numel(), _INTERSECTIONS_PER_SLICE):
            part = slice(first, first + _INTERSECTIONS_PER_SLICE)
            total.index_add_(
                0, walked.gaussian_ids[part],
                rows[walked.view, walked.rays[part]] * walked.weights[part].float().unsqueeze(-1),
            )
    return total, weight


class ImageFeatures(nn.Module):
    """
    The frozen extractor: feature maps of a set of views, and their
    lifting onto a field (lift()). Never trained and never saved with
    the refiner: everything here is fixed by its sources.
    """

    def __init__(self, sources: Sequence[str]):
        super().__init__()
        sources = tuple(sources)
        assert sources and all(s in SOURCES for s in sources), (
            f"Image feature sources are among {SOURCES}, got {sources}"
        )
        self.sources = tuple(s for s in SOURCES if s in sources)
        self.backbones = nn.ModuleDict()
        for name in sorted({_BACKBONE[s] for s in self.sources if s in _BACKBONE}):
            self.backbones[name] = build_backbone(name)
            projection = torch.load(projection_path(name), map_location="cpu")
            self.register_buffer(f"{name}_mean", projection["mean"])
            self.register_buffer(f"{name}_projection", projection["projection"])
        self.requires_grad_(False)
        self.eval()
        self.register_buffer("_image_mean", torch.tensor(_IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("_image_std", torch.tensor(_IMAGENET_STD).reshape(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Frozen, BatchNorm and all
        return super().train(False)

    @property
    def needs_render(self) -> bool:
        return any(s.endswith("-error") for s in self.sources)

    @property
    def channels(self) -> int:
        """What lift() returns per Gaussian: every source's channels, then the log weight."""
        widths = {"dino": COMPONENTS, "resnet": COMPONENTS, "rgb": 3, "rgb-error": 3, "resnet-error": COMPONENTS}
        return sum(widths[s] for s in self.sources) + 1

    def _backbone(self, name: str, images: Tensor, size: Tuple[int, int]) -> Tensor:
        """A backbone's features of (V, 3, H, W) images in [0, 1], projected and brought to 'size'."""
        x = F.interpolate(images.float(), size=(BACKBONE_RESOLUTION, BACKBONE_RESOLUTION),
                          mode="bilinear", align_corners=False, antialias=True)
        features = self.backbones[name]((x - self._image_mean) / self._image_std).float()
        mean, projection = getattr(self, f"{name}_mean"), getattr(self, f"{name}_projection")
        features = torch.einsum("vchw,ck->vkhw", features - mean.reshape(1, -1, 1, 1), projection)
        return F.interpolate(features, size=size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def maps(self, images: Tensor, rendered: Optional[Tensor] = None) -> Tensor:
        """
        (V, C - 1, h, w) the feature maps of (V, 3, H, W) views in [0, 1],
        at LIFT_RESOLUTION, every source in order; 'rendered' is the
        field's render of the same views, which the error sources need.
        """
        size = (LIFT_RESOLUTION, LIFT_RESOLUTION)
        assert rendered is not None or not self.needs_render, "The error sources need the field's render"
        small = lambda x: F.interpolate(x.float(), size=size, mode="area")
        cache: Dict[str, Tensor] = {}

        def backbone(name: str) -> Tensor:
            if name not in cache:
                cache[name] = self._backbone(name, images, size)
            return cache[name]

        out = []
        for source in self.sources:
            if source in ("dino", "resnet"):
                out.append(backbone(source))
            elif source == "rgb":
                out.append(small(images))
            elif source == "rgb-error":
                out.append(small(rendered) - small(images))
            elif source == "resnet-error":
                # The projection is affine, so its mean cancels
                out.append(self._backbone("resnet", rendered, size) - backbone("resnet"))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def lift(self, gaussians: Gaussians, images: Tensor, poses: Tensor, intrinsics: Tensor,
             views_per_pass: int = 4) -> Tensor:
        """
        (N, channels) what every Gaussian of the field sees of the views
        (V, 3, H, W) it was reconstructed from, with 'poses' and
        'intrinsics' (in pixels of 'images') as Gaussians.rasterize()
        takes them: the weighted mean of every source, then log(1 + the
        weight).
        """
        height, width = images.shape[-2:]
        rendered = None
        if self.needs_render:
            rendered, _ = gaussians.rasterize(poses, intrinsics, (height, width), views_per_pass=views_per_pass)
            rendered = rendered.clamp(0.0, 1.0)
        maps = self.maps(images, rendered)
        del rendered
        scale = torch.tensor([LIFT_RESOLUTION / width, LIFT_RESOLUTION / height, 1.0], device=intrinsics.device)
        summed, weight = splat_back_swept(gaussians, maps, poses, intrinsics * scale.reshape(1, 3, 1))
        features = summed / weight.clamp_min(1e-6).unsqueeze(-1)
        return torch.cat([features, weight.log1p().unsqueeze(-1)], dim=-1)


@torch.no_grad()
def fit_projections(images: Tensor, backbones: Sequence[str] = ("dino", "resnet"),
                    samples_per_image: int = 4096, generator: Optional[torch.Generator] = None) -> Dict[str, dict]:
    """
    The whitened PCA of each backbone's features, from (V, 3, H, W)
    training frames in [0, 1], at 'samples_per_image' random pixels of
    each feature map: {name: {mean (C,), projection (C, COMPONENTS),
    explained (COMPONENTS,) share of the variance}}.
    """
    out = {}
    device = images.device
    mean_ = torch.tensor(_IMAGENET_MEAN, device=device).reshape(1, 3, 1, 1)
    std_ = torch.tensor(_IMAGENET_STD, device=device).reshape(1, 3, 1, 1)
    for name in backbones:
        backbone = build_backbone(name).to(device).eval()
        rows = []
        for image in images.split(8):
            x = F.interpolate(image.float(), size=(BACKBONE_RESOLUTION, BACKBONE_RESOLUTION),
                              mode="bilinear", align_corners=False, antialias=True)
            features = backbone((x - mean_) / std_).float()
            features = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])
            for f in features:
                pick = torch.randint(f.shape[0], (min(samples_per_image, f.shape[0]),), generator=generator)
                rows.append(f[pick.to(device)])
        rows = torch.cat(rows).double()
        mean = rows.mean(0)
        covariance = torch.cov((rows - mean).T)
        values, vectors = torch.linalg.eigh(covariance)
        values, vectors = values.flip(0)[:COMPONENTS], vectors.flip(1)[:, :COMPONENTS]
        out[name] = {
            "mean": mean.float().cpu(),
            "projection": (vectors / values.clamp_min(1e-12).sqrt()).float().cpu(),
            "explained": (values / torch.diagonal(covariance).sum()).float().cpu(),
        }
        del backbone
    return out
