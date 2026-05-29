"""White-box PGD attack on the multi-order motion model.

WHY THIS EXISTS
  Goal: test whether an adversarial attack on this model of human motion processing
  ALSO fools humans. If a perturbation that breaks the model's motion percept also
  breaks a person's, that is evidence the model and the human visual system share the
  underlying computation. So the perturbation must be something humans can actually
  perceive as motion -- not invisible pixel noise -- which drives the design choices
  below (coherent drift, bandwidth limiting, EOT).

THE TARGET
  FFV1DNNV2, a biologically-inspired optical-flow net: a first-order motion-energy
  front end (V1-like, luminance-defined motion), a recurrent graph network (MT-like
  integration), and a second-order pathway (contrast/texture-defined motion). It is
  meant to mirror human motion perception, which is what makes it worth attacking.

THE MASK
  We corrupt motion only inside a target region. The mask is the model's OWN MaskCut
  segmentation of the salient moving region (from its attention map). It restricts the
  LOSS (which output region we want wrong), NOT where the perturbation may live --
  delta can be placed anywhere, including outside the mask, to corrupt the masked flow.

KNOWN OBSTACLES TO HUMAN TRANSFER
  - Transparency / common fate: a coherent overlay moving differently from the scene
    tends to be scissioned off as a separate transparent surface, so it may not alter
    the scene's PERCEIVED motion even when it fools the model.
  - Depth: the dense warp inherits parallax/looming structure from the flow field but
    does NOT model occlusion/disparity, so it can be depth-inconsistent (smears at
    depth edges), which worsens the transparency problem.
  - Nulling vs reversal: making humans see NO motion (objective=zero) is likely easier
    to induce than REVERSED motion (objective=counter) -- balanced opposing motion
    collapsing to "no motion" is a well-established human effect (motion nulling).

CONTROL
  A matched-norm random-noise condition is always run, to confirm the model is not
  merely fragile to any noise (it is not) -- the adversarial effect must exceed it.

Perturbation parameterizations:
  - free  : independent per-frame delta (max model attackability, but incoherent
            per-frame hash that humans scission off / don't perceive as motion).
  - global: a single texture rigidly drifting opposite the mean masked flow
            (coherent, human-visible; correct only for translational scene flow).
  - dense : a single texture advected each frame along the REVERSE of the scene's
            per-pixel flow via compositional grid_sample warps. Handles translation,
            rotation, divergence and any mix -- it replays the scene's motion backward,
            so the perturbation locally counters the perceived flow everywhere.

Bandwidth limiting (smooth_factor): delta parameterized at H/k x W/k, upsampled, so
the perturbation lives at a human-visible spatial scale (not pixel hash).
EOT (n_eot): gradients averaged over small random augmentations for robustness.
Both pathways in the loss: the model detaches the first-order branch internally, so
flow_seq (combined) AND flow_seq_1 (first-order) must both appear to get gradient
through both. First-order is the human-relevant channel.

Objective:
  zero    - minimize ||flow * mask||^2          (model sees no motion / nulling)
  counter - minimize ||(flow + clean_flow) * mask||^2  (model sees reversed motion)

Outputs to --save_dir:
  clean.png / attacked.png / noise.png    - flow quiver overlays (one per condition)
  clean_video.gif / attacked_video.gif / noise_video.gif - videos shown to the model
  perturbation.gif                         - delta amplified for visibility
  mask.png, metrics.json
"""
from __future__ import print_function, division
import argparse
import os
import json
import glob
import copy
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from io import BytesIO
import matplotlib.pyplot as plt

from model.nmi6.FFV1MT_MS import FFV1DNNV2
from torch.cuda.amp import autocast as autocast
import types
from utils.flow_utils import flow_to_image_relative
from mask_cut import maskcut
try:
    from mask_cut.crf import densecrf
    HAS_CRF = True
except Exception:
    HAS_CRF = False

DEVICE = "cuda"
flow_to_image = flow_to_image_relative


# --------------------------------------------------------------------------------------
# Image / IO helpers
# --------------------------------------------------------------------------------------
def load_image(imfile):
    img = Image.open(imfile)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).float()[None].to(DEVICE)


def load_clip(path, n=15, max_side=None):
    images = sorted(
        glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg"))
    )
    if not images:
        raise FileNotFoundError(f"No PNG/JPG frames found in {path}")
    try:
        images.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
    except Exception:
        images.sort()
    if len(images) < n:
        raise ValueError(f"Need at least {n} frames, found {len(images)} in {path}")
    # take a centered window of n frames
    start = (len(images) - n) // 2
    images = images[start : start + n]
    print(f"Loaded {len(images)} frames from {path}")
    frames = [load_image(f) for f in images]  # list of [1, 3, H, W]
    H, W = frames[0].shape[2:]
    if max_side is not None and max(H, W) > max_side:
        scale = max_side / max(H, W)
        H, W = int(H * scale), int(W * scale)
    H8, W8 = int(np.round(H / 8)) * 8, int(np.round(W / 8)) * 8
    frames = [F.interpolate(f, size=(H8, W8), mode="bicubic", align_corners=True).clamp_(0, 255) for f in frames]
    return frames  # list of [1, 3, H8, W8]


def to_numpy_uint8(frame_tensor):
    """[1,3,H,W] 0-255 float -> [H,W,3] uint8."""
    return frame_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)


def flow_to_rgb(flow_tensor):
    """[1,2,H,W] -> [H,W,3] uint8 visualization."""
    flow_np = flow_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return flow_to_image(flow_np)


def quiver_overlay(image_uint8, flow_tensor, spacing=30, title=""):
    """Render a quiver overlay on the image, return as RGB array."""
    flow = flow_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    H, W = flow.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * H / W), dpi=120)
    ax.imshow(image_uint8)
    xs = np.arange(spacing // 2, W, spacing)
    ys = np.arange(spacing // 2, H, spacing)
    X, Y = np.meshgrid(xs, ys)
    U = flow[ys[:, None], xs[None, :], 0]
    V = -flow[ys[:, None], xs[None, :], 1]
    ax.quiver(X, Y, U, V, color="red", scale=50, width=0.005, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def save_gif(frames_uint8, out_path, duration=0.1):
    imageio.mimsave(out_path, frames_uint8, "GIF", duration=duration, loop=0, palettesize=256)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def data_preprocess_with_grad(image_list):
    """Replacement for FFV1DNNV2.data_preprocess that preserves the autograd graph.

    Original has @torch.no_grad() and uses copy.deepcopy, both of which sever the
    gradient path from model input back to delta. This version is mathematically
    identical (same 0-1 scaling, RGB->grayscale conversion) but pure-functional and
    autograd-friendly."""
    if image_list[0].max() > 10:
        image_list = [img / 255.0 for img in image_list]
    if image_list[0].shape[1] == 1:
        image_list_rgb = [img.repeat(1, 3, 1, 1) for img in image_list]
    else:
        image_list_rgb = list(image_list)  # new list, same tensor refs (keeps grad)
    if image_list[0].shape[1] == 3:
        image_list = [
            img[:, 0, :, :] * 0.299 + img[:, 1, :, :] * 0.587 + img[:, 2, :, :] * 0.114
            for img in image_list
        ]
        image_list = [img.unsqueeze(1) for img in image_list]
    return image_list, image_list_rgb


def forward_for_attack(self, image_list, mix_enable=True, layer=None):
    """Replacement for FFV1DNNV2.forward that preserves gradient through both pathways.

    The original detaches st_component1 before the fuse layer (line 202 in FFV1MT_MS.py).
    During training this prevents the combined pathway's loss from updating V1 weights.
    For our attack (frozen weights), it only severs the gradient from flow_seq back through
    V1, meaning the combined loss can only attack the second-order (ffv2) pathway.
    Removing the detach lets flow_seq gradient reach ffv1 as well, so both loss terms
    attack both pathways:
      flow_seq_1 loss -> V1 gradient                   (as before)
      flow_seq   loss -> ffv2 gradient + V1 gradient   (new: V1 now reachable)
    """
    if layer is not None:
        self.MT.num_layers = layer
        self.num_layers = layer
    results_dict = {}
    image_list, image_list_rgb = self.data_preprocess(image_list)
    image_list = image_list[2:-2]
    assert len(image_list) == 11 and len(image_list_rgb) == 15
    B, _, H, W = image_list[0].shape
    with autocast(enabled=mix_enable):
        st_component1 = self.ffv1(image_list)
        st_component2 = self.ffv2(image_list_rgb)
        flow_0 = [self.decoder(st_component1)]
        value1, attn1 = self.MT.forward_save_mem(st_component1)
        flow_1 = [self.decoder(feature) for feature in value1]
        flow_1_bi = [self.upsample_flow(flows, feature=None, bilinear=True, upsample_factor=8)
                     for flows in flow_0 + flow_1]
        flow_1_up = [self.upsample_flow(flows, upsampler=self.upsampler, feature=attn, upsample_factor=8)
                     for flows, attn in zip(flow_1, attn1)]
        # No detach: gradient from flow_seq now flows through st_component1 -> ffv1 -> delta.
        st_component = self.fuse(torch.cat([st_component1, st_component2], dim=1)).square()
        value_all, attn = self.MT.forward_save_mem(st_component)
        flows_all = [self.decoder(feature) for feature in value_all]
        flows_up = [self.upsample_flow(flows, upsampler=self.upsampler, feature=attn, upsample_factor=8)
                    for flows, attn in zip(flows_all, attn)]
        flow_bi = [self.upsample_flow(flows, feature=None, bilinear=True, upsample_factor=8)
                   for flows in flows_all]
        results_dict["flow_seq"] = flows_up
        results_dict["flow_seq_bi"] = flow_bi
        results_dict["flow_seq_1_bi"] = flow_1_bi
        results_dict["flow_seq_1"] = flow_1_up
        results_dict["flow_attn"] = attn
        results_dict["flow_attn_1"] = attn1
        if layer == 0:
            results_dict["flow_seq"] = flow_1_bi
            results_dict["flow_seq_1"] = flow_1_bi
    return results_dict


def build_model(ckpt_path):
    model = FFV1DNNV2(num_scales=8, upsample_factor=8, scale_factor=16, num_layers=6)
    model = nn.DataParallel(model)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.module.cuda().eval()
    # Override the no-grad preprocess so adversarial gradient can flow to delta.
    model.data_preprocess = data_preprocess_with_grad
    # Override forward to remove the st_component1 detach, so flow_seq gradient
    # reaches both the V1 (ffv1) and second-order (ffv2) pathways.
    model.forward = types.MethodType(forward_for_attack, model)
    return model


def run_model(model, frames, iters=8, requires_grad=False):
    """frames: list of [1,3,H,W] 0-255 tensors. Returns the results_dict."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        return model.forward(frames, mix_enable=False, layer=iters)


# --------------------------------------------------------------------------------------
# Mask extraction (repo's maskcut_from_motion + optional CRF refinement)
# --------------------------------------------------------------------------------------
def extract_mask(model, frames, iters, image_uint8, tau=0.15, use_crf=True):
    out = run_model(model, frames, iters=iters, requires_grad=False)
    attention = out["flow_attn"][-1]  # combined-pathway attention
    bipartitions, _ = maskcut.maskcut_from_motion(attention, patch_size=8, tau=tau, N=1)
    mask = bipartitions[0].astype(np.float32)  # [h, w]
    if use_crf and HAS_CRF:
        try:
            refined = densecrf(image_uint8, mask)
            mask = (refined >= 0.5).astype(np.float32)
        except Exception as e:
            print(f"[mask] CRF refinement failed ({e}); falling back to raw MaskCut output.")
    return mask  # numpy [H, W] in {0, 1}


# --------------------------------------------------------------------------------------
# PGD attack
# --------------------------------------------------------------------------------------
def apply_eot_aug(perturbed, jitter_px=2, brightness_jitter=0.03, noise_std=0.5):
    """Small input augmentation for EOT. perturbed: [T, 3, H, W] in 0-255 space."""
    dx = random.randint(-jitter_px, jitter_px)
    dy = random.randint(-jitter_px, jitter_px)
    b = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
    aug = perturbed * b
    if noise_std > 0:
        aug = aug + torch.randn_like(aug) * noise_std
    if dx != 0 or dy != 0:
        aug = torch.roll(aug, shifts=(dy, dx), dims=(-2, -1))
    return aug.clamp(0.0, 255.0)


def pgd_attack(model, frames, mask_t, eps, steps, alpha, iters,
               smooth_factor=1, n_eot=1,
               clean_flow=None, clean_flow_1=None, objective="zero",
               drift_vec=None, warp_flow=None):
    """PGD with bandwidth limiting, EOT, configurable objective and parameterization.

    drift_vec set  -> global mode (single texture, rigid uniform drift).
    warp_flow set  -> dense mode (single texture advected along reversed per-pixel flow).
    neither        -> free mode (independent per-frame delta).
    """
    T = len(frames)
    _, _, H, W = frames[0].shape
    frames_stack = torch.cat(frames, dim=0)
    mid = T // 2

    if objective == "counter":
        assert clean_flow is not None and clean_flow_1 is not None, \
            "counter objective requires the clean flow as target"
        target_flow = clean_flow.detach()
        target_flow_1 = clean_flow_1.detach()

    Hs = max(1, H // smooth_factor)
    Ws = max(1, W // smooth_factor)
    if warp_flow is not None:
        mode = "dense"
    elif drift_vec is not None:
        mode = "global"
    else:
        mode = "free"
    single_texture = mode in ("global", "dense")
    print(f"[attack] delta parameterized at {Hs}x{Ws}, upsampled {smooth_factor}x to {H}x{W}")
    print(f"[attack] objective={objective}  n_eot={n_eot}  drift_mode={mode}  global_vec={drift_vec}")

    n_param = 1 if single_texture else T
    delta_lr = torch.empty(n_param, 3, Hs, Ws, device=DEVICE).uniform_(-eps, eps)
    delta_lr.requires_grad_(True)

    # Precompute the two sampling grids for the dense compositional warp.
    if mode == "dense":
        ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing="ij")
        base_xy = torch.stack([xs, ys], dim=-1).float()  # [H, W, 2] in (x, y) pixels
        f = warp_flow.detach().permute(0, 2, 3, 1)        # [1, H, W, 2]

        def _grid(sign):
            coords = base_xy[None] + sign * f             # [1, H, W, 2]
            cx = coords[..., 0] % W                        # circular wrap (infinite scrolling texture)
            cy = coords[..., 1] % H
            gx = 2.0 * cx / (W - 1) - 1.0
            gy = 2.0 * cy / (H - 1) - 1.0
            return torch.stack([gx, gy], dim=-1)           # [1, H, W, 2]

        grid_plus = _grid(+1.0)   # sample at x+flow  -> advances perturbation by -flow (forward in time)
        grid_minus = _grid(-1.0)  # sample at x-flow  -> advances perturbation by +flow (backward in time)

    def make_perturbation(d):
        base = d if smooth_factor == 1 else F.interpolate(d, size=(H, W), mode="bilinear", align_corners=True)
        if mode == "free":
            return base  # [T, 3, H, W]
        if mode == "global":
            vx, vy = drift_vec
            out_frames = []
            for t in range(T):
                dx = int(round(vx * (t - mid)))
                dy = int(round(vy * (t - mid)))
                out_frames.append(torch.roll(base, shifts=(dy, dx), dims=(-2, -1)))  # base is [1,3,H,W]
            return torch.cat(out_frames, dim=0)
        # dense: replay the scene flow backward by composing single-frame warps
        out_frames = [None] * T
        out_frames[mid] = base  # [1, 3, H, W]
        for t in range(mid + 1, T):
            out_frames[t] = F.grid_sample(out_frames[t - 1], grid_plus, mode="bilinear",
                                          padding_mode="border", align_corners=True)
        for t in range(mid - 1, -1, -1):
            out_frames[t] = F.grid_sample(out_frames[t + 1], grid_minus, mode="bilinear",
                                          padding_mode="border", align_corners=True)
        return torch.cat(out_frames, dim=0)  # [T, 3, H, W]

    def compute_loss(out):
        flow = out["flow_seq"][-1]
        flow_1 = out["flow_seq_1"][-1]
        if objective == "zero":
            return ((flow * mask_t) ** 2).mean() + ((flow_1 * mask_t) ** 2).mean(), flow, flow_1
        else:  # counter
            return (((flow + target_flow) * mask_t) ** 2).mean() + \
                   (((flow_1 + target_flow_1) * mask_t) ** 2).mean(), flow, flow_1

    for step in range(steps):
        grad_accum = torch.zeros_like(delta_lr)
        last_flow, last_flow_1, last_loss = None, None, None

        for eot_i in range(n_eot):
            delta = make_perturbation(delta_lr)
            perturbed = (frames_stack + delta).clamp(0.0, 255.0)
            if eot_i > 0:
                perturbed = apply_eot_aug(perturbed)
            perturbed_list = [perturbed[i : i + 1] for i in range(T)]

            out = model.forward(perturbed_list, mix_enable=False, layer=iters)
            loss, flow, flow_1 = compute_loss(out)
            grad = torch.autograd.grad(loss, delta_lr)[0]
            grad_accum = grad_accum + grad

            if eot_i == 0:
                last_flow, last_flow_1, last_loss = flow.detach(), flow_1.detach(), loss.item()

        grad = grad_accum / n_eot

        with torch.no_grad():
            delta_lr = delta_lr - alpha * grad.sign()
            delta_lr = torch.clamp(delta_lr, -eps, eps)
        delta_lr.requires_grad_(True)

        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():
                flow_mag = (last_flow * mask_t).pow(2).mean().sqrt().item()
                flow_1_mag = (last_flow_1 * mask_t).pow(2).mean().sqrt().item()
            print(f"  step {step:3d}  loss={last_loss:.4f}  "
                  f"masked_rms[flow]={flow_mag:.3f}  masked_rms[flow_1]={flow_1_mag:.3f}")
        # if last_loss < 1e-4:
        #    print(f"  [early stop] loss={last_loss:.2e} < 1e-4 at step {step}")
        #    break

    delta_lr = delta_lr.detach()
    delta_final = make_perturbation(delta_lr)
    perturbed_list = [(frames_stack[i : i + 1] + delta_final[i : i + 1]).clamp(0, 255) for i in range(T)]
    return perturbed_list, delta_final


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def masked_flow_magnitudes(model, frames, mask_t, iters):
    out = run_model(model, frames, iters=iters, requires_grad=False)
    flow = out["flow_seq"][-1]
    flow_1 = out["flow_seq_1"][-1]
    return {
        "rms_flow":   (flow * mask_t).pow(2).mean().sqrt().item(),
        "rms_flow_1": (flow_1 * mask_t).pow(2).mean().sqrt().item(),
        "mean_flow_mag":   (flow.norm(dim=1, keepdim=True) * mask_t).mean().item(),
        "mean_flow_mag_1": (flow_1.norm(dim=1, keepdim=True) * mask_t).mean().item(),
    }, out


def directional_alignment(flow, ref_flow, mask_t):
    """Mean cosine similarity between flow and ref_flow over the masked region.
      +1 = same direction as clean   0 = orthogonal / no motion   -1 = exactly opposite.
    Only counts pixels where both vectors have non-trivial magnitude."""
    dot = (flow * ref_flow).sum(dim=1, keepdim=True)
    norm = flow.norm(dim=1, keepdim=True) * ref_flow.norm(dim=1, keepdim=True)
    cos = dot / (norm + 1e-6)
    valid = (mask_t > 0.5) & (ref_flow.norm(dim=1, keepdim=True) > 0.5)
    if valid.sum() == 0:
        return float("nan")
    return cos[valid].mean().item()


# --------------------------------------------------------------------------------------
# Target weight maps (what region of the flow we try to corrupt)
# --------------------------------------------------------------------------------------
def periphery_weight(H, W, inner_frac=0.33, outer_frac=0.95):
    """Radial weight in [0,1]: ~0 in the central ellipse (r < inner_frac), ramping
    smoothly to 1 by r = outer_frac, where r is normalized so 1.0 is the edge midpoint.
    Targets the peripheral visual field, which dominates self-motion / vection."""
    ys = (torch.arange(H, device=DEVICE).float() - (H - 1) / 2) / ((H - 1) / 2)
    xs = (torch.arange(W, device=DEVICE).float() - (W - 1) / 2) / ((W - 1) / 2)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2)
    w = ((r - inner_frac) / max(1e-6, (outer_frac - inner_frac))).clamp(0.0, 1.0)
    return w[None, None]  # [1, 1, H, W]


def lowpass_flow_weight(clean_flow, sigma_frac=0.1):
    """Low-pass the clean flow magnitude to isolate the smooth, large-field component
    (the globally-coherent / self-motion-consistent flow that drives vection; localized
    object motion is blurred away). Returns [1,1,H,W] normalized to [0,1]."""
    mag = clean_flow.norm(dim=1, keepdim=True)  # [1,1,H,W]
    H, W = mag.shape[-2:]
    sigma = max(1.0, sigma_frac * min(H, W))
    k = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(k, device=mag.device).float() - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    blurred = F.conv2d(mag, g.view(1, 1, 1, k), padding=(0, k // 2))
    blurred = F.conv2d(blurred, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return blurred / (blurred.max() + 1e-6)


def build_target_mask(mode, clean_out, mid_img, H, W, args):
    """Build the soft target-weight map [1,1,H,W] that weights the loss.
      maskcut    - the model's MaskCut object/region segmentation (binary).
      global     - whole frame.
      periphery  - radial periphery weight (vection-dominant field; spares the fovea).
      selfmotion - periphery x low-pass(clean flow): globally-coherent peripheral flow.
    """
    if mode == "maskcut":
        attention = clean_out["flow_attn"][-1]
        bip, _ = maskcut.maskcut_from_motion(attention, patch_size=8, tau=args.mask_tau, N=1)
        mask = bip[0].astype(np.float32)
        if not args.no_crf and HAS_CRF:
            try:
                refined = densecrf(mid_img, mask)
                mask = (refined >= 0.5).astype(np.float32)
            except Exception as e:
                print(f"[mask] CRF refinement failed ({e}); using raw MaskCut output.")
        mask_t = torch.from_numpy(mask).to(DEVICE)[None, None].float()
        if mask_t.shape[-2:] != (H, W):
            mask_t = F.interpolate(mask_t, size=(H, W), mode="nearest")
        return mask_t
    if mode == "global":
        return torch.ones(1, 1, H, W, device=DEVICE)
    if mode == "periphery":
        return periphery_weight(H, W, args.periphery_inner)
    if mode == "selfmotion":
        return periphery_weight(H, W, args.periphery_inner) * lowpass_flow_weight(clean_out["flow_seq"][-1], args.lowpass_sigma)
    raise ValueError(f"unknown mask mode: {mode}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="modelckpt/final_sintel_kitti.pth")
    ap.add_argument("--path", required=True, help="folder of frames")
    ap.add_argument("--save_dir", default="result_attack")
    ap.add_argument("--eps", type=float, default=13.0, help="L_inf budget in 0-255 pixel space (13 = ~5%)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=None, help="PGD step size (default eps/10)")
    ap.add_argument("--iters", type=int, default=6, help="model recurrent iterations; lower to save memory")
    ap.add_argument("--mask-tau", type=float, default=0.15, help="MaskCut threshold")
    ap.add_argument("--no-crf", action="store_true", help="skip DenseCRF refinement of the mask")
    ap.add_argument("--smooth-factor", type=int, default=4,
                    help="parameterize delta at H/k x W/k, upsample to (H,W); k=4-8 forces a human-visible spatial scale")
    ap.add_argument("--max-side", type=int, default=480.0,
                    help="downscale input so longer side is <= this (helps with GPU memory)")
    ap.add_argument("--n-eot", type=int, default=3,
                    help="EOT samples per PGD step (>=2 averages gradients over random augmentations; 1 disables EOT)")
    ap.add_argument("--objective", choices=["zero", "counter"], default="counter",
                    help="zero: minimize ||flow||^2 (see no motion); counter: push toward OPPOSITE of clean flow")
    ap.add_argument("--drift", action="store_true",
                    help="constrain the perturbation to a single coherent moving texture (human-visible) instead of free per-frame noise")
    ap.add_argument("--drift-mode", choices=["global", "dense"], default="dense",
                    help="global: rigid uniform drift opposite the mean masked flow (translational only). dense: advect along reverse of per-pixel scene flow (handles rotation/divergence).")
    ap.add_argument("--drift-scale", type=float, default=1.0,
                    help="[global mode] multiplier on the drift speed (1.0 = match the mean masked flow speed)")
    ap.add_argument("--drift-vx", type=float, default=None, help="[global mode] override drift velocity x (px/frame)")
    ap.add_argument("--drift-vy", type=float, default=None, help="[global mode] override drift velocity y (px/frame)")
    ap.add_argument("--mask-mode", choices=["maskcut", "global", "periphery", "selfmotion"], default="maskcut",
                    help="maskcut: model's object/region cut. global: whole frame. periphery: radial periphery (vection-dominant). selfmotion: periphery x low-pass(flow) = globally-coherent peripheral flow.")
    ap.add_argument("--periphery-inner", type=float, default=0.33,
                    help="[periphery/selfmotion] normalized radius (0-1) of the central field left untouched")
    ap.add_argument("--lowpass-sigma", type=float, default=0.1,
                    help="[selfmotion] Gaussian sigma (fraction of min(H,W)) for isolating large-field flow")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    alpha = args.alpha if args.alpha is not None else args.eps / 10.0

    print(f"[setup] eps={args.eps}/255  alpha={alpha:.3f}  steps={args.steps}  iters={args.iters}")
    print(f"[setup] pydensecrf available: {HAS_CRF}")

    model = build_model(args.model)
    frames = load_clip(args.path, n=15, max_side=args.max_side)
    T, _, _, H, W = (len(frames),) + tuple(frames[0].shape)
    print(f"[setup] T={T}  H={H}  W={W}")

    # === Clean forward pass: flow (both pathways) + attention (for maskcut mode) ===
    mid_img = to_numpy_uint8(frames[len(frames) // 2])
    clean_out = run_model(model, frames, iters=args.iters, requires_grad=False)
    clean_flow_final = clean_out["flow_seq"][-1].detach().clone()
    clean_flow_1_final = clean_out["flow_seq_1"][-1].detach().clone()

    # === Build the target weight map per --mask-mode ===
    mask_t = build_target_mask(args.mask_mode, clean_out, mid_img, H, W, args).float()
    del clean_out
    torch.cuda.empty_cache()
    mask_coverage = float(mask_t.mean().item())
    wt_np = mask_t.squeeze().detach().cpu().numpy()
    Image.fromarray((wt_np / (wt_np.max() + 1e-6) * 255).astype(np.uint8)).save(os.path.join(args.save_dir, "mask.png"))
    print(f"[mask] mode={args.mask_mode}  mean weight={mask_coverage:.3f}")
    if (mask_t > 0.01).float().mean().item() < 0.001:
        print("[mask] WARNING: target weight is essentially empty.")
        return

    # === Clean baseline metrics over the weighted region (reuse the forward pass above) ===
    clean_metrics = {
        "rms_flow":   (clean_flow_final * mask_t).pow(2).mean().sqrt().item(),
        "rms_flow_1": (clean_flow_1_final * mask_t).pow(2).mean().sqrt().item(),
        "mean_flow_mag":   (clean_flow_final.norm(dim=1, keepdim=True) * mask_t).mean().item(),
        "mean_flow_mag_1": (clean_flow_1_final.norm(dim=1, keepdim=True) * mask_t).mean().item(),
    }
    print(f"[clean] {clean_metrics}")

    # === Determine perturbation parameterization (drift mode only) ===
    drift_vec = None
    warp_flow = None
    if args.drift and args.drift_mode == "dense":
        warp_flow = clean_flow_final
        m = mask_t > 0.5
        mvx = clean_flow_final[:, 0:1][m].mean().item()
        mvy = clean_flow_final[:, 1:2][m].mean().item()
        print(f"[drift] DENSE mode: advecting texture along reverse of per-pixel scene flow")
        print(f"[drift] mean masked flow=({mvx:.2f},{mvy:.2f}) px/frame (field is per-pixel, not just this mean)")
    elif args.drift and args.drift_mode == "global":
        if args.drift_vx is not None and args.drift_vy is not None:
            drift_vec = (args.drift_vx, args.drift_vy)
            print(f"[drift] GLOBAL mode: manual drift velocity {drift_vec} px/frame")
        else:
            m = mask_t > 0.5
            mean_vx = clean_flow_final[:, 0:1][m].mean().item()
            mean_vy = clean_flow_final[:, 1:2][m].mean().item()
            drift_vec = (-mean_vx * args.drift_scale, -mean_vy * args.drift_scale)
            print(f"[drift] GLOBAL mode: mean masked flow=({mean_vx:.2f},{mean_vy:.2f}) px/frame")
            print(f"[drift] -> drift velocity ({drift_vec[0]:.2f},{drift_vec[1]:.2f}) px/frame (opposite, scale={args.drift_scale})")

    # === PGD attack ===
    print(f"[attack] running PGD ({args.steps} steps, smooth_factor={args.smooth_factor}, n_eot={args.n_eot}, objective={args.objective})")
    perturbed_frames, delta = pgd_attack(model, frames, mask_t,
                                         eps=args.eps, steps=args.steps,
                                         alpha=alpha, iters=args.iters,
                                         smooth_factor=args.smooth_factor,
                                         n_eot=args.n_eot,
                                         clean_flow=clean_flow_final,
                                         clean_flow_1=clean_flow_1_final,
                                         objective=args.objective,
                                         drift_vec=drift_vec,
                                         warp_flow=warp_flow)
    torch.cuda.empty_cache()
    attack_metrics, attack_out = masked_flow_magnitudes(model, perturbed_frames, mask_t, args.iters)
    attack_flow_final = attack_out["flow_seq"][-1].detach().clone()
    attack_flow_1_final = attack_out["flow_seq_1"][-1].detach().clone()
    del attack_out
    torch.cuda.empty_cache()
    print(f"[attack] {attack_metrics}")

    # === Matched random noise control ===
    frames_stack = torch.cat(frames, dim=0)
    rand_delta = torch.empty_like(frames_stack).uniform_(-args.eps, args.eps)
    rand_perturbed = (frames_stack + rand_delta).clamp(0, 255)
    rand_frames = [rand_perturbed[i : i + 1] for i in range(T)]
    noise_metrics, noise_out = masked_flow_magnitudes(model, rand_frames, mask_t, args.iters)
    noise_flow_final = noise_out["flow_seq"][-1].detach().clone()
    noise_flow_1_final = noise_out["flow_seq_1"][-1].detach().clone()
    del noise_out
    torch.cuda.empty_cache()
    print(f"[noise] {noise_metrics}")

    # === Directional alignment vs clean, per pathway (combined readout AND first-order) ===
    align = {
        "attack_vs_clean":        directional_alignment(attack_flow_final, clean_flow_final, mask_t),
        "noise_vs_clean":         directional_alignment(noise_flow_final, clean_flow_final, mask_t),
        "attack_vs_clean_first":  directional_alignment(attack_flow_1_final, clean_flow_1_final, mask_t),
        "noise_vs_clean_first":   directional_alignment(noise_flow_1_final, clean_flow_1_final, mask_t),
    }
    print(f"[direction] combined: cosine(attacked,clean)={align['attack_vs_clean']:.3f}  noise={align['noise_vs_clean']:.3f}")
    print(f"[direction] first-order: cosine(attacked,clean)={align['attack_vs_clean_first']:.3f}  noise={align['noise_vs_clean_first']:.3f}")
    print(f"[direction] (want both attacked values near -1; first-order is the human-relevant channel)")

    # === Save artifacts ===
    metrics = {
        "eps_pixel": args.eps,
        "eps_pct_of_255": 100 * args.eps / 255,
        "steps": args.steps,
        "alpha": alpha,
        "smooth_factor": args.smooth_factor,
        "n_eot": args.n_eot,
        "objective": args.objective,
        "drift_enabled": bool(args.drift),
        "drift_mode": (args.drift_mode if args.drift else None),
        "drift_vec": list(drift_vec) if drift_vec is not None else None,
        "mask_target_mode": args.mask_mode,
        "mask_mean_weight": mask_coverage,
        "clean": clean_metrics,
        "attack": attack_metrics,
        "random_noise": noise_metrics,
        "directional_alignment_vs_clean": align,
        "delta_actual_max_abs": float(delta.abs().max().item()),
        "delta_actual_mean_abs": float(delta.abs().mean().item()),
    }
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Flow quivers: ONE PNG per condition (model produces one flow per 15-frame window).
    print(f"[save] writing visualizations to {args.save_dir}")
    mid = T // 2
    Image.fromarray(quiver_overlay(to_numpy_uint8(frames[mid]), clean_flow_final,
                                   title=f"CLEAN  rms={clean_metrics['rms_flow']:.3f}")).save(
        os.path.join(args.save_dir, "clean.png"))
    Image.fromarray(quiver_overlay(to_numpy_uint8(perturbed_frames[mid]), attack_flow_final,
                                   title=f"ATTACKED  rms={attack_metrics['rms_flow']:.3f}")).save(
        os.path.join(args.save_dir, "attacked.png"))
    Image.fromarray(quiver_overlay(to_numpy_uint8(rand_frames[mid]), noise_flow_final,
                                   title=f"NOISE  rms={noise_metrics['rms_flow']:.3f}")).save(
        os.path.join(args.save_dir, "noise.png"))

    # Input frames as separate animations (no overlay).
    save_gif([to_numpy_uint8(f) for f in frames],
             os.path.join(args.save_dir, "clean_video.gif"))
    save_gif([to_numpy_uint8(f) for f in perturbed_frames],
             os.path.join(args.save_dir, "attacked_video.gif"))
    save_gif([to_numpy_uint8(f) for f in rand_frames],
             os.path.join(args.save_dir, "noise_video.gif"))

    # Perturbation alone, amplified for visibility.
    amp = 255.0 / (2 * args.eps)
    pert_uint8 = ((delta * amp) + 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    pert_uint8 = pert_uint8.transpose(0, 2, 3, 1)  # [T, H, W, 3]
    save_gif([pert_uint8[i] for i in range(T)], os.path.join(args.save_dir, "perturbation.gif"))

    print("[done]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
