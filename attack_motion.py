"""White-box PGD attack on the multi-order motion model.

v1 design (see chat for rationale):
  - Per-frame perturbation delta of shape [T, 3, H, W], L_inf budget eps = 13/255
    (i.e. ~5% of full-scale RGB). Perturbation may be added anywhere on the frame.
  - Loss: minimize ||flow_seq * mask||^2 + ||flow_seq_1 * mask||^2 inside a moving-object
    mask. mask comes from the repo's maskcut_from_motion using the model's own attention,
    optionally refined by DenseCRF.
  - Both pathways in the loss because st_component1 is detached internally, so the
    combined-output gradient does NOT flow through the first-order branch.
  - 50 PGD steps, signed-gradient step alpha = eps/10, one random init.
  - No EOT / no TV. Treat this as a pipeline test, not a result fit for humans.

Outputs to --save_dir:
  - clean.gif, attacked.gif, noise.gif: flow visualizations
  - perturbation.gif: delta amplified for visibility
  - perturbed_video.gif: the attacker's input (x + delta) shown to the model
  - mask.png: the optimization mask
  - metrics.json: numerical comparison (clean vs attack vs random noise)
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
# Image / IO helpers (mirrors infer_motion.py so a zero-delta run reproduces its output)
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


def build_model(ckpt_path):
    model = FFV1DNNV2(num_scales=8, upsample_factor=8, scale_factor=16, num_layers=6)
    model = nn.DataParallel(model)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.module.cuda().eval()
    # Override the no-grad preprocess so adversarial gradient can flow to delta.
    # Instance-level attribute shadows the class method; since the original method
    # doesn't actually use `self`, no rebinding needed.
    model.data_preprocess = data_preprocess_with_grad
    return model


def run_model(model, frames, iters=8, requires_grad=False):
    """frames: list of [1,3,H,W] 0-255 tensors. Returns the results_dict."""
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        # mix_enable=False keeps everything in fp32, simpler gradient story
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
    """Small input augmentation for EOT. perturbed: [T, 3, H, W] in 0-255 space.

    - Random integer translation by up to +/- jitter_px (same shift for all frames,
      preserves the underlying motion structure).
    - Multiplicative brightness in (1 +/- brightness_jitter).
    - Additive Gaussian noise with std noise_std (in pixel units).

    All small enough that motion content is preserved; large enough that pixel-level
    overfitting in delta gets penalized."""
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
               clean_flow=None, clean_flow_1=None, objective="zero"):
    """PGD with bandwidth limiting (smooth_factor), EOT, and configurable objective.

    Bandwidth limiting: delta is parameterized at H/k x W/k and bilinearly upsampled
    so the perturbation lives at a spatial scale humans process.

    EOT: at each PGD step, average the gradient over `n_eot` random augmentations
    (translation/brightness/noise) of the perturbed input. Prevents the optimizer
    from finding brittle pixel-level patterns that only fool this model.

    Objective:
        "zero"    - minimize ||flow * mask||^2          (model sees no motion)
        "counter" - minimize ||(flow + clean_flow) * mask||^2
                    (push model toward seeing motion in the OPPOSITE direction;
                    encourages the perturbation to itself carry counter-motion
                    energy, which is what humans would perceive)."""
    T = len(frames)
    _, _, H, W = frames[0].shape
    frames_stack = torch.cat(frames, dim=0)

    if objective == "counter":
        assert clean_flow is not None and clean_flow_1 is not None, \
            "counter objective requires the clean flow as target"
        target_flow = clean_flow.detach()
        target_flow_1 = clean_flow_1.detach()

    Hs = max(1, H // smooth_factor)
    Ws = max(1, W // smooth_factor)
    print(f"[attack] delta parameterized at {Hs}x{Ws}, upsampled {smooth_factor}x to {H}x{W}")
    print(f"[attack] objective={objective}  n_eot={n_eot}")

    delta_lr = torch.empty(T, 3, Hs, Ws, device=DEVICE).uniform_(-eps, eps)
    delta_lr.requires_grad_(True)

    def upsample(d):
        if smooth_factor == 1:
            return d
        return F.interpolate(d, size=(H, W), mode="bilinear", align_corners=True)

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
            delta = upsample(delta_lr)
            perturbed = (frames_stack + delta).clamp(0.0, 255.0)
            # First sample uses unaugmented input (so step-end metrics reflect the
            # true loss on the actual perturbed video); subsequent samples are
            # randomly augmented to make the gradient EOT-style.
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

    delta_lr = delta_lr.detach()
    delta_final = upsample(delta_lr)
    perturbed_list = [(frames_stack[i : i + 1] + delta_final[i : i + 1]).clamp(0, 255) for i in range(T)]
    return perturbed_list, delta_final


# --------------------------------------------------------------------------------------
# Evaluation: masked flow magnitude under each condition
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
    ap.add_argument("--iters", type=int, default=8, help="model recurrent iterations; lower to save memory")
    ap.add_argument("--mask-tau", type=float, default=0.15, help="MaskCut threshold")
    ap.add_argument("--no-crf", action="store_true", help="skip DenseCRF refinement of the mask")
    ap.add_argument("--smooth-factor", type=int, default=4,
                    help="parameterize delta at H/k x W/k, upsample to (H,W); k=1 means per-pixel (likely won't transfer to humans), k=4-8 forces a human-visible spatial scale")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale input so longer side is <= this (helps with GPU memory)")
    ap.add_argument("--n-eot", type=int, default=3,
                    help="EOT samples per PGD step (>=2 averages gradients over random augmentations; 1 disables EOT)")
    ap.add_argument("--objective", choices=["zero", "counter"], default="counter",
                    help="zero: minimize ||flow||^2 (see no motion); counter: push toward OPPOSITE of clean flow (see counter-motion). counter is more likely to produce a perturbation that engages human motion processing.")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    alpha = args.alpha if args.alpha is not None else args.eps / 10.0

    print(f"[setup] eps={args.eps}/255  alpha={alpha:.3f}  steps={args.steps}  iters={args.iters}")
    print(f"[setup] pydensecrf available: {HAS_CRF}")

    model = build_model(args.model)
    frames = load_clip(args.path, n=15, max_side=args.max_side)
    T, _, _, H, W = (len(frames),) + tuple(frames[0].shape)
    print(f"[setup] T={T}  H={H}  W={W}")

    # Mask from clean forward pass
    mid_img = to_numpy_uint8(frames[len(frames) // 2])
    mask_np = extract_mask(model, frames, args.iters, mid_img,
                           tau=args.mask_tau, use_crf=not args.no_crf)
    print(f"[mask] mask covers {100*mask_np.mean():.1f}% of frame area")
    Image.fromarray((mask_np * 255).astype(np.uint8)).save(os.path.join(args.save_dir, "mask.png"))

    if mask_np.sum() < 100:
        print("[mask] WARNING: mask is essentially empty. Try lowering --mask-tau.")
        return

    # Resize mask to flow output resolution (= input resolution here)
    mask_t = torch.from_numpy(mask_np).to(DEVICE)[None, None]
    if mask_t.shape[-2:] != (H, W):
        mask_t = F.interpolate(mask_t, size=(H, W), mode="nearest")

    # === Clean baseline ===
    clean_metrics, clean_out = masked_flow_magnitudes(model, frames, mask_t, args.iters)
    print(f"[clean] {clean_metrics}")
    # Keep final flow tensors from both pathways (needed for counter-motion target and visualization)
    clean_flow_final = clean_out["flow_seq"][-1].detach().clone()
    clean_flow_1_final = clean_out["flow_seq_1"][-1].detach().clone()
    del clean_out
    torch.cuda.empty_cache()

    # === PGD attack ===
    print(f"[attack] running PGD ({args.steps} steps, smooth_factor={args.smooth_factor}, n_eot={args.n_eot}, objective={args.objective})")
    perturbed_frames, delta = pgd_attack(model, frames, mask_t,
                                         eps=args.eps, steps=args.steps,
                                         alpha=alpha, iters=args.iters,
                                         smooth_factor=args.smooth_factor,
                                         n_eot=args.n_eot,
                                         clean_flow=clean_flow_final,
                                         clean_flow_1=clean_flow_1_final,
                                         objective=args.objective)
    torch.cuda.empty_cache()
    attack_metrics, attack_out = masked_flow_magnitudes(model, perturbed_frames, mask_t, args.iters)
    attack_flow_final = attack_out["flow_seq"][-1].detach().clone()
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
    del noise_out
    torch.cuda.empty_cache()
    print(f"[noise] {noise_metrics}")

    # === Save artifacts ===
    metrics = {
        "eps_pixel": args.eps,
        "eps_pct_of_255": 100 * args.eps / 255,
        "steps": args.steps,
        "alpha": alpha,
        "smooth_factor": args.smooth_factor,
        "n_eot": args.n_eot,
        "objective": args.objective,
        "mask_coverage_pct": float(100 * mask_np.mean()),
        "clean": clean_metrics,
        "attack": attack_metrics,
        "random_noise": noise_metrics,
        "delta_actual_max_abs": float(delta.abs().max().item()),
        "delta_actual_mean_abs": float(delta.abs().mean().item()),
    }
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Flow quivers: ONE PNG per condition (the model produces one flow per 15-frame
    # window, so an animation would just show stationary arrows). Use the middle
    # frame as the canvas.
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

    # Input frames as separate animations (no overlay): the question these answer
    # is "does the moving object still visibly move?" - which can't be answered
    # from a still arrow plot.
    save_gif([to_numpy_uint8(f) for f in frames],
             os.path.join(args.save_dir, "clean_video.gif"))
    save_gif([to_numpy_uint8(f) for f in perturbed_frames],
             os.path.join(args.save_dir, "attacked_video.gif"))
    save_gif([to_numpy_uint8(f) for f in rand_frames],
             os.path.join(args.save_dir, "noise_video.gif"))

    # Perturbation alone, amplified for visibility (delta is in [-eps, eps] ~ [-13, 13];
    # amplify so it spans most of [0,255])
    amp = 255.0 / (2 * args.eps)
    pert_uint8 = ((delta * amp) + 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    pert_uint8 = pert_uint8.transpose(0, 2, 3, 1)  # [T, H, W, 3]
    save_gif([pert_uint8[i] for i in range(T)], os.path.join(args.save_dir, "perturbation.gif"))

    print("[done]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
