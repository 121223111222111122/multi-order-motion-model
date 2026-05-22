"""Probe the multi-order motion model's spatiotemporal tuning.

WHY: our self-motion / vection attack assumes the model shares human self-motion
properties -- a magnocellular-like low-spatial-frequency / high-temporal-frequency
preference, and peripheral dominance. But the model is an object optical-flow net,
validated only on second-order *object* motion. This script measures, empirically,
what tuning it actually has, so we know whether that structure is real in the model or
purely our own prior imposed via masks.

WHAT IT MEASURES
  1. SF x TF grid: drifting sinusoidal gratings at fixed contrast across a grid of
     spatial frequencies (cyc/px) and temporal frequencies (cyc/frame). For each cell
     we record the model's estimated drift speed (projection of flow onto the drift
     direction) and the GAIN = estimated / true speed. Reading the grid along iso-speed
     diagonals isolates SF tuning at fixed speed -- the magno question.
  2. Center vs periphery: the same grating windowed to a central disk vs a peripheral
     annulus; compares per-area response. The model is a uniform full-frame processor,
     so the prediction is "no eccentricity preference" -- we verify it.

Both the combined output (flow_seq) and the first-order pathway (flow_seq_1) are
measured, since flow_1 is the channel we've been treating as human-relevant.

Outputs to --save_dir:
  tuning_speed.png / tuning_gain.png  - SF x TF heatmaps (combined + first-order)
  eccentricity.png                    - center vs periphery bar chart
  tuning.json                         - all numbers
"""
import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.nmi6.FFV1MT_MS import FFV1DNNV2

DEVICE = "cuda"


def build_model(ckpt):
    model = FFV1DNNV2(num_scales=8, upsample_factor=8, scale_factor=16, num_layers=6)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.module.cuda().eval()


@torch.no_grad()
def model_flow(model, frames, iters=4):
    out = model.forward(frames, mix_enable=False, layer=iters)
    return out["flow_seq"][-1], out["flow_seq_1"][-1]  # [1,2,H,W] each


def drifting_grating(H, W, sf, tf, n=15, contrast=0.5, direction=(1.0, 0.0), aperture=None):
    """List of n frames [1,3,H,W] in 0-255. sf in cyc/px, tf in cyc/frame.
    direction = drift unit vector. aperture: optional [H,W] in {0,1}; outside is gray."""
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dx, dy = direction
    phase_space = 2 * np.pi * sf * (xs * dx + ys * dy)
    frames = []
    for t in range(n):
        g = 0.5 + 0.5 * contrast * np.sin(phase_space - 2 * np.pi * tf * t)  # [H,W] in [0,1]
        if aperture is not None:
            g = g * aperture + 0.5 * (1.0 - aperture)
        img = np.stack([g, g, g], axis=0).astype(np.float32) * 255.0  # [3,H,W]
        frames.append(torch.from_numpy(img)[None].to(DEVICE))
    return frames


def proj_speed(flow, direction, region):
    """Mean signed projection of flow onto drift direction over region. flow [1,2,H,W]."""
    d = torch.tensor(direction, device=DEVICE).float()
    d = d / (d.norm() + 1e-6)
    p = flow[0, 0] * d[0] + flow[0, 1] * d[1]  # [H,W]
    m = torch.from_numpy(region).to(DEVICE).bool()
    return p[m].mean().item()


def disk(H, W, r_in_frac, r_out_frac):
    """Annulus/disk aperture in {0,1}: 1 where r_in_frac <= r/half <= r_out_frac."""
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = (H - 1) / 2, (W - 1) / 2
    half = min(H, W) / 2
    r = np.sqrt(((xs - cx)) ** 2 + ((ys - cy)) ** 2) / half
    return ((r >= r_in_frac) & (r <= r_out_frac)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="modelckpt/final_sintel_kitti.pth")
    ap.add_argument("--save_dir", default="results/tuning_probe")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--size", type=int, default=256, help="frame size (square, multiple of 8)")
    ap.add_argument("--contrast", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    H = W = args.size
    model = build_model(args.model)
    direction = (1.0, 0.0)  # horizontal drift

    # Grid: spatial freq (cyc/px) and temporal freq (cyc/frame)
    sfs = [1 / 64, 1 / 32, 1 / 16, 1 / 8]      # periods 64, 32, 16, 8 px
    tfs = [0.0625, 0.125, 0.25, 0.5]            # cyc/frame
    central = disk(H, W, 0.0, 0.7)              # avoid edge artifacts for full-field reads

    speed_comb = np.zeros((len(sfs), len(tfs)))
    speed_first = np.zeros((len(sfs), len(tfs)))
    gain_comb = np.zeros((len(sfs), len(tfs)))
    gain_first = np.zeros((len(sfs), len(tfs)))
    print("[grid] SF(cyc/px) x TF(cyc/frame) -> est speed (true speed) | gain")
    for i, sf in enumerate(sfs):
        for j, tf in enumerate(tfs):
            true_speed = tf / sf  # px/frame in drift direction
            frames = drifting_grating(H, W, sf, tf, contrast=args.contrast, direction=direction)
            flow, flow1 = model_flow(model, frames, args.iters)
            sc = proj_speed(flow, direction, central)
            sf1 = proj_speed(flow1, direction, central)
            speed_comb[i, j] = sc
            speed_first[i, j] = sf1
            gain_comb[i, j] = sc / true_speed if true_speed > 0 else 0
            gain_first[i, j] = sf1 / true_speed if true_speed > 0 else 0
            print(f"  sf=1/{round(1/sf):<3d} tf={tf:<6.4f} true={true_speed:6.2f}  "
                  f"comb={sc:7.3f} (gain {gain_comb[i,j]:.2f})  first={sf1:7.3f} (gain {gain_first[i,j]:.2f})")

    # === SF x TF heatmaps ===
    def heatmap(mat, title, fname, fmt="{:.2f}"):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(tfs))); ax.set_xticklabels([f"{t:g}" for t in tfs])
        ax.set_yticks(range(len(sfs))); ax.set_yticklabels([f"1/{round(1/s)}" for s in sfs])
        ax.set_xlabel("temporal freq (cyc/frame)"); ax.set_ylabel("spatial freq (cyc/px)")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white", fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, fname), dpi=120)
        plt.close(fig)

    heatmap(speed_comb, "Estimated speed (combined)", "tuning_speed_combined.png")
    heatmap(speed_first, "Estimated speed (first-order)", "tuning_speed_first.png")
    heatmap(gain_comb, "Gain = est/true (combined)", "tuning_gain_combined.png")
    heatmap(gain_first, "Gain = est/true (first-order)", "tuning_gain_first.png")

    # === Center vs periphery (fixed mid SF/TF) ===
    sf_e, tf_e = 1 / 32, 0.25  # speed 8 px/frame, mid of the grid
    center_ap = disk(H, W, 0.0, 0.35)
    periph_ap = disk(H, W, 0.5, 1.0)
    ecc = {}
    for name, ap_mask in [("center", center_ap), ("periphery", periph_ap)]:
        frames = drifting_grating(H, W, sf_e, tf_e, contrast=args.contrast, direction=direction, aperture=ap_mask)
        flow, flow1 = model_flow(model, frames, args.iters)
        region = ap_mask  # measure where the grating actually is
        ecc[name] = {
            "speed_combined": proj_speed(flow, direction, region),
            "speed_first": proj_speed(flow1, direction, region),
            "true_speed": tf_e / sf_e,
        }
        print(f"[ecc] {name}: comb={ecc[name]['speed_combined']:.3f}  first={ecc[name]['speed_first']:.3f}  (true {tf_e/sf_e:.2f})")

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [ecc["center"]["speed_combined"], ecc["periphery"]["speed_combined"]], 0.4, label="combined")
    ax.bar(x + 0.2, [ecc["center"]["speed_first"], ecc["periphery"]["speed_first"]], 0.4, label="first-order")
    ax.axhline(tf_e / sf_e, ls="--", c="k", lw=1, label="true speed")
    ax.set_xticks(x); ax.set_xticklabels(["center", "periphery"])
    ax.set_ylabel("estimated speed (px/frame)")
    ax.set_title(f"Center vs periphery (sf=1/32, tf=0.25)")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.save_dir, "eccentricity.png"), dpi=120); plt.close(fig)

    # === Area-matched eccentricity: equal-area annular rings ===
    # Each ring has the SAME area (r_out^2 - r_in^2 constant), only mean eccentricity
    # differs -- so this isolates a true retinotopic effect from the stimulus-area
    # confound in the center-vs-periphery test above.
    n_rings = 4
    ring_results = []
    print(f"\n[area-matched eccentricity] {n_rings} equal-area rings, sf=1/32, tf=0.25")
    for k in range(n_rings):
        r_in = (k / n_rings) ** 0.5
        r_out = ((k + 1) / n_rings) ** 0.5
        ap_mask = disk(H, W, r_in, r_out)
        mean_ecc = 0.5 * (r_in + r_out)
        frames = drifting_grating(H, W, sf_e, tf_e, contrast=args.contrast, direction=direction, aperture=ap_mask)
        flow, flow1 = model_flow(model, frames, args.iters)
        sc = proj_speed(flow, direction, ap_mask)
        sf1 = proj_speed(flow1, direction, ap_mask)
        area_px = float(ap_mask.sum())
        ring_results.append({"ring": k, "r_in": r_in, "r_out": r_out, "mean_ecc": mean_ecc,
                             "area_px": area_px, "speed_combined": sc, "speed_first": sf1})
        print(f"  ring {k} ecc~{mean_ecc:.2f} (r {r_in:.2f}-{r_out:.2f}) area={area_px:.0f}px  "
              f"comb={sc:.3f}  first={sf1:.3f}  (true {tf_e/sf_e:.2f})")

    fig, ax = plt.subplots(figsize=(6, 4))
    eccs = [r["mean_ecc"] for r in ring_results]
    ax.plot(eccs, [r["speed_combined"] for r in ring_results], "o-", label="combined")
    ax.plot(eccs, [r["speed_first"] for r in ring_results], "s-", label="first-order")
    ax.axhline(tf_e / sf_e, ls="--", c="k", lw=1, label="true speed")
    ax.set_xlabel("mean eccentricity (0=center, 1=edge midpoint)")
    ax.set_ylabel("estimated speed (px/frame)")
    ax.set_title(f"Area-matched eccentricity ({n_rings} equal-area rings)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "eccentricity_area_matched.png"), dpi=120)
    plt.close(fig)

    # === Iso-speed SF-tuning readout (the magno question) ===
    # For each true speed that appears multiple times in the grid, list gain vs SF.
    iso = {}
    for i, sf in enumerate(sfs):
        for j, tf in enumerate(tfs):
            s = round(tf / sf, 3)
            iso.setdefault(s, []).append((f"1/{round(1/sf)}", round(gain_first[i, j], 3)))
    print("\n[iso-speed first-order gain vs SF] (does gain rise toward LOW SF? = magno-like)")
    for s in sorted(iso):
        if len(iso[s]) > 1:
            print(f"  speed {s:5.2f} px/frame: " + "  ".join(f"{sf}->{g}" for sf, g in iso[s]))

    out = {
        "sfs_cyc_per_px": sfs, "tfs_cyc_per_frame": tfs,
        "speed_combined": speed_comb.tolist(), "speed_first": speed_first.tolist(),
        "gain_combined": gain_comb.tolist(), "gain_first": gain_first.tolist(),
        "eccentricity": ecc,
        "eccentricity_area_matched": ring_results,
        "iso_speed_first_gain_vs_sf": {str(k): v for k, v in iso.items()},
    }
    with open(os.path.join(args.save_dir, "tuning.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] heatmaps + tuning.json in {args.save_dir}")


if __name__ == "__main__":
    main()
