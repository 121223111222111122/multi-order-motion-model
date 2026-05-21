"""Extract frames from a video file into a folder of PNGs, ready for infer_motion.py.

Usage:
    python extract_frames.py <video_path> <output_folder> [--fps N] [--max-frames N] [--max-side N]

Example:
    python extract_frames.py demo/test-stimuli/motion-standstill.mp4 demo/test-stimuli/motion-standstill --max-side 512
"""
import argparse
import os
import imageio.v3 as iio
from PIL import Image
import numpy as np


def resize_max_side(frame, max_side):
    """Resize so the longer side is <= max_side, preserving aspect ratio. Rounds to multiple of 8."""
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= max_side:
        return frame
    scale = max_side / long
    new_w = int(round(w * scale / 8)) * 8
    new_h = int(round(h * scale / 8)) * 8
    img = Image.fromarray(frame).resize((new_w, new_h), Image.BICUBIC)
    return np.array(img)


def extract(video_path, out_dir, fps=None, max_frames=None, max_side=None):
    os.makedirs(out_dir, exist_ok=True)
    kwargs = {}
    if fps is not None:
        kwargs["fps"] = fps
    n = 0
    first_shape = None
    for i, frame in enumerate(iio.imiter(video_path, **kwargs)):
        if max_side is not None:
            frame = resize_max_side(frame, max_side)
        if first_shape is None:
            first_shape = frame.shape
        out_path = os.path.join(out_dir, f"frame_{i:04d}.png")
        iio.imwrite(out_path, frame)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    print(f"Wrote {n} frames to {out_dir}  (frame size: {first_shape[1]}x{first_shape[0]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="input video file (mp4, gif, etc.)")
    ap.add_argument("out_dir", help="output folder for frames")
    ap.add_argument("--fps", type=float, default=None, help="resample to this frame rate (optional)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many frames (optional)")
    ap.add_argument("--max-side", type=int, default=None, help="downscale so longer side is at most this many pixels (optional)")
    args = ap.parse_args()
    extract(args.video, args.out_dir, fps=args.fps, max_frames=args.max_frames, max_side=args.max_side)
