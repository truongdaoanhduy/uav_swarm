"""
examples/record_video.py
Record episode thành MP4 video hoặc GIF.

Usage:
    # Từ project root:
    python examples/record_video.py --mode mp4 --steps 300 --seed 42
    python examples/record_video.py --mode gif  --steps 100 --seed 42
    
    # Hoặc:
    cd examples && python record_video.py --mode mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# FIX: Add project root to sys.path
# Đảm bảo import được dù chạy từ bất kỳ đâu
# ─────────────────────────────────────────────────────────────────────────────
def _setup_path():
    """Add project root to sys.path."""
    # File này ở: <project_root>/examples/record_video.py
    this_file   = os.path.abspath(__file__)
    examples_dir = os.path.dirname(this_file)
    project_root = os.path.dirname(examples_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return project_root

PROJECT_ROOT = _setup_path()


# ─────────────────────────────────────────────────────────────────────────────

def record_episode(
    mode:    str = "mp4",
    steps:   int = 300,
    seed:    int = 42,
    fps:     int = 15,
    out_dir: str = "results/videos",
) -> str:
    """
    Record 1 episode → MP4 hoặc GIF.

    Returns:
        Path to output file.
    """
    # Import SAU khi path đã setup
    from config import AppConfig
    from env.base_env import SARBaseEnv

    # Output dir relative to project root
    out_dir_abs = os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    print(f"[RECORD] Project root : {PROJECT_ROOT}")
    print(f"[RECORD] Output dir   : {out_dir_abs}")
    print(f"[RECORD] Mode={mode} | steps={steps} | seed={seed} | fps={fps}")

    cfg = AppConfig()
    cfg.env.max_steps = steps

    env = SARBaseEnv(
        cfg         = cfg,
        render_mode = "rgb_array",
        viz_mode    = "3d",
        verbose     = 1,
    )

    obs, info = env.reset(seed=seed)
    print(
        f"[RECORD] Reset OK | "
        f"{info['n_uav']} UAVs | "
        f"{info['n_victims']} victims | "
        f"map={info['map_size']}m"
    )

    # ── Collect frames ────────────────────────────────────────────────────────
    frames = []

    # Frame 0: initial state
    frame = env.render()
    if frame is not None:
        frames.append(frame)
        print(f"[RECORD] Frame shape: {frame.shape} dtype={frame.dtype}")

    t0 = time.time()
    for step in range(steps):
        actions = {
            uid: env.action_space.sample()
            for uid in obs.keys()
        }

        obs, rewards, done, truncated, info = env.step(actions)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if step % 50 == 0:
            elapsed = time.time() - t0
            fps_actual = len(frames) / max(elapsed, 1e-6)
            print(
                f"  step={step:4d} | "
                f"cov={info['coverage_rate']:.1%} | "
                f"victims={info['victims_found']}/{info['victims_total']} | "
                f"render_fps={fps_actual:.1f}"
            )

        if done or truncated:
            reason = "done" if done else "truncated"
            print(f"[RECORD] Episode ended at step {step} | reason={reason}")
            break

    env.close()

    elapsed_total = time.time() - t0
    print(
        f"[RECORD] Collected {len(frames)} frames | "
        f"total={elapsed_total:.1f}s | "
        f"avg={1000*elapsed_total/max(len(frames),1):.0f}ms/frame"
    )

    if not frames:
        print("[RECORD] ERROR: No frames collected!")
        return ""

    # ── Normalize frames ──────────────────────────────────────────────────────
    # Đảm bảo tất cả frames cùng shape và dtype
    h, w = frames[0].shape[:2]
    clean_frames = []
    for i, f in enumerate(frames):
        if f.shape[:2] != (h, w):
            print(f"[RECORD] WARNING: frame {i} shape mismatch {f.shape} vs ({h},{w})")
            continue
        clean_frames.append(f.astype(np.uint8))

    frames = clean_frames
    print(f"[RECORD] Final: {len(frames)} frames | {w}x{h}px")

    # ── Save ──────────────────────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if mode == "mp4":
        out_path = _save_mp4(frames, out_dir_abs, timestamp, fps)
    elif mode == "gif":
        out_path = _save_gif(frames, out_dir_abs, timestamp, fps)
    elif mode == "both":
        out_path = _save_mp4(frames, out_dir_abs, timestamp, fps)
        _save_gif(frames, out_dir_abs, timestamp, fps)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _save_mp4(frames, out_dir, timestamp, fps) -> str:
    """Save MP4 via OpenCV → imageio → fallback GIF."""
    out_path = os.path.join(out_dir, f"episode_{timestamp}.mp4")
    h, w     = frames[0].shape[:2]

    # ── Try OpenCV ────────────────────────────────────────────────────────────
    try:
        import cv2

        # Thử codec mp4v trước, fallback XVID
        for fourcc_str in ("mp4v", "XVID", "avc1"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
            if writer.isOpened():
                break
            writer.release()
        else:
            raise RuntimeError("No working codec found")

        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

        size_mb = os.path.getsize(out_path) / 1e6
        print(
            f"[RECORD] ✅ MP4 via OpenCV | "
            f"{len(frames)} frames | {fps}fps | {size_mb:.1f}MB\n"
            f"         → {out_path}"
        )
        return out_path

    except ImportError:
        print("[RECORD] OpenCV not found, trying imageio+ffmpeg...")
    except Exception as e:
        print(f"[RECORD] OpenCV failed: {e}, trying imageio...")

    # ── Try imageio + ffmpeg ──────────────────────────────────────────────────
    try:
        import imageio
        import imageio.plugins.ffmpeg  # noqa: ensure ffmpeg available

        writer = imageio.get_writer(
            out_path,
            fps     = fps,
            codec   = "libx264",
            quality = 8,
            macro_block_size = 1,
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        size_mb = os.path.getsize(out_path) / 1e6
        print(
            f"[RECORD] ✅ MP4 via imageio | "
            f"{len(frames)} frames | {fps}fps | {size_mb:.1f}MB\n"
            f"         → {out_path}"
        )
        return out_path

    except ImportError:
        print("[RECORD] imageio/ffmpeg not found, falling back to GIF...")
    except Exception as e:
        print(f"[RECORD] imageio failed: {e}, falling back to GIF...")

    # ── Fallback: GIF ─────────────────────────────────────────────────────────
    gif_path = out_path.replace(".mp4", ".gif")
    return _save_gif(frames, out_dir, timestamp, fps, path_override=gif_path)


def _save_gif(
    frames,
    out_dir,
    timestamp,
    fps,
    path_override: str | None = None,
) -> str:
    """Save GIF via Pillow."""
    out_path    = path_override or os.path.join(out_dir, f"episode_{timestamp}.gif")
    duration_ms = max(20, int(1000 / fps))  # Pillow min ~20ms

    try:
        from PIL import Image

        pil_frames = [
            Image.fromarray(f, mode="RGB")
            for f in frames
        ]

        # Quantize để giảm file size (256 màu per frame)
        pil_frames_q = []
        for pf in pil_frames:
            pil_frames_q.append(
                pf.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            )

        pil_frames_q[0].save(
            out_path,
            format       = "GIF",
            save_all     = True,
            append_images= pil_frames_q[1:],
            duration     = duration_ms,
            loop         = 0,
            optimize     = True,
        )

        size_mb = os.path.getsize(out_path) / 1e6
        print(
            f"[RECORD] ✅ GIF via Pillow | "
            f"{len(frames)} frames | {fps}fps | "
            f"{duration_ms}ms/frame | {size_mb:.1f}MB\n"
            f"         → {out_path}"
        )
        return out_path

    except ImportError:
        print("[RECORD] ERROR: Pillow not installed! pip install Pillow")
        return ""
    except Exception as e:
        print(f"[RECORD] ERROR saving GIF: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Record SAR UAV Swarm episode as video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",  default="mp4",
                   choices=["mp4", "gif", "both"],
                   help="Output format")
    p.add_argument("--steps", type=int, default=300,
                   help="Max episode steps to record")
    p.add_argument("--seed",  type=int, default=42,
                   help="Random seed")
    p.add_argument("--fps",   type=int, default=15,
                   help="Video FPS")
    p.add_argument("--out",   default="results/videos",
                   help="Output directory (relative to project root)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    record_episode(
        mode    = args.mode,
        steps   = args.steps,
        seed    = args.seed,
        fps     = args.fps,
        out_dir = args.out,
    )