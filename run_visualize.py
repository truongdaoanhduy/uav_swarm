#!/usr/bin/env python3
"""
run_visualization.py
Visualization cho 3 algo (MAPPO/MASAC/MATD3) trên 2 stages (HARD/EXTREME).
Tự động upload GIF/PNG lên HuggingFace sau khi chạy xong.

Usage:
    # 3 algo, 2 stages, upload HF
    python run_visualization.py \
        --mappo  mappo_s42/checkpoint_final.pt \
        --masac  masac_s42/checkpoint_final.pt \
        --matd3  matd3_s42/checkpoint_final.pt \
        --stages hard extreme \
        --hf-token  hf_xxxx \
        --hf-repo   username/sar-uav-viz \
        --n-episodes 2 \
        --fps 10

    # Chỉ 1 algo, không upload HF
    python run_visualization.py \
        --mappo mappo_s42/checkpoint_final.pt \
        --stages extreme
"""

import argparse
import os
import time
import json
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import AppConfig
from config.curriculum_config import STAGE_TRANSFER, STAGE_EXTREME

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BASE_OUTPUT_DIR = Path("results/viz")

STAGE_MAP = {
    "hard":    STAGE_TRANSFER,
    "extreme": STAGE_EXTREME,
}

ALGO_COLORS = {
    "mappo": "#2196F3",
    "masac": "#4CAF50",
    "matd3": "#FF9800",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SAR UAV Visualizer — 3 algos × 2 stages → HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoints (truyền riêng từng algo) ─────────────────────────────────
    p.add_argument("--mappo", type=str, default=None,
                   help="Path đến MAPPO checkpoint_final.pt")
    p.add_argument("--masac", type=str, default=None,
                   help="Path đến MASAC checkpoint_final.pt")
    p.add_argument("--matd3", type=str, default=None,
                   help="Path đến MATD3 checkpoint_final.pt")

    # ── Stages ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--stages",
        nargs   = "+",
        default = ["transfer", "extreme"],
        choices = ["transfer", "extreme"],
        help    = "Stages để visualize",
    )

    # ── Render ───────────────────────────────────────────────────────────────
    p.add_argument("--mode",       type=str, default="2d",
                   choices=["2d", "3d"])
    p.add_argument("--n-episodes", type=int, default=1,
                   help="Số episodes mỗi (algo × stage)")
    p.add_argument("--max-steps",  type=int, default=None,
                   help="Override max_steps (None = dùng stage default)")
    p.add_argument("--fps",        type=int, default=10,
                   help="FPS cho GIF output")
    p.add_argument("--no-gif",     action="store_true",
                   help="Không tạo GIF, chỉ lưu frames PNG")
    p.add_argument("--no-frames",  action="store_true",
                   help="Không lưu từng frame PNG (tiết kiệm disk)")

    # ── HuggingFace ──────────────────────────────────────────────────────────
    p.add_argument("--hf-token",  type=str, default=None,
                   help="HuggingFace API token để upload GIF/PNG")
    p.add_argument("--hf-repo",   type=str, default=None,
                   help="HF repo ID, vd: username/sar-uav-viz")
    p.add_argument("--no-upload", action="store_true",
                   help="Không upload lên HF dù có token")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--seed",   type=int, default=44)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n-uav",  type=int, default=4)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT ALGO TỪ CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def detect_algo_from_checkpoint(ckpt: dict) -> str:
    """
    Tự động detect algo từ keys trong state_dict.

    MAPPO  → có "log_std" (learnable, state-independent)
    MASAC  → có "move_mean_head" + "move_log_std_head"
    MATD3  → có "movement_head" nhưng KHÔNG có "log_std"
    """
    actor_state = (
        ckpt.get("actor_state_dict")
        or ckpt.get("actor")
        or ckpt.get("state_dict")
        or ckpt
    )

    if not isinstance(actor_state, dict):
        return "unknown"

    keys = set(actor_state.keys())

    if any("move_mean_head" in k for k in keys):
        return "masac"
    if "log_std" in keys:
        return "mappo"
    if any("movement_head" in k for k in keys):
        return "matd3"

    # Fallback: metadata trong checkpoint
    meta = ckpt.get("algo", ckpt.get("algorithm", "")).lower()
    if meta in ("mappo", "masac", "matd3"):
        return meta

    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# ACTOR LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_actor(
    checkpoint_path: str,
    algo:            str,
    config:          AppConfig,
    device:          str,
) -> Tuple[torch.nn.Module, int, str]:
    """
    Load actor từ checkpoint.
    Tự động detect algo nếu truyền sai --algo.

    Returns:
        (actor, trained_episodes, actual_algo)
    """
    obs_dim    = config.obs.actor_dim
    action_dim = 4
    tr         = config.train

    print(f"\n  📂 Loading: {Path(checkpoint_path).name}")

    ckpt = torch.load(
        checkpoint_path,
        map_location = device,
        weights_only = False,
    )

    # Auto-detect
    detected = detect_algo_from_checkpoint(ckpt)
    if detected != "unknown" and detected != algo:
        print(f"  ⚠️  --algo={algo} nhưng checkpoint là {detected.upper()}")
        print(f"  🔄  Tự động dùng: {detected.upper()}")
        algo = detected
    elif detected == "unknown":
        print(f"  ⚠️  Không detect được algo, dùng: {algo.upper()}")

    # Build network
    if algo == "mappo":
        from training.algorithms.mappo.actor import ActorNetwork
        actor = ActorNetwork(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_dims    = tr.mappo_actor_hidden,
            activation     = tr.mappo_activation,
            use_layer_norm = tr.mappo_use_layer_norm,
            log_std_init   = -0.5,
        )
    elif algo == "masac":
        from training.algorithms.masac.actor import SACActorNetwork
        actor = SACActorNetwork(
            obs_dim     = obs_dim,
            action_dim  = action_dim,
            hidden_dims = tr.masac_actor_hidden,
        )
    elif algo == "matd3":
        from training.algorithms.matd3.actor import TD3ActorNetwork
        actor = TD3ActorNetwork(
            obs_dim     = obs_dim,
            action_dim  = action_dim,
            hidden_dims = tr.matd3_actor_hidden,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # Tìm state dict
    actor_state = None
    for key in ["actor_state_dict", "actor", "model", "state_dict"]:
        if key in ckpt:
            actor_state = ckpt[key]
            print(f"  📦 Key: '{key}'")
            break
    if actor_state is None:
        actor_state = ckpt

    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        print(f"  ⚠️  Missing  ({len(missing)}): {missing[:2]}...")
    if unexpected:
        print(f"  ⚠️  Unexpected ({len(unexpected)}): {unexpected[:2]}...")

    actor.eval()
    actor.to(device)

    ep = ckpt.get("total_episodes_done", ckpt.get("episode", 0))
    print(f"  ✅ {algo.upper()} loaded | trained {ep:,} eps")

    return actor, ep, algo


# ══════════════════════════════════════════════════════════════════════════════
# SAVE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_frame_png(frame: np.ndarray, path: Path, title: str = ""):
    """Lưu 1 frame ra PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=11, pad=6)
        fig.tight_layout(pad=0.5)
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️  PNG save failed: {e}")


def save_gif_file(frames: list, path: Path, fps: int = 10) -> bool:
    """Lưu frames thành GIF. Thử Pillow → imageio → matplotlib."""
    if not frames:
        print("  ⚠️  No frames to save")
        return False

    valid = [f for f in frames if f is not None]
    if not valid:
        return False

    # Method 1: Pillow
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in valid]
        imgs[0].save(
            str(path),
            save_all      = True,
            append_images = imgs[1:],
            duration      = 1000 // fps,
            loop          = 0,
            optimize      = False,
        )
        print(f"  ✅ GIF saved: {path.name} ({len(imgs)} frames, {fps}fps)")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠️  Pillow failed: {e}")

    # Method 2: imageio
    try:
        import imageio
        imageio.mimsave(str(path), [f.astype(np.uint8) for f in valid], fps=fps)
        print(f"  ✅ GIF saved via imageio: {path.name}")
        return True
    except Exception as e:
        print(f"  ⚠️  imageio failed: {e}")

    # Method 3: matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        im = ax.imshow(valid[0])

        def update(i):
            im.set_data(valid[i])
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=len(valid),
            interval=1000 // fps, blit=True,
        )
        ani.save(str(path), writer="pillow", fps=fps)
        plt.close(fig)
        print(f"  ✅ GIF saved via matplotlib: {path.name}")
        return True
    except Exception as e:
        print(f"  ⚠️  matplotlib failed: {e}")

    return False


def save_summary_plot(
    results:      Dict,   # {algo: {stage: [result_dicts]}}
    run_dir:      Path,
    stages:       List[str],
    algos:        List[str],
) -> Path:
    """
    Lưu summary plot so sánh tất cả algo × stage.

    Layout:
        Rows = metrics (Reward, Coverage, Victims, Success)
        Cols = stages  (HARD, EXTREME)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics_keys = [
            ("ep_reward", "Episode Reward",    False),
            ("coverage",  "Coverage (%)",      False),
            ("victims_f", "Victims Found",     False),
            ("success",   "Success Rate (%)",  True),
        ]

        n_rows = len(metrics_keys)
        n_cols = len(stages)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize = (6 * n_cols, 4 * n_rows),
            squeeze = False,
        )
        fig.suptitle(
            "SAR UAV — Visualization Summary",
            fontsize=15, fontweight="bold", y=1.01,
        )

        for col_i, stage in enumerate(stages):
            for row_i, (mkey, mlabel, is_pct) in enumerate(metrics_keys):
                ax = axes[row_i][col_i]
                ax.set_title(f"{mlabel}\n[{stage.upper()}]", fontweight="bold")
                ax.set_facecolor("#F8F8F8")
                ax.grid(axis="y", alpha=0.3)

                x_pos  = np.arange(len(algos))
                width  = 0.6

                for bar_i, algo in enumerate(algos):
                    eps_list = results.get(algo, {}).get(stage, [])
                    if not eps_list:
                        continue

                    vals = [r.get(mkey, 0) for r in eps_list]
                    # Success → %
                    if is_pct:
                        vals = [v * 100 if v <= 1 else v for v in vals]

                    mean = np.mean(vals)
                    std  = np.std(vals) if len(vals) > 1 else 0

                    bar = ax.bar(
                        bar_i, mean,
                        width     = width,
                        color     = ALGO_COLORS.get(algo, "#999"),
                        alpha     = 0.85,
                        edgecolor = "white",
                        linewidth = 1.2,
                        label     = algo.upper(),
                    )
                    if std > 0:
                        ax.errorbar(
                            bar_i, mean, yerr=std,
                            fmt="none", color="black",
                            capsize=5, linewidth=1.5,
                        )
                    ax.text(
                        bar_i, mean + std + 0.01 * abs(mean + 1),
                        f"{mean:.1f}",
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                    )

                ax.set_xticks(x_pos)
                ax.set_xticklabels([a.upper() for a in algos])
                ax.set_ylabel(mlabel, fontsize=9)

        # Legend chung
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                color = ALGO_COLORS.get(a, "#999"),
                alpha = 0.85,
                label = a.upper(),
            )
            for a in algos
        ]
        fig.legend(
            handles    = handles,
            loc        = "upper right",
            fontsize   = 10,
            framealpha = 0.9,
        )

        plt.tight_layout()
        out = run_dir / "summary_comparison.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Summary plot: {out.name}")
        return out

    except Exception as e:
        print(f"  ⚠️  Summary plot failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# RUN EPISODE
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    config:      AppConfig,
    seed:        int,
    actor:       torch.nn.Module,
    algo:        str,
    device:      str,
    run_dir:     Path,
    ep_idx:      int,
    stage:       str,
    save_frames: bool = True,
) -> Tuple[List[np.ndarray], dict]:
    """
    Chạy 1 episode và capture frames.

    Returns:
        (frames, result_dict)
    """
    from env_setup.sar_pettingzoo_env import SARPettingZooEnv

    env      = SARPettingZooEnv(config, render_mode="rgb_array")
    n_agents = config.env.n_uav
    obs_dim  = config.obs.actor_dim

    obs_d, _ = env.reset(seed=seed)

    ep_reward = 0.0
    ep_steps  = 0
    frames    = []
    done      = False

    # Thư mục frames cho episode này
    frames_dir = run_dir / "frames" / f"{algo}_{stage}_ep{ep_idx:02d}"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  🚁 {algo.upper()} | {stage.upper()} | ep={ep_idx+1} | seed={seed}")
    print(f"  {'─'*55}")

    while not done:
        # Get actions
        obs_arr = np.array(
            [obs_d.get(f"uav_{i}", np.zeros(obs_dim, np.float32))
             for i in range(n_agents)],
            dtype=np.float32,
        )
        obs_t = torch.FloatTensor(obs_arr).to(device)

        with torch.no_grad():
            if algo in ("mappo", "masac"):
                act_t, _ = actor.get_action(obs_t, deterministic=True)
            else:  # matd3
                act_t, _ = actor.get_action(
                    obs_t, explore_noise=0.0, deterministic=True
                )

        act_np   = np.clip(act_t.cpu().numpy(), -1.0, 1.0)
        act_dict = {f"uav_{i}": act_np[i] for i in range(n_agents)}

        # Step
        obs_d, rew_d, term_d, trunc_d, info = env.step(act_dict)
        ep_reward += sum(rew_d.values())
        ep_steps  += 1
        done       = any(term_d.values()) or any(trunc_d.values())

        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())
            if save_frames:
                frame_path = frames_dir / f"frame_{ep_steps:04d}.png"
                save_frame_png(
                    frame, frame_path,
                    title=(f"{algo.upper()} | {stage.upper()} "
                           f"| Step {ep_steps}/{config.env.max_steps}"),
                )

        # Progress log
        if ep_steps % 100 == 0 or done:
            u0  = info.get("uav_0", {})
            cov = u0.get("coverage_rate", 0.0) * 100
            vf  = u0.get("victims_found", 0)
            vt  = u0.get("victims_total", 1)
            na  = u0.get("n_active",      0)
            nc  = u0.get("n_charging",    0)
            nd  = u0.get("n_disabled",    0)
            bat = u0.get("battery_stats", {}).get("mean", 0.0)
            print(
                f"  step={ep_steps:4d}/{config.env.max_steps}"
                f" | cov={cov:5.1f}%"
                f" | vic={vf}/{vt}"
                f" | rew={ep_reward:+8.1f}"
                f" | {na}act {nc}chg {nd}dis"
                f" | bat={bat:.0f}%"
            )

    # Episode summary
    u0          = info.get("uav_0", {})
    ep_metrics  = u0.get("episode", {})
    coverage    = float(ep_metrics.get("coverage_rate", u0.get("coverage_rate", 0.0)))
    victims_f   = int(ep_metrics.get("victims_found",   u0.get("victims_found",  0)))
    victims_t   = int(ep_metrics.get("total_victims",   u0.get("victims_total",  1)))
    success     = bool(ep_metrics.get("success",        u0.get("success",        False)))
    done_reason = ep_metrics.get("done_reason",         u0.get("done_reason",    "unknown"))

    env.close()

    print(f"\n  {'─'*55}")
    print(f"  ✅ Done | steps={ep_steps} | rew={ep_reward:+.1f} "
          f"| cov={coverage*100:.1f}% "
          f"| vic={victims_f}/{victims_t} "
          f"| {'✓' if success else '✗'} {done_reason}")

    return frames, {
        "ep_reward": ep_reward,
        "ep_steps":  ep_steps,
        "coverage":  coverage * 100,
        "victims_f": victims_f,
        "victims_t": victims_t,
        "success":   success,
        "algo":      algo,
        "stage":     stage,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE UPLOADER
# ══════════════════════════════════════════════════════════════════════════════

class VizHFUploader:
    """Upload GIF/PNG visualization lên HuggingFace."""

    def __init__(self, token: str, repo_id: str):
        self.token   = token
        self.repo_id = repo_id
        self._api    = None
        self._init()

    def _init(self):
        try:
            from huggingface_hub import HfApi, create_repo
            self._api = HfApi(token=self.token)
            create_repo(
                self.repo_id,
                token     = self.token,
                exist_ok  = True,
                repo_type = "dataset",
            )
            print(f"  ✅ HF repo sẵn sàng: {self.repo_id}")
        except ImportError:
            print("  ❌ pip install huggingface-hub")
        except Exception as e:
            print(f"  ⚠️  HF init: {e}")

    def upload(
        self,
        local_path: str,
        repo_path:  str,
        msg:        str = "Upload viz",
    ) -> bool:
        if self._api is None:
            return False
        try:
            self._api.upload_file(
                path_or_fileobj = local_path,
                path_in_repo    = repo_path,
                repo_id         = self.repo_id,
                repo_type       = "dataset",
                commit_message  = msg,
            )
            print(f"  ☁️  Uploaded → {repo_path}")
            return True
        except Exception as e:
            print(f"  ❌ Upload failed {Path(local_path).name}: {e}")
            return False

    def upload_run_results(
        self,
        run_dir:   Path,
        gif_files: Dict[str, Path],   # {"{algo}_{stage}": path}
        png_files: Dict[str, Path],   # {"{algo}_{stage}": path}
        summary:   Optional[Path],
        meta:      dict,
    ):
        """
        Upload toàn bộ kết quả 1 run lên HF.

        Cấu trúc HF repo:
            visualizations/
            ├── {timestamp}/
            │   ├── mappo_hard.gif
            │   ├── mappo_extreme.gif
            │   ├── masac_hard.gif
            │   ├── masac_extreme.gif
            │   ├── matd3_hard.gif
            │   ├── matd3_extreme.gif
            │   ├── summary_comparison.png
            │   └── meta.json
            └── latest/
                ├── mappo_hard.gif       ← luôn overwrite
                └── ...
        """
        timestamp = meta.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))

        print(f"\n  📤 Uploading to HF: {self.repo_id}")
        print(f"  {'─'*50}")

        uploaded = 0
        total    = 0

        # GIFs
        for key, gif_path in gif_files.items():
            if gif_path and gif_path.exists():
                total += 1
                # Upload vào timestamped folder
                ok1 = self.upload(
                    str(gif_path),
                    f"visualizations/{timestamp}/{key}.gif",
                    f"GIF {key}",
                )
                # Upload vào latest/ (overwrite)
                ok2 = self.upload(
                    str(gif_path),
                    f"visualizations/latest/{key}.gif",
                    f"Latest GIF {key}",
                )
                uploaded += ok1

        # PNGs (summary)
        for key, png_path in png_files.items():
            if png_path and png_path.exists():
                total += 1
                ok = self.upload(
                    str(png_path),
                    f"visualizations/{timestamp}/{key}.png",
                    f"PNG {key}",
                )
                self.upload(
                    str(png_path),
                    f"visualizations/latest/{key}.png",
                    f"Latest PNG {key}",
                )
                uploaded += ok

        # Summary plot
        if summary and summary.exists():
            total += 1
            ok = self.upload(
                str(summary),
                f"visualizations/{timestamp}/summary_comparison.png",
                "Summary comparison plot",
            )
            self.upload(
                str(summary),
                "visualizations/latest/summary_comparison.png",
                "Latest summary",
            )
            uploaded += ok

        # Meta JSON
        meta_path = run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        total += 1
        ok = self.upload(
            str(meta_path),
            f"visualizations/{timestamp}/meta.json",
            "Run metadata",
        )
        uploaded += ok

        print(f"\n  ✅ Uploaded {uploaded}/{total} files")
        print(f"  🔗 https://huggingface.co/datasets/{self.repo_id}"
              f"/tree/main/visualizations/{timestamp}")

        return uploaded == total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    t_start = time.time()

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    # ── Collect checkpoints ───────────────────────────────────────────────────
    # {algo: path_string}
    raw_ckpts = {
        "mappo": args.mappo,
        "masac": args.masac,
        "matd3": args.matd3,
    }
    checkpoints = {
        algo: path
        for algo, path in raw_ckpts.items()
        if path is not None and Path(path).exists()
    }

    # Báo lỗi nếu path không tồn tại
    for algo, path in raw_ckpts.items():
        if path is not None and not Path(path).exists():
            print(f"  ⚠️  {algo.upper()}: file không tồn tại: {path}")

    if not checkpoints:
        print("❌ Không có checkpoint nào hợp lệ!")
        print("   Truyền: --mappo path.pt --masac path.pt --matd3 path.pt")
        return

    stages = args.stages

    # ── HF setup ──────────────────────────────────────────────────────────────
    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    uploader = None
    if hf_token and args.hf_repo and not args.no_upload:
        uploader = VizHFUploader(hf_token, args.hf_repo)
    elif not args.no_upload and (not hf_token or not args.hf_repo):
        print("  ℹ️  Không upload HF (thiếu --hf-token hoặc --hf-repo)")

    # ── Output dir ────────────────────────────────────────────────────────────
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    algo_label  = "_".join(checkpoints.keys())
    stage_label = "_".join(stages)
    run_dir     = BASE_OUTPUT_DIR / f"{timestamp}_{algo_label}_{stage_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  🎬 SAR UAV VISUALIZATION")
    print(f"{'═'*65}")
    print(f"  Algos      : {list(checkpoints.keys())}")
    print(f"  Stages     : {stages}")
    print(f"  Episodes   : {args.n_episodes} per (algo × stage)")
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  Device     : {device}")
    print(f"  FPS        : {args.fps}")
    print(f"  Upload HF  : {'✅ Yes' if uploader else '❌ No'}")
    print(f"  Output     : {run_dir.resolve()}")
    print(f"{'═'*65}")

    # ── Load actors (1 lần, dùng lại cho cả 2 stages) ────────────────────────
    # Config chỉ dùng để lấy obs_dim (không đổi theo stage)
    base_cfg = AppConfig()
    base_cfg.apply_stage(STAGE_HARD)
    base_cfg.env.n_uav = args.n_uav
    base_cfg.viz_mode  = args.mode

    actors: Dict[str, Tuple[torch.nn.Module, int, str]] = {}
    print(f"\n  📦 Loading actors...")
    for algo, ckpt_path in checkpoints.items():
        try:
            actor, ep, actual_algo = load_actor(
                ckpt_path, algo, base_cfg, device
            )
            actors[actual_algo] = (actor, ep, actual_algo)
        except Exception as e:
            print(f"  ❌ {algo.upper()} load failed: {e}")

    if not actors:
        print("❌ Không load được actor nào!")
        return

    # ── Main loop: stage × algo × episodes ───────────────────────────────────
    # Lưu kết quả
    all_results: Dict[str, Dict[str, List[dict]]] = {}
    # {algo: {stage: [result_dict, ...]}}

    gif_files: Dict[str, Path] = {}   # {"mappo_hard": Path, ...}
    png_files: Dict[str, Path] = {}   # {"mappo_hard_ep0": Path, ...}

    for stage in stages:
        stage_cfg = AppConfig()
        stage_cfg.apply_stage(STAGE_MAP[stage])
        stage_cfg.env.n_uav = args.n_uav
        stage_cfg.viz_mode  = args.mode

        if args.max_steps is not None:
            stage_cfg.env.max_steps = args.max_steps

        print(f"\n{'─'*65}")
        print(f"  📍 STAGE: {stage.upper()} | "
              f"map={stage_cfg.env.map_size}×{stage_cfg.env.map_size}m | "
              f"max_steps={stage_cfg.env.max_steps}")
        print(f"{'─'*65}")

        for algo, (actor, trained_ep, actual_algo) in actors.items():
            all_results.setdefault(algo, {})
            all_results[algo][stage] = []

            algo_stage_frames = []   # Tổng hợp frames của algo+stage này

            for ep_i in range(args.n_episodes):
                ep_seed = args.seed + ep_i * 13 + hash(stage) % 100

                frames, result = run_episode(
                    config      = stage_cfg,
                    seed        = ep_seed,
                    actor       = actor,
                    algo        = algo,
                    device      = device,
                    run_dir     = run_dir,
                    ep_idx      = ep_i,
                    stage       = stage,
                    save_frames = not args.no_frames,
                )

                all_results[algo][stage].append(result)
                algo_stage_frames.extend(frames)

                # Lưu GIF từng episode
                if not args.no_gif and frames:
                    ep_gif_path = run_dir / f"{algo}_{stage}_ep{ep_i:02d}.gif"
                    save_gif_file(frames, ep_gif_path, fps=args.fps)

            # Lưu GIF tổng hợp cho algo + stage này
            if not args.no_gif and algo_stage_frames:
                combo_key  = f"{algo}_{stage}"
                combo_gif  = run_dir / f"{combo_key}_all.gif"
                print(f"\n  💾 Saving {combo_key} GIF "
                      f"({len(algo_stage_frames)} frames)...")
                ok = save_gif_file(algo_stage_frames, combo_gif, fps=args.fps)
                if ok:
                    gif_files[combo_key] = combo_gif

                # Lưu latest
                latest_gif = BASE_OUTPUT_DIR / f"latest_{combo_key}.gif"
                try:
                    if latest_gif.exists() or latest_gif.is_symlink():
                        latest_gif.unlink()
                    latest_gif.symlink_to(combo_gif.resolve())
                except Exception:
                    shutil.copy2(str(combo_gif), str(latest_gif))

            # Lưu frame cuối → PNG
            if algo_stage_frames:
                last_png = run_dir / f"{algo}_{stage}_last_frame.png"
                save_frame_png(
                    algo_stage_frames[-1], last_png,
                    title=f"{algo.upper()} | {stage.upper()} | Last Frame",
                )
                png_files[f"{algo}_{stage}_last"] = last_png

    # ── Summary plot ──────────────────────────────────────────────────────────
    print(f"\n  📊 Generating summary plot...")
    summary_path = save_summary_plot(
        results = all_results,
        run_dir = run_dir,
        stages  = stages,
        algos   = list(actors.keys()),
    )

    # ── Print tổng kết ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═'*65}")
    print(f"  📊 RESULTS SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Algo':<8} {'Stage':<10} "
          f"{'Reward':>10} {'Coverage':>10} "
          f"{'Victims':>10} {'Success':>10}")
    print(f"  {'-'*58}")

    for algo in actors:
        for stage in stages:
            eps = all_results.get(algo, {}).get(stage, [])
            if not eps:
                continue
            mean_rew = np.mean([r["ep_reward"] for r in eps])
            mean_cov = np.mean([r["coverage"]  for r in eps])
            mean_vic = np.mean([r["victims_f"] for r in eps])
            mean_suc = np.mean([r["success"]   for r in eps]) * 100
            print(
                f"  {algo.upper():<8} {stage.upper():<10} "
                f"{mean_rew:>+10.1f} {mean_cov:>9.1f}% "
                f"{mean_vic:>10.1f} {mean_suc:>9.1f}%"
            )

    print(f"\n  ⏱️  Total time: {elapsed:.1f}s")
    print(f"  📁 Output   : {run_dir.resolve()}")

    # ── Upload HF ─────────────────────────────────────────────────────────────
    if uploader is not None:
        meta = {
            "timestamp":  timestamp,
            "algos":      list(actors.keys()),
            "stages":     stages,
            "n_episodes": args.n_episodes,
            "device":     device,
            "fps":        args.fps,
            "results":    {
                algo: {
                    stage: [
                        {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in r.items()}
                        for r in eps_list
                    ]
                    for stage, eps_list in stage_dict.items()
                }
                for algo, stage_dict in all_results.items()
            },
            "checkpoints": {
                algo: str(checkpoints.get(algo, ""))
                for algo in actors
            },
        }

        uploader.upload_run_results(
            run_dir   = run_dir,
            gif_files = gif_files,
            png_files = png_files,
            summary   = summary_path,
            meta      = meta,
        )

    print(f"\n{'═'*65}")
    print(f"  ✅ Done!")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()