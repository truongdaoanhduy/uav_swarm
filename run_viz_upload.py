#!/usr/bin/env python3
"""
run_visualization.py
3 algo × 2 stages (transfer + extreme).
Upload lên HuggingFace NGAY sau khi hoàn thành từng (algo × stage).
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
from config.curriculum_config import STAGE_HARD, STAGE_TRANSFER, STAGE_EXTREME

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BASE_OUTPUT_DIR = Path("results/viz")

STAGE_MAP = {
    "transfer": STAGE_TRANSFER,
    "extreme":  STAGE_EXTREME,
}

ALGO_COLORS = {
    "mappo": "#2196F3",
    "masac": "#4CAF50",
    "matd3": "#FF9800",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SAR UAV Visualizer — upload ngay sau mỗi algo×stage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mappo", type=str, default=None)
    p.add_argument("--masac", type=str, default=None)
    p.add_argument("--matd3", type=str, default=None)

    p.add_argument(
        "--stages", nargs="+",
        default=["transfer", "extreme"],
        choices=["transfer", "extreme"],
    )

    p.add_argument("--mode",       type=str, default="2d",
                   choices=["2d", "3d"])
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-steps",  type=int, default=None)
    p.add_argument("--fps",        type=int, default=10)
    p.add_argument("--no-gif",     action="store_true")
    p.add_argument("--no-frames",  action="store_true")

    p.add_argument("--hf-token",  type=str, default=None,
                   help="HuggingFace API token")
    p.add_argument("--hf-repo",   type=str, default=None,
                   help="HF repo ID, vd: username/sar-uav-viz")
    p.add_argument("--no-upload", action="store_true")

    p.add_argument("--seed",   type=int, default=44)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n-uav",  type=int, default=4)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE UPLOADER — upload từng file ngay lập tức
# ══════════════════════════════════════════════════════════════════════════════

class VizHFUploader:
    """
    Upload file lên HuggingFace NGAY khi được gọi.
    Không queue, không batch — upload 1 file = 1 commit.
    """

    def __init__(self, token: str, repo_id: str, timestamp: str):
        self.token     = token
        self.repo_id   = repo_id
        self.timestamp = timestamp  # dùng làm folder name trong HF
        self._api      = None
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
            print(f"  ✅ HF repo: {self.repo_id}")
        except ImportError:
            print("  ❌ pip install huggingface-hub")
        except Exception as e:
            print(f"  ⚠️  HF init: {e}")

    def _upload_one(self, local: str, remote: str, msg: str) -> bool:
        """Upload 1 file, retry 1 lần nếu thất bại."""
        if self._api is None or not Path(local).exists():
            return False
        for attempt in range(2):
            try:
                self._api.upload_file(
                    path_or_fileobj = local,
                    path_in_repo    = remote,
                    repo_id         = self.repo_id,
                    repo_type       = "dataset",
                    commit_message  = msg,
                )
                print(f"  ☁️  → {remote}")
                return True
            except Exception as e:
                if attempt == 0:
                    print(f"  ⚠️  Retry upload {Path(local).name}: {e}")
                    time.sleep(2)
                else:
                    print(f"  ❌ Upload failed {Path(local).name}: {e}")
        return False

    def upload_algo_stage_result(
        self,
        algo:       str,
        stage:      str,
        gif_path:   Optional[Path],
        png_path:   Optional[Path],
        result:     dict,
        run_dir:    Path,
    ) -> bool:
        """
        Upload kết quả của 1 (algo × stage) NGAY sau khi chạy xong.

        Upload:
            1. GIF animation
            2. Last frame PNG
            3. result.json (metrics)

        Cấu trúc HF:
            visualizations/
            ├── {timestamp}/
            │   ├── transfer/
            │   │   ├── mappo.gif
            │   │   ├── mappo_last.png
            │   │   ├── mappo_result.json
            │   │   ├── masac.gif  ← upload ngay sau khi masac xong
            │   │   └── ...
            │   └── extreme/
            │       └── ...
            └── latest/
                ├── transfer_mappo.gif   ← overwrite
                └── ...
        """
        key     = f"{algo}_{stage}"
        base_ts = f"visualizations/{self.timestamp}/{stage}"
        base_lt = f"visualizations/latest"

        print(f"\n  📤 Uploading {algo.upper()}×{stage.upper()} → HF...")

        uploaded = 0
        total    = 0

        # ── 1. GIF ────────────────────────────────────────────────────────────
        if gif_path and gif_path.exists():
            total += 1
            ok = self._upload_one(
                str(gif_path),
                f"{base_ts}/{algo}.gif",
                f"GIF {algo.upper()} on {stage.upper()}",
            )
            # latest/ (overwrite mỗi lần)
            self._upload_one(
                str(gif_path),
                f"{base_lt}/{key}.gif",
                f"Latest GIF {key}",
            )
            uploaded += ok

        # ── 2. Last frame PNG ─────────────────────────────────────────────────
        if png_path and png_path.exists():
            total += 1
            ok = self._upload_one(
                str(png_path),
                f"{base_ts}/{algo}_last.png",
                f"Last frame {algo.upper()} on {stage.upper()}",
            )
            self._upload_one(
                str(png_path),
                f"{base_lt}/{key}_last.png",
                f"Latest PNG {key}",
            )
            uploaded += ok

        # ── 3. Result JSON ────────────────────────────────────────────────────
        json_path = run_dir / f"{key}_result.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=_json_safe)
        total += 1
        ok = self._upload_one(
            str(json_path),
            f"{base_ts}/{algo}_result.json",
            f"Result {algo.upper()} on {stage.upper()}",
        )
        uploaded += ok

        status = "✅" if uploaded == total else f"⚠️ {uploaded}/{total}"
        print(f"  {status} Upload {algo.upper()}×{stage.upper()} done")
        return uploaded == total

    def upload_summary(
        self,
        summary_path: Optional[Path],
        meta:         dict,
        run_dir:      Path,
    ):
        """Upload summary plot + meta.json sau khi tất cả xong."""
        print(f"\n  📤 Uploading final summary → HF...")

        base_ts = f"visualizations/{self.timestamp}"

        # Summary plot
        if summary_path and summary_path.exists():
            self._upload_one(
                str(summary_path),
                f"{base_ts}/summary_comparison.png",
                "Summary comparison plot",
            )
            self._upload_one(
                str(summary_path),
                "visualizations/latest/summary_comparison.png",
                "Latest summary",
            )

        # Meta JSON
        meta_path = run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=_json_safe)
        self._upload_one(
            str(meta_path),
            f"{base_ts}/meta.json",
            "Run metadata",
        )

        print(f"  🔗 https://huggingface.co/datasets/{self.repo_id}"
              f"/tree/main/visualizations/{self.timestamp}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _json_safe(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, bool):        return bool(obj)
    return str(obj)


def detect_algo_from_checkpoint(ckpt: dict) -> str:
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
    meta = ckpt.get("algo", ckpt.get("algorithm", "")).lower()
    return meta if meta in ("mappo", "masac", "matd3") else "unknown"


def load_actor(
    checkpoint_path: str,
    algo:            str,
    config:          AppConfig,
    device:          str,
) -> Tuple[torch.nn.Module, int, str]:
    obs_dim    = config.obs.actor_dim
    action_dim = 4
    tr         = config.train

    print(f"\n  📂 Loading: {Path(checkpoint_path).name}")
    ckpt = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    detected = detect_algo_from_checkpoint(ckpt)
    if detected != "unknown" and detected != algo:
        print(f"  ⚠️  detect={detected.upper()} (was --algo={algo})")
        algo = detected
    elif detected == "unknown":
        print(f"  ⚠️  Không detect được, dùng: {algo.upper()}")

    if algo == "mappo":
        from training.algorithms.mappo.actor import ActorNetwork
        actor = ActorNetwork(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_dims=tr.mappo_actor_hidden,
            activation=tr.mappo_activation,
            use_layer_norm=tr.mappo_use_layer_norm,
            log_std_init=-0.5,
        )
    elif algo == "masac":
        from training.algorithms.masac.actor import SACActorNetwork
        actor = SACActorNetwork(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_dims=tr.masac_actor_hidden,
        )
    elif algo == "matd3":
        from training.algorithms.matd3.actor import TD3ActorNetwork
        actor = TD3ActorNetwork(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_dims=tr.matd3_actor_hidden,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    actor_state = None
    for key in ["actor_state_dict", "actor", "model", "state_dict"]:
        if key in ckpt:
            actor_state = ckpt[key]
            break
    if actor_state is None:
        actor_state = ckpt

    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        print(f"  ⚠️  Missing ({len(missing)}): {missing[:2]}")
    if unexpected:
        print(f"  ⚠️  Unexpected ({len(unexpected)}): {unexpected[:2]}")

    actor.eval()
    actor.to(device)
    ep = ckpt.get("total_episodes_done", ckpt.get("episode", 0))
    print(f"  ✅ {algo.upper()} loaded | {ep:,} eps trained")
    return actor, ep, algo


def save_frame_png(frame: np.ndarray, path: Path, title: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Giảm kích thước cho 3D
        figsize = (10, 7) if frame.shape[0] > 800 else (12, 8)
        dpi = 72 if frame.shape[0] > 800 else 100
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10, pad=5)
        
        fig.tight_layout(pad=0.3)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        
    except Exception as e:
        print(f"  ⚠️  PNG save failed: {e}")


def save_gif_file(frames: list, path: Path, fps: int = 10) -> bool:
    """
    Lưu GIF với FPS động - TỰ ĐỘNG CHẬM LẠI
    """
    import gc
    
    valid = [f for f in frames if f is not None]
    if not valid:
        return False

    try:
        from PIL import Image
        
        # ═══════════════════════════════════════════════════════════
        # 🔥 FIX: BẮT BUỘC CHẬM LẠI CHO 3D VIZ
        # ═══════════════════════════════════════════════════════════
        # LUÔN dùng FPS thấp để dễ xem
        actual_fps = 2  # ← 2 FPS = 500ms/frame = CHẬM ĐỂ XEM
        
        # Thêm pause frames ở đầu và cuối
        first_frame = valid[0]
        last_frame = valid[-1]
        
        pause_frames = [first_frame] * 20  # Pause 10s ở đầu (20 frames × 0.5s)
        end_frames = [last_frame] * 40     # Pause 20s ở cuối (40 frames × 0.5s)
        
        all_frames = pause_frames + valid + end_frames
        
        total_duration = len(all_frames) / actual_fps
        # ═══════════════════════════════════════════════════════════
        
        optimize = len(all_frames) > 200
        
        print(f"  💾 Creating SLOW GIF:")
        print(f"      Frames: {len(all_frames)} ({len(valid)} + {len(pause_frames)+len(end_frames)} pause)")
        print(f"      FPS: {actual_fps} (500ms per frame)")
        print(f"      Duration: {total_duration:.1f}s (~{total_duration/60:.1f} min)")
        
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in all_frames]
        
        imgs[0].save(
            str(path), 
            save_all=True, 
            append_images=imgs[1:],
            duration=1000 // actual_fps,  # 500ms per frame
            loop=0, 
            optimize=optimize,
            quality=85 if optimize else 95,
        )
        
        # Cleanup
        imgs.clear()
        del imgs, all_frames, pause_frames, end_frames
        gc.collect()
        
        print(f"  ✅ GIF saved: {path.name}")
        print(f"      View duration: {total_duration:.1f}s")
        return True
        
    except Exception as e:
        print(f"  ⚠️  GIF creation failed: {e}")
        return False


def save_summary_plot(
    all_results: Dict,
    run_dir:     Path,
    stages:      List[str],
    algos:       List[str],
) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = [
            ("ep_reward", "Episode Reward",   False),
            ("coverage",  "Coverage (%)",     False),
            ("victims_f", "Victims Found",    False),
            ("success",   "Success Rate (%)", True),
        ]

        fig, axes = plt.subplots(
            len(metrics), len(stages),
            figsize=(6 * len(stages), 4 * len(metrics)),
            squeeze=False,
        )
        fig.suptitle(
            "SAR UAV — Visualization Summary\n"
            f"({' | '.join(a.upper() for a in algos)}) × "
            f"({' | '.join(s.upper() for s in stages)})",
            fontsize=13, fontweight="bold",
        )

        for col_i, stage in enumerate(stages):
            for row_i, (mkey, mlabel, is_pct) in enumerate(metrics):
                ax = axes[row_i][col_i]
                ax.set_title(f"{mlabel}\n[{stage.upper()}]",
                             fontweight="bold", fontsize=10)
                ax.set_facecolor("#F8F8F8")
                ax.grid(axis="y", alpha=0.3)

                for bar_i, algo in enumerate(algos):
                    eps = all_results.get(algo, {}).get(stage, [])
                    if not eps:
                        continue
                    vals = [r.get(mkey, 0) for r in eps]
                    if is_pct:
                        vals = [v * 100 if v <= 1 else v for v in vals]
                    mean = float(np.mean(vals))
                    std  = float(np.std(vals)) if len(vals) > 1 else 0.0

                    ax.bar(
                        bar_i, mean, width=0.6,
                        color=ALGO_COLORS.get(algo, "#999"),
                        alpha=0.85, edgecolor="white", linewidth=1.2,
                    )
                    if std > 0:
                        ax.errorbar(bar_i, mean, yerr=std,
                                    fmt="none", color="black", capsize=5)
                    ax.text(bar_i, mean + std + 0.5, f"{mean:.1f}",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

                ax.set_xticks(range(len(algos)))
                ax.set_xticklabels([a.upper() for a in algos])
                ax.set_ylabel(mlabel, fontsize=9)

        plt.tight_layout()
        out = run_dir / "summary_comparison.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Summary: {out.name}")
        return out
    except Exception as e:
        print(f"  ⚠️  Summary plot: {e}")
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
    from env_setup.sar_pettingzoo_env import SARPettingZooEnv
    import gc  # ← THÊM import


    # 🔥 FIX: GIẢM FRAME_SKIP ĐỂ MƯỢT HƠN
    FRAME_SKIP = 3 if config.viz_mode == "3d" else 2
    # Transfer: 3200 steps → 3200/6 = 533 frames
    # Extreme:  4000 steps → 4000/6 = 666 frames
# ═══════════════════════════════════════════════════════════════    # ═══════════════════════════════════════════════════════════════

    env      = SARPettingZooEnv(config, render_mode="rgb_array")
    n_agents = config.env.n_uav
    obs_dim  = config.obs.actor_dim

    obs_d, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_steps  = 0
    frames    = []
    done      = False

    frames_dir = run_dir / "frames" / f"{algo}_{stage}_ep{ep_idx:02d}"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  🚁 {algo.upper()} | {stage.upper()} "
          f"| ep={ep_idx+1} | seed={seed}")
    
    # Hiển thị frame skip nếu 3D
    if config.viz_mode == "3d":
        print(f"  ℹ️  3D mode: saving every {FRAME_SKIP} frames to prevent OOM")

    while not done:
        obs_arr = np.array(
            [obs_d.get(f"uav_{i}", np.zeros(obs_dim, np.float32))
             for i in range(n_agents)],
            dtype=np.float32,
        )
        obs_t = torch.FloatTensor(obs_arr).to(device)

        with torch.no_grad():
            if algo in ("mappo", "masac"):
                act_t, _ = actor.get_action(obs_t, deterministic=True)
            else:
                act_t, _ = actor.get_action(
                    obs_t, explore_noise=0.0, deterministic=True
                )

        act_np   = np.clip(act_t.cpu().numpy(), -1.0, 1.0)
        act_dict = {f"uav_{i}": act_np[i] for i in range(n_agents)}

        obs_d, rew_d, term_d, trunc_d, info = env.step(act_dict)
        ep_reward += sum(rew_d.values())
        ep_steps  += 1
        done       = any(term_d.values()) or any(trunc_d.values())

        # ═══════════════════════════════════════════════════════════════
        # 🔥 FIX: CHỈ RENDER VÀ LƯU MỖI FRAME_SKIP FRAMES
        # ═══════════════════════════════════════════════════════════════
        should_save = (ep_steps % FRAME_SKIP == 0) or done
        
        if should_save:
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
                if save_frames:
                    save_frame_png(
                        frame,
                        frames_dir / f"frame_{ep_steps:04d}.png",
                        title=(f"{algo.upper()} | {stage.upper()} "
                               f"| Step {ep_steps}/{config.env.max_steps}"),
                    )
                
                # Giải phóng RAM mỗi 100 frames
                if len(frames) % 100 == 0:
                    gc.collect()
        # ═══════════════════════════════════════════════════════════════

        if ep_steps % 100 == 0 or done:
            u0  = info.get("uav_0", {})
            cov = u0.get("coverage_rate", 0.0) * 100
            vf  = u0.get("victims_found", 0)
            vt  = u0.get("victims_total", 1)
            print(
                f"  step={ep_steps:4d}/{config.env.max_steps}"
                f" cov={cov:.1f}% vic={vf}/{vt}"
                f" rew={ep_reward:+.1f}"
            )

    u0         = info.get("uav_0", {})
    ep_metrics = u0.get("episode", {})
    coverage   = float(ep_metrics.get("coverage_rate", u0.get("coverage_rate", 0.0)))
    victims_f  = int(ep_metrics.get("victims_found",   u0.get("victims_found",  0)))
    victims_t  = int(ep_metrics.get("total_victims",   u0.get("victims_total",  1)))
    success    = bool(ep_metrics.get("success",        u0.get("success",        False)))
    done_reason = ep_metrics.get("done_reason",        u0.get("done_reason",    "?"))

    env.close()
    
    print(f"  ✅ ep done | rew={ep_reward:+.1f} "
          f"cov={coverage*100:.1f}% vic={victims_f}/{victims_t} "
          f"{'✓' if success else '✗'} [{done_reason}]")
    print(f"  📊 Total frames saved: {len(frames)} (from {ep_steps} steps)")

    return frames, {
        "ep_reward": float(ep_reward),
        "ep_steps":  int(ep_steps),
        "coverage":  float(coverage * 100),
        "victims_f": int(victims_f),
        "victims_t": int(victims_t),
        "success":   bool(success),
        "algo":      algo,
        "stage":     stage,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    t_start = time.time()

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    # ── Checkpoints ───────────────────────────────────────────────────────────
    raw_ckpts = {
        "mappo": args.mappo,
        "masac": args.masac,
        "matd3": args.matd3,
    }
    checkpoints = {
        algo: path
        for algo, path in raw_ckpts.items()
        if path and Path(path).exists()
    }
    for algo, path in raw_ckpts.items():
        if path and not Path(path).exists():
            print(f"  ⚠️  {algo}: không tồn tại: {path}")

    if not checkpoints:
        print("❌ Không có checkpoint hợp lệ!")
        return

    stages = args.stages

    # ── HF ───────────────────────────────────────────────────────────────────
    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    uploader  = None
    if hf_token and args.hf_repo and not args.no_upload:
        uploader = VizHFUploader(hf_token, args.hf_repo, timestamp)
    else:
        print("  ℹ️  Không upload HF")

    # ── Output dir ────────────────────────────────────────────────────────────
    algo_label  = "_".join(checkpoints.keys())
    stage_label = "_".join(stages)
    run_dir     = BASE_OUTPUT_DIR / f"{timestamp}_{algo_label}_{stage_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  🎬 SAR UAV VISUALIZATION")
    print(f"{'═'*65}")
    print(f"  Algos    : {list(checkpoints.keys())}")
    print(f"  Stages   : {stages}")
    print(f"  Episodes : {args.n_episodes} per (algo × stage)")
    print(f"  Upload   : {'✅ Ngay sau mỗi algo×stage' if uploader else '❌ No'}")
    print(f"  Output   : {run_dir.resolve()}")
    print(f"{'═'*65}")

    # ── Load actors ───────────────────────────────────────────────────────────
    base_cfg = AppConfig()
    base_cfg.apply_stage(STAGE_TRANSFER)  # chỉ để lấy obs_dim
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
            print(f"  ❌ {algo.upper()} failed: {e}")

    if not actors:
        print("❌ Không load được actor nào!")
        return

    # ── Tracking ──────────────────────────────────────────────────────────────
    all_results: Dict[str, Dict[str, List[dict]]] = {}
    gif_files:   Dict[str, Path] = {}
    png_files:   Dict[str, Path] = {}

    # ── MAIN LOOP: stage → algo → episodes → upload NGAY ─────────────────────
    for stage in stages:
        stage_cfg = AppConfig()
        stage_cfg.apply_stage(STAGE_MAP[stage])
        stage_cfg.env.n_uav = args.n_uav
        stage_cfg.viz_mode  = args.mode
        if args.max_steps is not None:
            stage_cfg.env.max_steps = args.max_steps

        print(f"\n{'═'*65}")
        print(f"  📍 STAGE: {stage.upper()} | "
              f"map={stage_cfg.env.map_size}×{stage_cfg.env.map_size}m | "
              f"steps={stage_cfg.env.max_steps}")
        print(f"{'═'*65}")

        for algo, (actor, trained_ep, actual_algo) in actors.items():
            all_results.setdefault(algo, {})
            all_results[algo][stage] = []

            print(f"\n  {'─'*55}")
            print(f"  🤖 {algo.upper()} × {stage.upper()} "
                  f"| {args.n_episodes} episode(s)")
            print(f"  {'─'*55}")

            algo_stage_frames = []

            # ── Episodes ──────────────────────────────────────────────────────
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

                # GIF từng episode (optional)
                if not args.no_gif and frames and args.n_episodes > 1:
                    ep_gif = run_dir / f"{algo}_{stage}_ep{ep_i:02d}.gif"
                    save_gif_file(frames, ep_gif, fps=args.fps)

            # ── Lưu GIF tổng hợp (algo × stage) ──────────────────────────────
            combo_key = f"{algo}_{stage}"
            gif_path  = None
            png_path  = None

            if not args.no_gif and algo_stage_frames:
                combo_gif = run_dir / f"{combo_key}_all.gif"
                print(f"\n  💾 Saving GIF: {combo_gif.name} "
                      f"({len(algo_stage_frames)} frames)...")
                ok = save_gif_file(algo_stage_frames, combo_gif, fps=args.fps)
                if ok:
                    gif_path              = combo_gif
                    gif_files[combo_key]  = combo_gif
                    # latest symlink
                    latest = BASE_OUTPUT_DIR / f"latest_{combo_key}.gif"
                    try:
                        if latest.exists() or latest.is_symlink():
                            latest.unlink()
                        latest.symlink_to(combo_gif.resolve())
                    except Exception:
                        shutil.copy2(str(combo_gif), str(latest))

            # Last frame PNG
            if algo_stage_frames:
                last_png = run_dir / f"{combo_key}_last_frame.png"
                save_frame_png(
                    algo_stage_frames[-1], last_png,
                    title=f"{algo.upper()} | {stage.upper()} | Last Frame",
                )
                png_path             = last_png
                png_files[combo_key] = last_png

            # ╔══════════════════════════════════════════════════════════════╗
            # ║  UPLOAD NGAY SAU KHI ALGO × STAGE XONG                     ║
            # ╚══════════════════════════════════════════════════════════════╝
            if uploader is not None:
                # Tổng hợp result cho combo này
                combo_result = {
                    "algo":       algo,
                    "stage":      stage,
                    "n_episodes": args.n_episodes,
                    "episodes":   all_results[algo][stage],
                    "mean": {
                        "reward":      float(np.mean([r["ep_reward"] for r in all_results[algo][stage]])),
                        "coverage":    float(np.mean([r["coverage"]  for r in all_results[algo][stage]])),
                        "victims_f":   float(np.mean([r["victims_f"] for r in all_results[algo][stage]])),
                        "success_rate":float(np.mean([r["success"]   for r in all_results[algo][stage]])),
                    },
                    "trained_episodes": trained_ep,
                    "timestamp":        timestamp,
                }
                uploader.upload_algo_stage_result(
                    algo     = algo,
                    stage    = stage,
                    gif_path = gif_path,
                    png_path = png_path,
                    result   = combo_result,
                    run_dir  = run_dir,
                )
            else:
                print(f"  ℹ️  (No HF upload)")

    # ── Summary plot (sau khi tất cả xong) ───────────────────────────────────
    print(f"\n  📊 Generating summary plot...")
    summary_path = save_summary_plot(
        all_results = all_results,
        run_dir     = run_dir,
        stages      = stages,
        algos       = list(actors.keys()),
    )

    # ── Print tổng kết ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═'*65}")
    print(f"  📊 FINAL RESULTS")
    print(f"{'═'*65}")
    print(f"  {'Algo':<8} {'Stage':<12} "
          f"{'Reward':>10} {'Coverage':>10} {'Success':>10}")
    print(f"  {'-'*52}")
    for algo in actors:
        for stage in stages:
            eps = all_results.get(algo, {}).get(stage, [])
            if not eps:
                continue
            print(
                f"  {algo.upper():<8} {stage.upper():<12} "
                f"{np.mean([r['ep_reward'] for r in eps]):>+10.1f} "
                f"{np.mean([r['coverage']  for r in eps]):>9.1f}% "
                f"{np.mean([r['success']   for r in eps])*100:>9.1f}%"
            )

    print(f"\n  ⏱️  Total: {elapsed:.1f}s")
    print(f"  📁 Output: {run_dir.resolve()}")

    # ── Upload summary (lần cuối) ─────────────────────────────────────────────
    if uploader is not None:
        meta = {
            "timestamp":  timestamp,
            "algos":      list(actors.keys()),
            "stages":     stages,
            "n_episodes": args.n_episodes,
            "elapsed_s":  round(elapsed, 1),
            "results":    {
                f"{algo}_{stage}": all_results.get(algo, {}).get(stage, [])
                for algo in actors
                for stage in stages
            },
        }
        uploader.upload_summary(summary_path, meta, run_dir)

    print(f"\n{'═'*65}")
    print(f"  ✅ Done!")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()