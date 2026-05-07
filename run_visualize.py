#!/usr/bin/env python3
"""
run_visualization.py
CLI entry point cho visualization — AUTO SAVE VERSION

Output luôn được lưu tại:
    results/viz/
    ├── latest.gif          ← GIF mới nhất (luôn overwrite)
    ├── latest.png          ← Frame cuối cùng
    └── {timestamp}/
        ├── episode.gif     ← GIF đầy đủ
        ├── summary.png     ← Plot summary
        └── frames/
            ├── frame_0001.png
            └── ...

Usage:
    python run_visualization.py --mode 2d
    python run_visualization.py --mode 2d --policy circle
    python run_visualization.py --mode 2d --policy untrained --algo mappo
    python run_visualization.py --mode 2d --checkpoint results/mappo/.../checkpoint_final.pt --algo mappo
"""

import argparse
import os
import time
import numpy as np
import torch
from pathlib import Path

from config import AppConfig, STAGE_HARD

# ══════════════════════════════════════════════════════════════════════════════
# FIXED OUTPUT DIR — Luôn lưu vào đây
# ══════════════════════════════════════════════════════════════════════════════
BASE_OUTPUT_DIR = Path("results/viz")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SAR UAV Policy Visualizer")

    p.add_argument("--mode", type=str, default="2d",
                   choices=["2d", "3d"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (None = scripted/untrained policy)")
    p.add_argument("--algo", type=str, default="mappo",
                   choices=["mappo", "masac", "matd3"])
    p.add_argument("--policy", type=str, default="random",
                   choices=["random", "hover", "circle", "untrained"])
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--n-episodes",  type=int,   default=1)
    p.add_argument("--max-steps",   type=int,   default=None)
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--n-uav",       type=int,   default=4)
    p.add_argument("--fps",         type=int,   default=10,
                   help="FPS cho GIF output")
    p.add_argument("--no-gif",      action="store_true",
                   help="Không tạo GIF (chỉ lưu frames PNG)")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT DIR SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_output_dir(policy_label: str) -> Path:
    """
    Tạo thư mục output với timestamp.
    Luôn tạo thư mục mới mỗi lần chạy.
    
    Returns:
        run_dir: Path  ← Thư mục lưu toàn bộ output của run này
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Sanitize policy_label cho tên thư mục
    safe_label = policy_label.replace(" ", "_") \
                             .replace("(", "") \
                             .replace(")", "") \
                             .replace("—", "") \
                             .replace(",", "") \
                             .strip("_")
    safe_label = safe_label[:40]  # Giới hạn độ dài

    run_dir = BASE_OUTPUT_DIR / f"{timestamp}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = run_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    return run_dir


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPTED POLICIES
# ══════════════════════════════════════════════════════════════════════════════

def get_scripted_action(policy: str, step: int, uav_id: int, n_agents: int) -> np.ndarray:
    if policy == "random":
        act    = np.random.uniform(-1, 1, 4)
        act[3] = 0.0
        return act
    elif policy == "hover":
        return np.zeros(4, dtype=np.float32)
    elif policy == "circle":
        phase = (2 * np.pi * uav_id) / n_agents
        t     = step * 0.05 + phase
        return np.array([np.cos(t)*0.6, np.sin(t)*0.6, 0.0, 0.0], dtype=np.float32)
    return np.zeros(4, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ACTOR LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_actor(checkpoint_path: str, algo: str, config: AppConfig, device: str):
    obs_dim    = config.obs.actor_dim
    action_dim = 4

    if algo == "mappo":
        from training.algorithms.mappo.actor import ActorNetwork
        tr    = config.train
        actor = ActorNetwork(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_dims    = tr.mappo_actor_hidden,
            activation     = tr.mappo_activation,
            use_layer_norm = tr.mappo_use_layer_norm,
        )
    elif algo == "masac":
        from training.algorithms.masac.actor import SACActorNetwork
        actor = SACActorNetwork(obs_dim=obs_dim, action_dim=action_dim)
    elif algo == "matd3":
        from training.algorithms.matd3.actor import TD3ActorNetwork
        actor = TD3ActorNetwork(obs_dim=obs_dim, action_dim=action_dim)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # ✅ FIX: weights_only=False cho PyTorch >= 2.6
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    actor.to(device)

    ep = ckpt.get("total_episodes_done", ckpt.get("episode", 0))
    print(f"✅ Loaded {algo.upper()} actor — trained {ep:,} episodes")
    return actor, ep


def create_untrained_actor(algo: str, config: AppConfig, device: str):
    obs_dim    = config.obs.actor_dim
    action_dim = 4

    if algo == "mappo":
        from training.algorithms.mappo.actor import ActorNetwork
        tr    = config.train
        actor = ActorNetwork(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_dims    = tr.mappo_actor_hidden,
            activation     = tr.mappo_activation,
            use_layer_norm = tr.mappo_use_layer_norm,
        )
    elif algo == "masac":
        from training.algorithms.masac.actor import SACActorNetwork
        actor = SACActorNetwork(obs_dim=obs_dim, action_dim=action_dim)
    elif algo == "matd3":
        from training.algorithms.matd3.actor import TD3ActorNetwork
        actor = TD3ActorNetwork(obs_dim=obs_dim, action_dim=action_dim)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    actor.eval()
    actor.to(device)
    print(f"🎲 Created UNTRAINED {algo.upper()} actor (random weights)")
    return actor


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


def save_gif_file(frames: list, path: Path, fps: int = 10):
    """Lưu danh sách frames thành GIF."""
    if not frames:
        print("  ⚠️  No frames to save")
        return False

    # Method 1: Pillow (ưu tiên)
    try:
        from PIL import Image
        imgs = []
        for f in frames:
            if f is not None:
                imgs.append(Image.fromarray(f.astype(np.uint8)))

        if imgs:
            imgs[0].save(
                str(path),
                save_all      = True,
                append_images = imgs[1:],
                duration      = 1000 // fps,
                loop          = 0,
                optimize      = False,
            )
            print(f"  ✅ GIF saved ({len(imgs)} frames, {fps}fps)")
            return True
    except ImportError:
        print("  ⚠️  Pillow not found — trying imageio...")
    except Exception as e:
        print(f"  ⚠️  Pillow GIF failed: {e} — trying imageio...")

    # Method 2: imageio (fallback)
    try:
        import imageio
        imageio.mimsave(
            str(path),
            [f.astype(np.uint8) for f in frames if f is not None],
            fps=fps,
        )
        print(f"  ✅ GIF saved via imageio ({len(frames)} frames)")
        return True
    except ImportError:
        print("  ⚠️  imageio not found")
    except Exception as e:
        print(f"  ⚠️  imageio failed: {e}")

    # Method 3: matplotlib animation (last resort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")

        valid = [f for f in frames if f is not None]
        im    = ax.imshow(valid[0])

        def update(i):
            im.set_data(valid[i])
            return [im]

        ani = animation.FuncAnimation(
            fig, update,
            frames=len(valid),
            interval=1000//fps,
            blit=True,
        )
        ani.save(str(path), writer="pillow", fps=fps)
        plt.close(fig)
        print(f"  ✅ GIF saved via matplotlib ({len(valid)} frames)")
        return True
    except Exception as e:
        print(f"  ⚠️  matplotlib animation failed: {e}")

    return False


def save_summary_plot(results: list, run_dir: Path, policy_label: str):
    """Lưu summary plot PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(results)
        if n == 0:
            return

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"Visualization Summary — {policy_label}",
            fontsize=13, fontweight="bold"
        )

        metrics = {
            "Reward":   [r["ep_reward"]  for r in results],
            "Coverage": [r["coverage"]   for r in results],
            "Victims":  [r["victims_f"]  for r in results],
            "Steps":    [r["ep_steps"]   for r in results],
        }

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

        for ax, (metric, values), color in zip(axes, metrics.items(), colors):
            if n == 1:
                ax.bar([0], values, color=color, alpha=0.8, edgecolor="white")
                ax.set_xticks([0])
                ax.set_xticklabels(["Ep 1"])
            else:
                ax.bar(range(n), values, color=color, alpha=0.8, edgecolor="white")
                ax.set_xticks(range(n))
                ax.set_xticklabels([f"Ep{i+1}" for i in range(n)])
                ax.axhline(
                    np.mean(values), color="red",
                    linestyle="--", linewidth=1.5,
                    label=f"Mean: {np.mean(values):.1f}"
                )
                ax.legend(fontsize=8)

            ax.set_title(metric, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.set_facecolor("#F5F5F5")

        plt.tight_layout()
        out = run_dir / "summary.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Summary plot: {out.resolve()}")

    except Exception as e:
        print(f"  ⚠️  Summary plot failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN EPISODE
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    config:      AppConfig,
    seed:        int,
    policy_type: str,
    policy_arg,
    algo:        str,
    device:      str,
    run_dir:     Path,
    ep_idx:      int,
):
    """
    Chạy 1 episode, lưu frames tự động.
    
    Returns:
        frames: list[np.ndarray]
        result: dict
    """
    from env_setup.sar_pettingzoo_env import SARPettingZooEnv

    # ✅ Luôn dùng rgb_array để capture frames (không cần GUI)
    env      = SARPettingZooEnv(config, render_mode="rgb_array")
    n_agents = config.env.n_uav
    obs_dim  = config.obs.actor_dim

    obs_d, info = env.reset(seed=seed)

    ep_reward  = 0.0
    ep_steps   = 0
    frames     = []
    done       = False

    frames_dir = run_dir / "frames" / f"ep{ep_idx:02d}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    policy_label = policy_arg if policy_type == "scripted" else f"{algo.upper()}_actor"
    print(f"\n  🚁 Policy: [{policy_label}] | seed={seed}")
    print(f"  {'─'*50}")

    while not done:
        # ── Get actions ───────────────────────────────────────────────────
        if policy_type == "scripted":
            act_dict = {
                f"uav_{i}": get_scripted_action(policy_arg, ep_steps, i, n_agents)
                for i in range(n_agents)
            }
        else:
            actor   = policy_arg
            obs_arr = np.array(
                [obs_d.get(f"uav_{i}", np.zeros(obs_dim, np.float32))
                 for i in range(n_agents)],
                dtype=np.float32,
            )
            obs_t = torch.FloatTensor(obs_arr).to(device)

            with torch.no_grad():
                if algo in ("mappo", "masac"):
                    act_t, _ = actor.get_action(obs_t, deterministic=True)
                elif algo == "matd3":
                    act_t, _ = actor.get_action(obs_t, explore_noise=0.0, deterministic=True)

            act_np   = np.clip(act_t.cpu().numpy(), -1.0, 1.0)
            act_dict = {f"uav_{i}": act_np[i] for i in range(n_agents)}

        # ── Step ──────────────────────────────────────────────────────────
        obs_d, rew_d, term_d, trunc_d, info = env.step(act_dict)

        ep_reward += sum(rew_d.values())
        ep_steps  += 1
        done       = any(term_d.values()) or any(trunc_d.values())

        # ── Capture frame ─────────────────────────────────────────────────
        frame = env.render()  # rgb_array → np.ndarray
        if frame is not None:
            frames.append(frame.copy())

            # Lưu từng frame PNG
            frame_path = frames_dir / f"frame_{ep_steps:04d}.png"
            save_frame_png(
                frame, frame_path,
                title=f"Step {ep_steps}/{config.env.max_steps}",
            )

        # ── Print progress ────────────────────────────────────────────────
        if ep_steps % 50 == 0 or done:
            u0     = info.get("uav_0", {})
            cov    = u0.get("coverage_rate", 0.0) * 100
            vf     = u0.get("victims_found",  0)
            vt     = u0.get("victims_total",  1)
            na     = u0.get("n_active",       0)
            nc     = u0.get("n_charging",     0)
            nd     = u0.get("n_disabled",     0)
            bat    = u0.get("battery_stats",  {}).get("mean", 0.0)

            print(
                f"  Step {ep_steps:3d}/{config.env.max_steps}"
                f" | cov={cov:5.1f}%"
                f" | vic={vf}/{vt}"
                f" | rew={ep_reward:+7.1f}"
                f" | {na}act {nc}chg {nd}dis"
                f" | bat={bat:.0f}%"
                f" | frames={len(frames)}"
            )

    # ── Episode summary ───────────────────────────────────────────────────
    u0          = info.get("uav_0", {})
    ep_metrics  = u0.get("episode", {})
    coverage    = float(ep_metrics.get("coverage_rate",   u0.get("coverage_rate",   0.0)))
    victims_f   = int(ep_metrics.get("victims_found",     u0.get("victims_found",   0)))
    victims_t   = int(ep_metrics.get("total_victims",     u0.get("victims_total",   1)))
    success     = bool(ep_metrics.get("success",          u0.get("success",         False)))
    done_reason = ep_metrics.get("done_reason",           u0.get("done_reason",     "unknown"))

    env.close()

    print(f"\n  {'─'*50}")
    print(f"  📊 Episode {ep_idx+1} Done")
    print(f"     Steps    : {ep_steps}/{config.env.max_steps}")
    print(f"     Reward   : {ep_reward:+.1f}")
    print(f"     Coverage : {coverage*100:.1f}%")
    print(f"     Victims  : {victims_f}/{victims_t}")
    print(f"     Success  : {'✓ YES' if success else '✗ NO'}")
    print(f"     Reason   : {done_reason}")
    print(f"     Frames   : {len(frames)} → {frames_dir.resolve()}")

    return frames, {
        "ep_reward": ep_reward,
        "ep_steps":  ep_steps,
        "coverage":  coverage * 100,
        "victims_f": victims_f,
        "victims_t": victims_t,
        "success":   success,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    # ── Config ────────────────────────────────────────────────────────────
    cfg = AppConfig()
    cfg.apply_stage(STAGE_HARD)
    cfg.env.n_uav = args.n_uav
    cfg.viz_mode  = args.mode

    if args.max_steps is not None:
        cfg.env.max_steps = args.max_steps

    # ── Policy setup ──────────────────────────────────────────────────────
    if args.checkpoint:
        actor, trained_ep = load_actor(args.checkpoint, args.algo, cfg, device)
        policy_type  = "actor"
        policy_arg   = actor
        policy_label = f"{args.algo.upper()}_trained_{trained_ep}eps"

    elif args.policy == "untrained":
        actor        = create_untrained_actor(args.algo, cfg, device)
        policy_type  = "actor"
        policy_arg   = actor
        policy_label = f"{args.algo.upper()}_untrained"

    else:
        policy_type  = "scripted"
        policy_arg   = args.policy
        policy_label = args.policy

    # ── Setup output dir ──────────────────────────────────────────────────
    run_dir = setup_output_dir(policy_label)

    # ── Print header ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"🎬 SAR UAV VISUALIZATION")
    print(f"{'='*65}")
    print(f"  render mode : {args.mode.upper()}")
    print(f"  policy      : {policy_label}")
    print(f"  n_episodes  : {args.n_episodes}")
    print(f"  max_steps   : {cfg.env.max_steps}")
    print(f"  map_size    : {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"  n_uav       : {cfg.env.n_uav}")
    print(f"  device      : {device}")
    print(f"  fps         : {args.fps}")
    print(f"")
    print(f"  📁 Output dir : {run_dir.resolve()}")
    print(f"{'='*65}")

    # ── Run episodes ──────────────────────────────────────────────────────
    all_frames  = []
    all_results = []

    for ep_i in range(args.n_episodes):
        print(f"\n{'─'*65}")
        print(f"  Episode {ep_i+1}/{args.n_episodes}")

        frames, result = run_episode(
            config      = cfg,
            seed        = args.seed + ep_i,
            policy_type = policy_type,
            policy_arg  = policy_arg,
            algo        = args.algo,
            device      = device,
            run_dir     = run_dir,
            ep_idx      = ep_i,
        )

        all_frames.extend(frames)
        all_results.append(result)

        # Lưu GIF riêng cho từng episode
        if not args.no_gif and frames:
            ep_gif = run_dir / f"ep{ep_i:02d}.gif"
            print(f"\n  💾 Saving episode GIF...")
            save_gif_file(frames, ep_gif, fps=args.fps)

    # ── Lưu GIF tổng hợp tất cả episodes ─────────────────────────────────
    if not args.no_gif and all_frames:
        all_gif = run_dir / "all_episodes.gif"
        print(f"\n  💾 Saving combined GIF ({len(all_frames)} frames)...")
        save_gif_file(all_frames, all_gif, fps=args.fps)

        # Symlink → latest.gif (dễ tìm)
        latest = BASE_OUTPUT_DIR / "latest.gif"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(all_gif.resolve())
        except Exception:
            # Windows không support symlink → copy thay thế
            import shutil
            shutil.copy2(str(all_gif), str(latest))

    # ── Lưu frame cuối → latest.png ───────────────────────────────────────
    if all_frames:
        latest_png = BASE_OUTPUT_DIR / "latest.png"
        save_frame_png(
            all_frames[-1],
            latest_png,
            title=f"Last frame — {policy_label}",
        )

    # ── Summary plot ──────────────────────────────────────────────────────
    save_summary_plot(all_results, run_dir, policy_label)

    # ── Overall summary ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"📊 OVERALL SUMMARY ({args.n_episodes} episode(s))")
    print(f"{'='*65}")

    rewards   = [r["ep_reward"]  for r in all_results]
    coverages = [r["coverage"]   for r in all_results]
    successes = [r["success"]    for r in all_results]

    print(f"  Reward   : {np.mean(rewards):+.1f}" +
          (f" ± {np.std(rewards):.1f}" if len(rewards) > 1 else ""))
    print(f"  Coverage : {np.mean(coverages):.1f}%" +
          (f" ± {np.std(coverages):.1f}%" if len(coverages) > 1 else ""))
    print(f"  Success  : {sum(successes)}/{args.n_episodes}")
    print(f"  Frames   : {len(all_frames)} total")

    # ── Hiển thị đường dẫn cụ thể ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"📁 OUTPUT FILES")
    print(f"{'='*65}")
    print(f"  Run dir    : {run_dir.resolve()}")
    print(f"  Latest GIF : {(BASE_OUTPUT_DIR / 'latest.gif').resolve()}")
    print(f"  Latest PNG : {(BASE_OUTPUT_DIR / 'latest.png').resolve()}")
    print(f"")
    print(f"  Cấu trúc:")

    for f in sorted(run_dir.rglob("*"))[:20]:  # Show tối đa 20 files
        rel = f.relative_to(run_dir)
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size/1024:.0f}KB"
            else:
                size_str = f"{size}B"
            print(f"    {rel}  ({size_str})")

    n_files = sum(1 for f in run_dir.rglob("*") if f.is_file())
    if n_files > 20:
        print(f"    ... và {n_files - 20} files nữa")

    print(f"\n{'='*65}")
    print(f"✅ Done!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()