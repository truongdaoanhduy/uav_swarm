"""
examples/run_3d_demo.py
Demo 3D visualization với Panda3D.

MODES:
    --mode interactive   → Cửa sổ 3D real-time
    --mode screenshot    → Lưu PNG từng step
    --mode video         → Record MP4 video

USAGE:
    python examples/run_3d_demo.py --mode interactive
    python examples/run_3d_demo.py --mode screenshot --steps 50
    python examples/run_3d_demo.py --mode video --episodes 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import AppConfig
from config.curriculum_config import STAGE_EASY
from env_setup.base_env import SARBaseEnv


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SAR UAV Swarm — Panda3D 3D Demo")
    
    p.add_argument(
        "--mode",
        choices=["interactive", "screenshot", "video"],
        default="interactive",
        help="Visualization mode"
    )
    p.add_argument("--steps",    type=int,   default=100,  help="Max steps per episode")
    p.add_argument("--episodes", type=int,   default=1,    help="Number of episodes")
    p.add_argument("--seed",     type=int,   default=42,   help="Random seed")
    p.add_argument("--follow",   type=int,   default=None, help="Follow UAV ID (None=overview)")
    p.add_argument("--fps",      type=int,   default=30,   help="Target FPS")
    p.add_argument("--width",    type=int,   default=1280, help="Window width")
    p.add_argument("--height",   type=int,   default=720,  help="Window height")
    
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("  SAR UAV SWARM — PANDA3D 3D VISUALIZATION")
    print("="*70)
    print(f"  Mode:     {args.mode}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.steps}")
    print("="*70 + "\n")
    
    # ── Build config ─────────────────────────────────────────────────────
    cfg = AppConfig()
    cfg.apply_stage(STAGE_EASY)  # Dùng EASY stage cho demo
    
    # Override max_steps
    cfg.env.max_steps = args.steps
    
    # 3D viz config
    cfg.viz_3d_cfg = {
        "window_width":      args.width,
        "window_height":     args.height,
        "follow_uav_id":     args.follow,
        "cam_distance":      cfg.env.map_size * 1.0,  # Auto-scale
        "cam_elevation_deg": 45.0,
        "cam_azimuth_deg":   225.0,
        "antialiasing":      True,
        "fog_enabled":       False,
        "coverage_alpha":    0.30,
    }
    
    # ── Create environment ───────────────────────────────────────────────
    render_mode = "human" if args.mode == "interactive" else "rgb_array"
    
    print(f"Creating environment (render_mode={render_mode}, viz_mode=3d)...")
    
    env = SARBaseEnv(
        cfg         = cfg,
        render_mode = render_mode,
        viz_mode    = "3d",      # ✅ KEY: Enable 3D visualization
        verbose     = 1,
    )
    
    print(f"✅ Environment created")
    print(f"   Map: {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"   UAVs: {cfg.env.n_uav}")
    print(f"   Victims: {cfg.victim.n_victims_min}-{cfg.victim.n_victims_max}\n")
    
    # ── Run episodes ─────────────────────────────────────────────────────
    for ep in range(args.episodes):
        print(f"\n{'─'*70}")
        print(f"  Episode {ep+1}/{args.episodes}")
        print(f"{'─'*70}\n")
        
        run_episode(env, args, ep)
    
    # ── Cleanup ──────────────────────────────────────────────────────────
    env.close()
    print("\n✅ Demo complete!\n")


def run_episode(env, args, ep_idx):
    """Chạy 1 episode với visualization."""
    seed = args.seed + ep_idx
    
    # Reset
    obs, info = env.reset(seed=seed)
    print(f"  Reset done (seed={seed})")
    print(f"    Victims: {info['n_victims']}")
    print(f"    Obstacles: {info['n_obstacles']}\n")
    
    # Tracking
    frames = []
    total_reward = 0.0
    step_time = 1.0 / args.fps if args.mode == "interactive" else 0
    
    # Run
    for step in range(args.steps):
        t_start = time.time()
        
        # ── Random actions (replace với trained policy sau) ─────────────
        actions = {
            uid: env.action_space.sample()
            for uid in range(env.cfg.env.n_uav)
        }
        
        # ── Step environment ────────────────────────────────────────────
        obs, rewards, done, truncated, info = env.step(actions)
        
        # Accumulate reward
        if isinstance(rewards, dict):
            step_reward = np.mean(list(rewards.values()))
        else:
            step_reward = float(rewards)
        total_reward += step_reward
        
        # ── Render ──────────────────────────────────────────────────────
        frame = env.render()
        if frame is not None and args.mode != "interactive":
            frames.append(frame)
        
        # ── Progress print ──────────────────────────────────────────────
        if step % 20 == 0 or done or truncated:
            cov = info.get("coverage_rate", 0.0)
            v_found = info.get("victims_found", 0)
            v_total = info.get("victims_total", 1)
            
            print(
                f"  Step {step:3d} | "
                f"cov={cov:5.1%} | "
                f"victims={v_found}/{v_total} | "
                f"reward={step_reward:+6.1f}"
            )
        
        # ── FPS pacing (interactive mode) ───────────────────────────────
        if args.mode == "interactive":
            elapsed = time.time() - t_start
            sleep_time = step_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # ── Check done ──────────────────────────────────────────────────
        if done or truncated:
            reason = info.get("done_reason", "truncated" if truncated else "unknown")
            print(f"\n  Episode ended: {reason}")
            break
    
    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  Episode {ep_idx+1} summary:")
    print(f"    Steps:         {step+1}")
    print(f"    Total reward:  {total_reward:.1f}")
    print(f"    Coverage:      {info.get('coverage_rate', 0)*100:.1f}%")
    print(f"    Victims found: {info.get('victims_found', 0)}/{info.get('victims_total', 0)}")
    
    # ── Save output ──────────────────────────────────────────────────────
    if frames:
        save_output(frames, args, ep_idx)


def save_output(frames, args, ep_idx):
    """Save screenshots or video."""
    output_dir = Path("results/3d_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "screenshot":
        # Save first + last frame
        try:
            from PIL import Image
            
            # First frame
            Image.fromarray(frames[0]).save(
                output_dir / f"ep{ep_idx:02d}_start.png"
            )
            
            # Last frame
            Image.fromarray(frames[-1]).save(
                output_dir / f"ep{ep_idx:02d}_end.png"
            )
            
            print(f"\n  💾 Saved screenshots → {output_dir}/")
        
        except ImportError:
            print("\n  ⚠️  Pillow not installed (pip install Pillow)")
    
    elif args.mode == "video":
        # Try imageio first
        try:
            import imageio
            
            video_path = output_dir / f"ep{ep_idx:02d}.mp4"
            imageio.mimwrite(
                str(video_path),
                frames,
                fps=args.fps,
                quality=8,
            )
            print(f"\n  🎬 Saved video → {video_path} ({len(frames)} frames)")
        
        except ImportError:
            print("\n  ⚠️  imageio not installed")
            print("     Install: pip install imageio[ffmpeg]")
            
            # Fallback: save GIF
            try:
                from PIL import Image
                
                gif_path = output_dir / f"ep{ep_idx:02d}.gif"
                imgs = [Image.fromarray(f) for f in frames]
                imgs[0].save(
                    gif_path,
                    save_all=True,
                    append_images=imgs[1:],
                    duration=int(1000/args.fps),
                    loop=0,
                )
                print(f"\n  💾 Saved GIF → {gif_path}")
            
            except ImportError:
                print("     Pillow also not installed, cannot save")


if __name__ == "__main__":
    main()