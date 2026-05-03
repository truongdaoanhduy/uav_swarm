# File: test_visualize_stages.py

"""
Visualize 2D Demo - 3 Curriculum Stages

Chạy 1 episode mỗi stage (EASY/MEDIUM/HARD) với random policy.
Output:
  - GIF animation (toàn bộ episode)
  - PNG frame cuối (final state)

Usage:
    python test_visualize_stages.py
"""

import os
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend

import numpy as np
import matplotlib
matplotlib.use('Agg')
import time
from pathlib import Path

from config import AppConfig
from config.curriculum_config import STAGE_EASY, STAGE_MEDIUM, STAGE_HARD
from env_setup import SARBaseEnv


# ══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def visualize_stage(
    stage_config,
    stage_name: str,
    max_steps: int = 200,
    seed: int = 42,
    output_dir: str = "results/visualize_demo",
) -> None:
    """
    Chạy 1 episode và save GIF + PNG.
    
    Args:
        stage_config: StageConfig object (STAGE_EASY/MEDIUM/HARD)
        stage_name: Tên stage để save file ("easy"/"medium"/"hard")
        max_steps: Số steps tối đa (giảm để GIF nhẹ hơn)
        seed: Random seed
        output_dir: Thư mục output
    """
    print(f"\n{'='*70}")
    print(f"  VISUALIZING: {stage_name.upper()} STAGE")
    print(f"{'='*70}")
    
    # ─── [1] Setup ───────────────────────────────────────────────────────
    cfg = AppConfig()
    cfg.apply_stage(stage_config)
    cfg.env.max_steps = 1000
    print(f"  Map: {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"  UAVs: {cfg.env.n_uav}")
    print(f"  Victims: {cfg.victim.n_victims_min}-{cfg.victim.n_victims_max}")
    print(f"  Max steps: {max_steps}")
    print(f"  Seed: {seed}")
    
    # ─── [2] Create Environment ──────────────────────────────────────────
    env = SARBaseEnv(
        cfg=cfg,
        render_mode="rgb_array",  # ← Return numpy arrays
        viz_mode="2d",            # ← Use Visualizer2D
        verbose=0,                # ← Silent
    )
    
    # ─── [3] Reset ───────────────────────────────────────────────────────
    obs, info = env.reset(seed=seed)
    
    print(f"\n  Episode started:")
    print(f"    UAVs: {info['n_uav']}")
    print(f"    Victims: {info['n_victims']}")
    print(f"    Obstacles: {info['n_obstacles']}")
    
    # ─── [4] Run Episode & Collect Frames ────────────────────────────────
    frames = []
    done = False
    step = 0
    
    print(f"\n  Running episode...")
    start_time = time.time()
    
    # Capture FIRST frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    # Run episode
    while not done and step < max_steps:
        # Random actions
        actions = {i: env.action_space.sample() 
                   for i in range(cfg.env.n_uav)}
        
        # Step
        obs, rewards_dict, dones, truncs, infos = env.step(actions)
        
        # ═══ OPTIMIZE: Capture mỗi 5 steps (giảm GIF size) ═══
        if step % 5 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Check done
        if isinstance(dones, dict):
            done = all(dones.values()) or all(truncs.values())
        else:
            done = bool(dones) or bool(truncs)
        
        step += 1
        
        # Progress indicator mỗi 50 steps
        if step % 450 == 0:
            print(f"    Step {step}/{max_steps} | "
                  f"Coverage: {infos['coverage_rate']:.1%} | "
                  f"Victims: {infos['victims_found']}/{infos['victims_total']}")
    
    # Capture LAST frame (quan trọng!)
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    elapsed = time.time() - start_time
    
    # Final metrics
    final_coverage = infos['coverage_rate']
    final_victims = infos['victims_found']
    total_victims = infos['victims_total']
    done_reason = infos.get('done_reason', 'truncated')
    
    print(f"\n  Episode complete:")
    print(f"    Steps: {step}/{max_steps}")
    print(f"    Coverage: {final_coverage:.1%}")
    print(f"    Victims: {final_victims}/{total_victims}")
    print(f"    Reason: {done_reason}")
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Frames: {len(frames)}")
    
    env.close()
    
    # ─── [5] Save Outputs ────────────────────────────────────────────────
    if len(frames) == 0:
        print("\n  ⚠️  WARNING: No frames captured!")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save GIF
    gif_path = output_path / f"{stage_name}_episode.gif"
    _save_gif(frames, gif_path, fps=10)
    
    # Save PNG (last frame only)
    png_path = output_path / f"{stage_name}_final.png"
    _save_png(frames[-1], png_path)
    
    print(f"\n  ✅ Outputs saved:")
    print(f"    GIF:  {gif_path}")
    print(f"    PNG:  {png_path}")


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _save_gif(frames, filepath, fps=10):
    """Save frames as GIF using Pillow."""
    try:
        from PIL import Image
        
        print(f"\n  Creating GIF ({len(frames)} frames @ {fps} fps)...")
        
        # Convert to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save GIF
        duration = int(1000 / fps)  # milliseconds per frame
        pil_frames[0].save(
            filepath,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,  # Loop forever
            optimize=False,  # ← Faster (không optimize file size)
        )
        
        # File size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✅ GIF saved: {filepath.name} ({size_mb:.1f} MB)")
        
    except ImportError:
        print("  ❌ Pillow not installed! Install with: pip install Pillow")
    except Exception as e:
        print(f"  ❌ GIF save failed: {e}")


def _save_png(frame, filepath):
    """Save single frame as PNG."""
    try:
        from PIL import Image
        
        img = Image.fromarray(frame)
        img.save(filepath, optimize=True)
        
        size_kb = filepath.stat().st_size / 1024
        print(f"  ✅ PNG saved: {filepath.name} ({size_kb:.0f} KB)")
        
    except Exception as e:
        print(f"  ❌ PNG save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  2D VISUALIZATION DEMO - 3 CURRICULUM STAGES")
    print("="*70)
    print("\n  Output:")
    print("    - GIF animation (1 per stage)")
    print("    - PNG final frame (1 per stage)")
    print("\n  Settings:")
    print("    - Random policy")
    print("    - 200 steps max per episode")
    print("    - Capture every 5 steps (smoother GIF)")
    print("    - FPS: 10 (balanced speed)")
    
    # Create output directory
    output_dir = "results/visualize_demo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ─── Run 3 stages ────────────────────────────────────────────────────
    stages = [
        (STAGE_EASY, "easy"),
        (STAGE_MEDIUM, "medium"),
        (STAGE_HARD, "hard"),
    ]
    
    total_start = time.time()
    
    for stage_config, stage_name in stages:
        visualize_stage(
            stage_config=stage_config,
            stage_name=stage_name,
            max_steps=1000,      # ← Giảm từ 300 để GIF nhẹ hơn
            seed=42,            # ← Reproducible
            output_dir=output_dir,
        )
    
    total_elapsed = time.time() - total_start
    
    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(f"\n  Outputs in: {output_dir}/")
    print("\n  Files:")
    
    for _, stage_name in stages:
        gif_path = Path(output_dir) / f"{stage_name}_episode.gif"
        png_path = Path(output_dir) / f"{stage_name}_final.png"
        
        if gif_path.exists():
            size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"    ✅ {gif_path.name:<25} ({size_mb:5.1f} MB)")
        
        if png_path.exists():
            size_kb = png_path.stat().st_size / 1024
            print(f"    ✅ {png_path.name:<25} ({size_kb:6.0f} KB)")
    
    print("\n" + "="*70)
    print("\n  Next steps:")
    print("    1. Open GIF files to see animation:")
    print(f"       - {output_dir}/easy_episode.gif")
    print(f"       - {output_dir}/medium_episode.gif")
    print(f"       - {output_dir}/hard_episode.gif")
    print("\n    2. Check PNG files for final states:")
    print(f"       - {output_dir}/easy_final.png")
    print(f"       - {output_dir}/medium_final.png")
    print(f"       - {output_dir}/hard_final.png")
    print("\n    3. Verify visualization quality:")
    print("       - Colors clear & distinguishable?")
    print("       - UAVs visible in all states?")
    print("       - Victims easy to identify?")
    print("       - Coverage overlay readable?")
    print("\n  ✅ Visualization demo complete!")
    print("="*70 + "\n")