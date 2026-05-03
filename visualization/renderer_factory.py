"""
visualization/renderer_factory.py
Factory function tạo renderer phù hợp (2D hoặc 3D).
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def create_renderer(cfg, render_mode: str, viz_mode: str = "2d"):
    """
    Factory: tạo đúng renderer theo viz_mode.

    Args:
        cfg:         AppConfig
        render_mode: "human" | "rgb_array"
        viz_mode:    "2d" | "3d"

    Returns:
        Visualizer2D hoặc Visualizer3D instance
    """
    mode = viz_mode.lower().strip()

    if mode == "3d":
        try:
            from visualization.visualizer3d import Visualizer3D
            logger.info("Using Visualizer3D (Matplotlib 3D)")
            return Visualizer3D(cfg=cfg, render_mode=render_mode)
        except ImportError as e:
            logger.warning(
                "Visualizer3D unavailable (%s), falling back to 2D", e
            )
            mode = "2d"

    # Default: 2D
    from visualization.visualizer2d import Visualizer2D
    logger.info("Using Visualizer2D (Matplotlib 2D)")
    return Visualizer2D(cfg=cfg, render_mode=render_mode)