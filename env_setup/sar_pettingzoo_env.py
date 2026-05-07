"""
PettingZoo wrapper for SAR UAV Swarm environment.

Provides ParallelEnv API for multi-agent RL frameworks (EPyMARL, RLlib, MARLlib).

Key differences from base_env:
  - Agent IDs: int → str ("uav_0", "uav_1", ...)
  - Done signal: bool → dict (per-agent terminations/truncations)
  - Infos: dict → dict[str, dict] (per-agent infos)

Thin wrapper - no logic, just API conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces



# PettingZoo imports
try:
    from pettingzoo import ParallelEnv
    from pettingzoo.utils import parallel_to_aec, wrappers
    PETTINGZOO_AVAILABLE = True
except ImportError:
    # Fallback if pettingzoo not installed
    PETTINGZOO_AVAILABLE = False
    ParallelEnv = object  # Dummy base class

from config import AppConfig
from env_setup.base_env import SARBaseEnv
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════

class SARPettingZooEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for SAR UAV Swarm.
    
    Wraps SARBaseEnv (composition pattern) and converts API:
      - reset() returns (obs_dict, infos_dict)
      - step() returns (obs, rews, terms, truncs, infos)
      - Agent names are strings: "uav_0", "uav_1", ...
      - Per-agent terminations/truncations
    
    Usage:
        env = SARPettingZooEnv(cfg)
        obs, infos = env.reset(seed=42)
        
        for step in range(1000):
            actions = {agent: env.action_space(agent).sample() 
                      for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            if not env.agents:  # All agents done
                break
    
    Compatible with:
      - PettingZoo utilities (parallel_to_aec, SuperSuit)
      - RLlib MultiAgentEnv
      - EPyMARL
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "sar_uav_swarm_v0",
        "is_parallelizable": True,
    }
    
    def __init__(
        self,
        cfg: AppConfig | None = None,
        backend: str = "logic",
        render_mode: str | None = None,
        n_victims: int | None = None,
        verbose: int = 0,
        viz_mode:  str = "2d"
    ):
        """
        Initialize PettingZoo environment.
        
        Args:
            cfg: AppConfig instance (or None for default)
            backend: "logic" | "pybullet" | "isaac"
            render_mode: "human" | "rgb_array" | None
            n_victims: Override number of victims (for testing)
            verbose: 0=silent, 1=info, 2=debug
        """
        if not PETTINGZOO_AVAILABLE:
            raise ImportError(
                "PettingZoo not installed. Install with: pip install pettingzoo"
            )
        
        super().__init__()
        
        # Wrap base environment
        self._base_env = SARBaseEnv(
            cfg=cfg,
            backend=backend,
            render_mode=render_mode,
            n_victims_override=n_victims,
            verbose=verbose,
            viz_mode= viz_mode
        )
        
        self.cfg = self._base_env.cfg
        
        # PettingZoo metadata
        self.possible_agents = [f"uav_{i}" for i in range(self.cfg.env.n_uav)]
        self.agents: list[str] = []  # Updated in reset()
        
        # Action/Observation spaces (per agent)
        actor_dim = self.cfg.obs.actor_dim  # 68 with n_stations=2
        
        self._observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(actor_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        
        self._action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),  # (vx, vy, vz)
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        
        self.render_mode = render_mode
        self._episode_info: dict[str, Any] = {}
    
    # ══════════════════════════════════════════════════════════════════════
    # PETTINGZOO CORE API
    # ══════════════════════════════════════════════════════════════════════
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """
        Reset environment.
        
        Returns:
            observations: {agent_name: obs_array(actor_dim,)}
            infos: {agent_name: info_dict}
        """
        # Reset base environment (returns int keys)
        obs_dict_int, info = self._base_env.reset(seed=seed)
        
        # Convert int keys → str keys
        self.agents = [f"uav_{uid}" for uid in sorted(obs_dict_int.keys())]
        
        observations = {
            f"uav_{uid}": obs
            for uid, obs in obs_dict_int.items()
        }
        
        # Per-agent infos (all agents share same global info for now)
        infos = {agent: info.copy() for agent in self.agents}
        
        # Store episode info
        self._episode_info = info
        
        return observations, infos
    
    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],  # observations
        dict[str, float],        # rewards
        dict[str, bool],         # terminations
        dict[str, bool],         # truncations
        dict[str, dict],         # infos
    ]:
        """
        Step environment with actions.
        
        Args:
            actions: {agent_name: action_array}
                     e.g., {"uav_0": [0.5, 0.3, -0.1], "uav_1": ...}
        
        Returns:
            observations: {agent_name: obs_array}
            rewards: {agent_name: reward_value}
            terminations: {agent_name: bool} - episode ended for agent
            truncations: {agent_name: bool} - episode truncated
            infos: {agent_name: info_dict}
        """
        # Convert str keys → int keys
        actions_int = {
            int(agent.split("_")[1]): action
            for agent, action in actions.items()
        }
        
        # Step base environment
        obs_dict, rewards_dict, done, truncated, info = self._base_env.step(actions_int)
        
        # Convert int keys → str keys
        observations = {
            f"uav_{uid}": obs
            for uid, obs in obs_dict.items()
        }
        
        rewards = {
            f"uav_{uid}": rew
            for uid, rew in rewards_dict.items()
        }
        
        # PettingZoo requires per-agent terminations/truncations
        # In our env, all agents terminate together (shared done)
        # But we support per-agent for API compliance
        
        current_agents = list(observations.keys())
        
        all_agents_for_signal = self.possible_agents

        terminations = {agent: done     for agent in all_agents_for_signal}
        truncations  = {agent: truncated for agent in all_agents_for_signal}
        infos        = {agent: info.copy() for agent in all_agents_for_signal}
        
        if done or truncated:
            self.agents = []
        else:
            self.agents = current_agents
        
        return observations, rewards, terminations, truncations, infos
    
    # ══════════════════════════════════════════════════════════════════════
    # PETTINGZOO REQUIRED PROPERTIES
    # ══════════════════════════════════════════════════════════════════════
    
    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for agent."""
        return self._observation_spaces[agent]
    
    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for agent."""
        return self._action_spaces[agent]
    
    # ══════════════════════════════════════════════════════════════════════
    # OPTIONAL METHODS
    # ══════════════════════════════════════════════════════════════════════
    
    def render(self) -> np.ndarray | None:
        """
        Render environment.
        
        Delegates to base_env.render().
        
        Returns:
            np.ndarray: RGB array if render_mode="rgb_array"
            None: if render_mode="human" (displays window)
        """
        return self._base_env.render()
    
    def close(self):
        """Close environment and cleanup resources."""
        self._base_env.close()
    
    @property
    def unwrapped(self) -> SARBaseEnv:
        """Return unwrapped base environment."""
        return self._base_env
    
    # ══════════════════════════════════════════════════════════════════════
    # CUSTOM PROPERTIES (for convenience)
    # ══════════════════════════════════════════════════════════════════════
    
    @property
    def num_agents(self) -> int:
        """Current number of active agents."""
        return len(self.agents)
    
    @property
    def max_num_agents(self) -> int:
        """Maximum possible agents."""
        return len(self.possible_agents)


# ═══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def make_parallel_env(
    cfg: AppConfig | None = None,
    **kwargs,
) -> SARPettingZooEnv:
    """
    Factory function for parallel env.
    
    Usage:
        env = make_parallel_env(verbose=1)
        env = make_parallel_env(cfg=my_cfg, n_victims=10)
    
    Args:
        cfg: AppConfig instance
        **kwargs: Passed to SARPettingZooEnv
    
    Returns:
        SARPettingZooEnv instance
    """
    return SARPettingZooEnv(cfg=cfg, **kwargs)


def make_aec_env(
    cfg: AppConfig | None = None,
    **kwargs,
) -> wrappers.OrderEnforcingWrapper:
    """
    Factory function for AEC (agent-environment-cycle) env.
    
    Some frameworks prefer AEC API over parallel API.
    
    Usage:
        env = make_aec_env(verbose=1)
    
    Args:
        cfg: AppConfig instance
        **kwargs: Passed to SARPettingZooEnv
    
    Returns:
        AEC-wrapped environment
    """
    if not PETTINGZOO_AVAILABLE:
        raise ImportError("PettingZoo not installed")
    
    parallel_env = SARPettingZooEnv(cfg=cfg, **kwargs)
    aec_env = parallel_to_aec(parallel_env)
    return aec_env
