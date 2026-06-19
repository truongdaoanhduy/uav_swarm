"""
env/base_env.py - FIXED VERSION
"""
from __future__ import annotations

import time
import logging
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

from config import AppConfig
from core.map_generator import MapGenerator
from entities.uav import UAV, UAVState
from observation.obs_builder import ObservationBuilder
from rewards.baseline_reward import BaselineReward
from utils.logger import EpisodeLogger
from visualization.renderer_factory import create_renderer

logger = logging.getLogger(__name__)

_DONE_COVERAGE_THRESHOLD = 0.95

_INFO_STEP          = "step"
_INFO_COVERAGE      = "coverage_rate"
_INFO_VICTIMS_FOUND = "victims_found"
_INFO_VICTIMS_TOTAL = "victims_total"
_INFO_N_ACTIVE      = "n_active"
_INFO_N_CHARGING    = "n_charging"
_INFO_N_DISABLED    = "n_disabled"
_INFO_SUCCESS       = "success"
_INFO_DONE_REASON   = "done_reason"
_INFO_EPISODE_TIME  = "episode_time_s"
_INFO_REWARDS       = "rewards_breakdown"


class SARBaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg:                AppConfig | None = None,
        backend:            str = "logic",
        render_mode:        str | None = None,
        n_victims_override: int | None = None,
        verbose:            int = 0,
        viz_mode:           str = "2d",
    ):
        super().__init__()
        self.cfg           = cfg or AppConfig()
        self.render_mode   = render_mode
        self._n_victims_ov = n_victims_override
        self.verbose       = verbose
        self._viz_mode     = self.cfg.viz_mode  # ✅ FIX: cfg có thể None

        actor_dim = self.cfg.obs.actor_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(actor_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32,
        )

        self._map_gen        = MapGenerator(self.cfg)
        self._reward_fn      = BaselineReward(self.cfg)
        self.baseline_reward = None

        if backend == "logic":
            from env_setup.backends.logic_backend import LogicBackend
            self.backend = LogicBackend(self.cfg)
        elif backend == "pybullet":
            raise NotImplementedError("PyBullet backend not yet implemented")
        elif backend == "isaac":
            raise NotImplementedError("IsaacLab backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        self._obs_builder:          ObservationBuilder | None = None
        self._step_count:           int   = 0
        self._prev_coverage:        float = 0.0
        self._episode_seed:         int   = 0
        self._episode_id:           int   = 0
        self._ep_logger:            EpisodeLogger | None = None
        self._ep_start_time:        float = 0.0
        self._episode_reward_sum:   float = 0.0
        self._step_rewards_history: list  = []
        self._renderer              = None
        self._prev_uav_states:      dict[int, UAVState] = {}

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[int, np.ndarray], dict]:
        # ✅ FIX Bug 2: Indentation đồng đều 4 spaces
        super().reset(seed=seed)

        if seed is None:
            if self.cfg.env.deterministic_eval:
                seed = self.cfg.env.eval_seed
            else:
                seed = (self.cfg.env.global_seed + self._episode_id) % (2**31)

        self._episode_seed         = seed
        self._episode_id          += 1
        self._ep_start_time        = time.time()
        self._episode_reward_sum   = 0.0
        self._step_rewards_history = []

        self._active_reward_fn.reset()

        map_data = self._map_gen.generate(
            n_victims_override = self._n_victims_ov,
            seed               = seed,
        )
        self.backend.reset(map_data)

        self._step_count    = 0
        self._prev_coverage = 0.0

        state = self.backend.get_state()
        self._obs_builder = ObservationBuilder(
            state["coverage_map"],
            self.cfg,
        )

        self._ep_logger = EpisodeLogger(
            episode_id = self._episode_id,
            seed       = seed,
        )
        self._ep_logger.set_total_victims(len(state["victims"]))

        self._prev_uav_states = {
            uav.id: uav.state
            for uav in state["uavs"]
        }

        obs_dict, critic_obs = self._build_obs_dict(
            state["uavs"],
            state["stations"],
            state["victims"],
            state["obstacles"],
        )

        if self._renderer is not None and hasattr(self._renderer, "reset_scene"):
            self._renderer.reset_scene()

        info = {
            "seed":              seed,
            "n_uav":             len(state["uavs"]),
            "n_stations":        len(state["stations"]),
            "n_victims":         len(state["victims"]),
            "n_obstacles":       len(state["obstacles"]),
            "map_size":          self.cfg.env.map_size,
            "coverage":          0.0,
            _INFO_COVERAGE:      0.0,
            _INFO_VICTIMS_FOUND: 0,
            _INFO_VICTIMS_TOTAL: len(state["victims"]),
            "global_obs":        critic_obs,
        }

        if self.verbose >= 2:
            print(
                f"[ENV] Episode {self._episode_id} | seed={seed} "
                f"| {len(state['uavs'])} UAVs "
                f"| {len(state['victims'])} victims"
            )

        return obs_dict, info

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: dict[int, np.ndarray],
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        bool,
        bool,
        dict,
    ]:
        self.backend.apply_actions(actions)
        self.backend.step_physics()
        self.backend.step_world()
        self._step_count += 1

        state         = self.backend.get_state()
        uavs          = state["uavs"]
        victims       = state["victims"]
        obstacles     = state["obstacles"]
        stations      = state["stations"]
        coverage_map  = state["coverage_map"]
        fleet_manager = state["fleet_manager"]

        if self._ep_logger is not None:
            for uav in uavs:
                curr_state = uav.state
                prev_state = self._prev_uav_states.get(uav.id, curr_state)

                if (prev_state != UAVState.CHARGING
                        and curr_state == UAVState.CHARGING):
                    battery_before = getattr(
                        uav, '_battery_before_charge', uav.battery_pct
                    )
                    self._ep_logger.log_landing(
                        uav_id         = uav.id,
                        step           = self._step_count,
                        battery_before = battery_before,
                        battery_after  = uav.battery_pct,
                    )

                if curr_state == UAVState.CHARGING:
                    self._ep_logger.log_charging_step(uav.id)

                if curr_state == UAVState.RETURNING:
                    uav._battery_before_charge = uav.battery_pct

            for uav in uavs:
                self._prev_uav_states[uav.id] = uav.state

        cur_coverage = coverage_map.get_coverage_rate()

        newly_found = [
            v for v in victims
            if v.is_found and v.found_at_step == self._step_count
        ]

        if self._ep_logger is not None:
            for _ in newly_found:
                self._ep_logger.log_event("victim_found")

        done_reason = self._check_done(cur_coverage, victims, uavs)
        done        = done_reason is not None
        truncated   = self._step_count >= self.cfg.env.max_steps
        is_terminal = done or truncated

        reward_fn = self._active_reward_fn

        rewards_dict: dict[int, float] = {}
        for uav in uavs:
            if uav.state == UAVState.DISABLED:
                rewards_dict[uav.id] = 0.0
                continue

            newly_found_by_uav = [
                v for v in newly_found if v.found_by_uav == uav.id
            ]

            breakdown = reward_fn.compute_per_uav(
                uav                = uav,
                newly_found_by_uav = newly_found_by_uav,
                uavs               = uavs,
                victims            = victims,
                obstacles          = obstacles,
                coverage_map       = coverage_map,
                fleet_manager      = fleet_manager,
                prev_coverage      = self._prev_coverage,
                current_step       = self._step_count,
                done               = is_terminal,
                stations           = stations,
            )
            rewards_dict[uav.id] = breakdown["total"]

        global_reward = reward_fn.compute(
            uavs          = uavs,
            victims       = victims,
            obstacles     = obstacles,
            coverage_map  = coverage_map,
            fleet_manager = fleet_manager,
            newly_found   = newly_found,
            prev_coverage = self._prev_coverage,
            current_step  = self._step_count,
            done          = is_terminal,
            stations      = stations,
        )

        if self._step_count == 1:
            reward_type = (
                "LLM" if self.baseline_reward is not None
                else "Baseline v4.0"
            )
            print(f"Using reward: {reward_type}")

        self._episode_reward_sum += global_reward["total"]
        self._step_rewards_history.append({
            "step":  self._step_count,
            "total": global_reward["total"],
        })

        # ✅ FIX Bug 4: Dùng _active_reward_fn thay vì _reward_fn
        if global_reward["total"] < -100:
            logger.warning(
                "[EP %d | STEP %d] Extreme step reward: %.1f | %s",
                self._episode_id,
                self._step_count,
                global_reward["total"],
                self._active_reward_fn.summarize(global_reward),
            )

        self._log_step(
            rewards_dict, cur_coverage, newly_found, uavs, obstacles,
            global_breakdown = global_reward,
        )
        self._prev_coverage = cur_coverage

        obs_dict, critic_obs = self._build_obs_dict(uavs, stations, victims, obstacles)

        n_found     = sum(1 for v in victims if v.is_found)
        n_total     = len(victims)
        n_active    = sum(1 for u in uavs if u.state == UAVState.ACTIVE)
        n_returning = sum(1 for u in uavs if u.state == UAVState.RETURNING)
        n_charge    = sum(1 for u in uavs if u.state == UAVState.CHARGING)
        n_deploying = sum(1 for u in uavs if u.state == UAVState.DEPLOYING)
        n_dead      = sum(1 for u in uavs if u.state == UAVState.DISABLED)

        n_total_uavs = len(uavs)
        n_accounted  = n_active + n_returning + n_charge + n_deploying + n_dead
        success      = done_reason in ("coverage", "victims")
        fleet_stats  = fleet_manager.get_battery_stats()

        info = {
            "coverage":          cur_coverage,
            "victims_found":     n_found,
            "victims_total":     n_total,
            _INFO_STEP:          self._step_count,
            _INFO_COVERAGE:      cur_coverage,
            _INFO_VICTIMS_FOUND: n_found,
            _INFO_VICTIMS_TOTAL: n_total,
            _INFO_N_ACTIVE:      n_active,
            _INFO_N_CHARGING:    n_charge,
            _INFO_N_DISABLED:    n_dead,
            "n_returning":       n_returning,
            "n_deploying":       n_deploying,
            "n_total_uavs":      n_total_uavs,
            "n_accounted":       n_accounted,
            _INFO_SUCCESS:       success,
            _INFO_DONE_REASON:   done_reason or ("truncated" if truncated else None),
            _INFO_REWARDS:       global_reward,
            "newly_found_ids":   [v.id for v in newly_found],
            "battery_stats":     fleet_stats,
            "global_obs":        critic_obs,
        }

        if is_terminal:
            if self._ep_logger is not None:
                ep_metrics = self._ep_logger.finalize()
                info["episode"]          = ep_metrics
                info[_INFO_EPISODE_TIME] = time.time() - self._ep_start_time

                if self.verbose >= 1:
                    self._print_episode_summary(ep_metrics, done_reason, truncated)

            if self._episode_reward_sum < -500:
                self._log_extreme_episode(
                    episode_reward = self._episode_reward_sum,
                    cur_coverage   = cur_coverage,
                    n_found        = n_found,
                    n_total        = n_total,
                    uavs           = uavs,
                    obstacles      = obstacles,
                    done_reason    = done_reason,
                    truncated      = truncated,
                )

        return obs_dict, rewards_dict, done, truncated, info

    # ── Private ──────────────────────────────────────────────────────────────

    def _check_done(self, coverage, victims, uavs) -> str | None:
        threshold = getattr(
            self.cfg.env,
            "done_coverage_threshold",
            _DONE_COVERAGE_THRESHOLD,
        )

        if coverage >= threshold:
            return "coverage"

        # ✅ FIX Bug 1: Logic đúng
        if victims and all(v.is_found for v in victims):
            return "victims"

        all_disabled = all(u.state == UAVState.DISABLED for u in uavs)
        if all_disabled:
            dead_batteries = sum(1 for u in uavs if u.battery_death)
            if dead_batteries == len(uavs):
                return "disabled:battery_death"
            else:
                return "disabled:other"

        return None

    def _log_step(
        self,
        rewards:          dict,
        coverage:         float,
        newly_found:      list,
        uavs:             list,
        obstacles:        list,
        global_breakdown: Optional[dict] = None,  # ✅ FIX Bug 3: Optional import
    ) -> None:
        if self._ep_logger is None:
            return

        self._ep_logger.log_step(
            rewards   = rewards,
            coverage  = coverage,
            breakdown = global_breakdown,
        )

        for uav in uavs:
            for obs in obstacles:
                if hasattr(obs, "causes_collision") and obs.causes_collision(uav.pos):
                    obs_info = {
                        "id":     obs.id if hasattr(obs, "id") else None,
                        "type":   type(obs).__name__,
                        "pos":    obs.pos.tolist() if hasattr(obs, "pos") else None,
                        "height": getattr(obs, "height_3d", None),
                    }
                    self._ep_logger.log_collision(uav.id, self._step_count, obs_info)
                    self._ep_logger.log_event("collision_obstacle")
                    break

    # ── Các methods còn lại giữ nguyên ───────────────────────────────────────

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = self._init_renderer()
        state = self.backend.get_state()
        return self._renderer.render(
            uavs      = state["uavs"],
            victims   = state["victims"],
            obstacles = state["obstacles"],
            stations  = state["stations"],
            cov_map   = state["coverage_map"],
            step      = self._step_count,
        )

    def _init_renderer(self):
        return create_renderer(
            cfg         = self.cfg,
            render_mode = self.render_mode,
            viz_mode    = self._viz_mode,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @property
    def n_agents(self) -> int:
        return sum(
            1 for u in self.backend.get_state()["uavs"]
            if u.state != UAVState.DISABLED
        )

    @property
    def active_uav_ids(self) -> list[int]:
        return [
            u.id for u in self.backend.get_state()["uavs"]
            if u.state == UAVState.ACTIVE
        ]

    @property
    def alive_uav_ids(self) -> list[int]:
        return [
            u.id for u in self.backend.get_state()["uavs"]
            if u.state != UAVState.DISABLED
        ]

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def coverage_rate(self) -> float:
        return self.backend.get_state()["coverage_map"].get_coverage_rate()

    @property
    def _active_reward_fn(self):
        return self.baseline_reward if self.baseline_reward is not None else self._reward_fn

    def _build_obs_dict(
        self,
        uavs:      list,
        stations:  list,
        victims:   list,
        obstacles: list,
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        obs_dict: dict[int, np.ndarray] = {}
        result = self._obs_builder.build_all(
            all_uavs     = uavs,
            stations     = stations,
            victims      = victims,
            obstacles    = obstacles,
            current_step = self._step_count,
        )
        for uid, obs in result.actor_obs.items():
            uav = self._get_uav_from_list(uid, uavs)
            if uav is None or uav.state == UAVState.DISABLED:
                continue
            obs_dict[uid] = obs.astype(np.float32)
        return obs_dict, result.critic_obs.copy()

    def _print_episode_summary(
        self,
        metrics:     dict,
        done_reason: str | None,
        truncated:   bool,
    ) -> None:
        success = metrics.get("success", False)
        status  = "SUCCESS" if success else "FAIL"
        reason  = done_reason or ("truncated" if truncated else "unknown")
        cov     = (
            metrics.get("coverage_rate")
            or metrics.get("coverage")
            or metrics.get("final_coverage")
            or 0.0
        )
        v_found           = metrics.get("victims_found") or metrics.get("n_found") or 0
        v_total           = metrics.get("total_victims") or metrics.get("victims_total") or 0
        total_landings    = metrics.get("total_landings", 0)
        total_charge_time = metrics.get("total_charge_time", 0)

        print(
            f"[ENV] Ep {self._episode_id} {status} | "
            f"reason={reason} | "
            f"steps={self._step_count} | "
            f"cov={cov:.1f}% | "
            f"victims={v_found}/{v_total} | "
            f"landings={total_landings}× ({total_charge_time} charge steps) | "
            f"ep_reward={self._episode_reward_sum:.1f}"
        )

    def _log_extreme_episode(
        self,
        episode_reward: float,
        cur_coverage:   float,
        n_found:        int,
        n_total:        int,
        uavs:           list,
        obstacles:      list,
        done_reason:    str | None,
        truncated:      bool,
    ) -> None:
        from entities.obstacle import DangerZone

        n_in_danger = sum(
            1 for u in uavs for obs in obstacles
            if isinstance(obs, DangerZone) and obs.is_inside(u.pos)
        )
        _rfn           = self._active_reward_fn
        n_collisions   = len(getattr(_rfn, "_collision_penalized",     set()))
        n_dead_battery = len(getattr(_rfn, "_battery_death_penalized", set()))

        emergency_pct = getattr(
            self.cfg.uav, "battery_emergency_pct",
            getattr(self.cfg.uav, "battery_penalty_emergency", 5.0),
        )
        n_battery_critical = sum(1 for u in uavs if u.battery < emergency_pct)
        n_disabled         = sum(1 for u in uavs if u.state == UAVState.DISABLED)

        worst_str = "N/A"
        if self._step_rewards_history:
            worst     = sorted(self._step_rewards_history, key=lambda x: x["total"])[:3]
            worst_str = ", ".join(
                f"step {s['step']}={s['total']:.1f}" for s in worst
            )

        logger.warning(
            "\n%s\n⚠️  EXTREME EPISODE REWARD\n"
            "  Ep=%d | reward=%.1f | steps=%d | reason=%s\n"
            "  cov=%.1f%% | victims=%d/%d\n"
            "  in_danger=%d | collisions=%d | battery_dead=%d"
            " | battery_critical=%d | disabled=%d\n"
            "  Worst steps: %s\n%s",
            "=" * 60,
            self._episode_id, episode_reward, self._step_count,
            done_reason or ("truncated" if truncated else "N/A"),
            cur_coverage * 100, n_found, n_total,
            n_in_danger, n_collisions, n_dead_battery,
            n_battery_critical, n_disabled,
            worst_str,
            "=" * 60,
        )

        if self.verbose >= 3:
            import json
            import os
            os.makedirs("results/extreme_episodes", exist_ok=True)
            fname = (
                f"results/extreme_episodes/"
                f"ep_{self._episode_id}_{episode_reward:.0f}.json"
            )
            try:
                with open(fname, "w") as f:
                    json.dump(
                        {
                            "episode_id":   self._episode_id,
                            "reward":       episode_reward,
                            "steps":        self._step_count,
                            "step_rewards": self._step_rewards_history,
                        },
                        f, indent=2,
                    )
            except Exception as e:
                logger.warning("Failed to save extreme episode: %s", e)

    def _get_uav_from_list(self, uid: int, uavs: list) -> UAV | None:
        for u in uavs:
            if u.id == uid:
                return u
        return None

    @classmethod
    def make(
        cls,
        cfg:         AppConfig | None = None,
        render_mode: str | None = None,
        n_victims:   int | None = None,
        verbose:     int = 0,
        viz_mode:    str = "2d",
    ) -> "SARBaseEnv":
        return cls(
            cfg                = cfg,
            render_mode        = render_mode,
            n_victims_override = n_victims,
            verbose            = verbose,
            viz_mode           = viz_mode,
        )