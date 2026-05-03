# sensors/fov_sensor.py - Enhanced với noise
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from utils.geometry import dist_2d, check_los_2d

if TYPE_CHECKING:
    from entities.uav import UAV
    from entities.victim import BaseVictim
    from config import AppConfig

logger = logging.getLogger(__name__)


class FOVSensor:
    """
    Field-of-View Sensor với noise model.
    
    Noise pipeline:
        P_final = P_altitude × E_smoke × E_motion × E_victim
        
        P_altitude: cao → khó thấy (đã có)
        E_smoke:    victim trong smoke/fire → khó thấy hơn  
        E_motion:   UAV bay nhanh → motion blur → khó thấy hơn
        E_victim:   Injured stationary → dễ thấy hơn mobile
    """

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg          = cfg
        self._fov_tan     = float(cfg.sensor.fov_tan)
        self._p_base      = float(cfg.sensor.p_detect_base)
        self._p_decay     = float(cfg.sensor.p_detect_decay)
        self._n_victims   = cfg.obs.n_obs_victims
        self._n_obstacles = cfg.obs.n_obs_obstacles
        self._v_dims      = 5
        self._o_dims      = 3
        self.victim_obs_dim   = self._n_victims   * self._v_dims
        self.obstacle_obs_dim = self._n_obstacles * self._o_dims

        # ── Noise params (từ SensorConfig hoặc default) ──────────
        self._enable_noise      = getattr(cfg.sensor, "enable_noise",       True)
        self._motion_blur_coeff = getattr(cfg.sensor, "motion_blur_coeff",  0.06)
        self._base_miss_rate    = getattr(cfg.sensor, "base_miss_rate",     0.03)

        # RNG riêng cho sensor (reproducible)
        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set seed cho reproducible evaluation."""
        self._rng = np.random.default_rng(seed)

    # ── Core: FOV geometry ────────────────────────────────────────────────────

    def calculate_fov_radius(self, altitude: float) -> float:
        return float(altitude) * self._fov_tan

    # ── Core: Detection probability ───────────────────────────────────────────

    def calculate_detection_prob(
        self,
        altitude:    float,
        uav_speed:   float = 0.0,
        env_factor:  float = 1.0,
        victim_factor: float = 1.0,
    ) -> float:
        """
        Tính P(detect) với đầy đủ noise factors.

        Công thức:
            p = p_base × exp(-decay × alt)   ← altitude factor (đã có)
            p = p × env_factor               ← smoke/fire degradation
            p = p × (1 - motion_penalty)     ← motion blur
            p = p × victim_factor            ← victim type
            p = p × (1 - base_miss_rate)     ← hardware limitation

        Args:
            altitude:      UAV altitude (m)
            uav_speed:     UAV horizontal speed (m/s)
            env_factor:    Environmental visibility [0.0, 1.0]
                           1.0 = clear sky
                           0.4 = heavy smoke
                           0.2 = fire zone
            victim_factor: Victim detectability [0.75, 1.15]
                           1.15 = injured (stationary, dễ thấy)
                           0.75 = mobile (đang chạy, khó thấy)

        Returns:
            p ∈ [0.0, 1.0]
        """
        # ── Step 1: Base altitude probability ────────────────────
        # altitude=3m  → 0.95 × exp(-0.12) = 0.84
        # altitude=20m → 0.95 × exp(-0.80) = 0.43
        # altitude=40m → 0.95 × exp(-1.60) = 0.19
        p = self._p_base * np.exp(-self._p_decay * float(altitude))

        if not self._enable_noise:
            return float(np.clip(p, 0.0, 1.0))

        # ── Step 2: Environmental degradation ────────────────────
        # Victim đứng trong smoke → camera khó thấy
        # env_factor đến từ DangerZone.get_sensor_modifier()
        #   smoke:     0.40 → p giảm 60%
        #   fire:      0.55 → p giảm 45%
        #   gas:       0.80 → p giảm 20%
        #   radiation: 0.90 → p giảm 10%
        #   collapse:  0.95 → p giảm 5%
        p *= env_factor

        # ── Step 3: Motion blur ───────────────────────────────────
        # UAV bay nhanh → camera bị nhòe theo chiều ngang
        # max_speed = 5 m/s, coeff = 0.06
        # speed=0   → penalty = 0.0  → p không đổi
        # speed=2.5 → penalty = 0.03 → p giảm 3%
        # speed=5.0 → penalty = 0.06 → p giảm 6%
        max_speed = float(self.cfg.uav.max_speed_xy_mps)
        if max_speed > 0 and uav_speed > 0:
            speed_ratio    = np.clip(uav_speed / max_speed, 0.0, 1.0)
            motion_penalty = self._motion_blur_coeff * speed_ratio
            p *= (1.0 - motion_penalty)

        # ── Step 4: Victim type factor ────────────────────────────
        # Injured (nằm im) → dễ nhận dạng hơn (1.15)
        # Mobile (đang chạy) → khó nhận dạng hơn (0.75-0.90)
        p *= victim_factor

        # ── Step 5: Hardware base miss rate ──────────────────────
        # Dù điều kiện hoàn hảo, camera vẫn có 3% miss rate
        # (lens flare, compression artifacts, etc.)
        p *= (1.0 - self._base_miss_rate)

        return float(np.clip(p, 0.0, 1.0))

    # ── Noise factor extractors ───────────────────────────────────────────────

    def _get_env_factor(
        self,
        victim_pos: np.ndarray,
        obstacles:  list,
    ) -> float:
        """
        Tính environmental visibility tại vị trí victim.
        
        Logic:
            - Duyệt qua tất cả DangerZones
            - Nếu victim đứng trong zone → lấy sensor_modifier
            - Nếu nhiều zones overlap → lấy worst case (min)
        
        Returns:
            1.0 = clear (không có danger zone)
            0.40 = heavy smoke
            0.20 = fire
        """
        if not obstacles:
            return 1.0

        from entities.obstacle import DangerZone

        env_factor = 1.0
        for obs in obstacles:
            if not isinstance(obs, DangerZone):
                continue
            # Check nếu victim position nằm trong zone (2D check)
            if obs.is_inside(victim_pos):
                modifier   = obs.get_sensor_modifier()
                env_factor = min(env_factor, modifier)

        return float(env_factor)

    def _get_victim_factor(self, victim: "BaseVictim") -> float:
        """
        Victim type ảnh hưởng detection:
        
        InjuredVictim (stationary):
            - Nằm im → thermal signature ổn định → dễ detect
            - factor = 1.15
        
        MobileVictim (moving):
            - Đang di chuyển → blur, khó lock
            - speed cao → factor thấp hơn
            - factor = 0.75 ~ 0.95
        
        BaseVictim (unknown):
            - factor = 1.0
        """
        from entities.victim import InjuredVictim, MobileVictim

        if isinstance(victim, InjuredVictim):
            return 1.15

        if isinstance(victim, MobileVictim):
            speed = float(getattr(victim, "speed", 0.3))
            # speed=0.2 m/s → factor=0.93
            # speed=0.4 m/s → factor=0.83
            # speed=0.6 m/s → factor=0.75
            factor = 1.0 - np.clip(speed * 0.5, 0.05, 0.25)
            return float(factor)

        return 1.0

    # ── Core: Single victim detection ─────────────────────────────────────────

    def check_detected(
        self,
        uav:       "UAV",
        victim:    "BaseVictim",
        obstacles: Optional[list] = None,
    ) -> bool:
        """
        Kiểm tra UAV có detect được victim không.
        
        Pipeline:
            1. FOV geometric check  → fast reject
            2. LOS check            → fast reject  
            3. Noise probability    → stochastic
        """
        fov_r = self.calculate_fov_radius(uav.pos[2])

        # ── 1. FOV check ─────────────────────────────────────────
        if dist_2d(uav.pos, victim.pos) > fov_r:
            return False

        # ── 2. LOS check ─────────────────────────────────────────
        if obstacles and not check_los_2d(uav.pos, victim.pos, obstacles):
            return False

        # ── 3. Compute P(detect) với noise ───────────────────────
        altitude = float(uav.pos[2])

        # Environmental factor (victim trong smoke/fire?)
        env_factor = self._get_env_factor(victim.pos, obstacles or [])

        # UAV speed (motion blur)
        vel       = getattr(uav, "vel", np.zeros(3))
        uav_speed = float(np.linalg.norm(vel[:2]))

        # Victim type factor
        victim_factor = self._get_victim_factor(victim)

        # Final probability
        p = self.calculate_detection_prob(
            altitude      = altitude,
            uav_speed     = uav_speed,
            env_factor    = env_factor,
            victim_factor = victim_factor,
        )

        # ── 4. Stochastic sample ──────────────────────────────────
        return bool(self._rng.random() < p)

    # ── Batch scan: Victims ───────────────────────────────────────────────────

    def scan_victims(
        self,
        uav:       "UAV",
        victims:   List["BaseVictim"],
        obstacles: Optional[list] = None,
    ) -> np.ndarray:
        """
        Quét victims trong FOV → observation vector.
        
        Noise ảnh hưởng:
            - check_detected() dùng noise → newly found
            - obs vector vẫn show victims trong FOV
              (UAV "thấy" nhưng chưa chắc "nhận ra")
        """
        fov_r = self.calculate_fov_radius(uav.pos[2])
        if fov_r <= 0:
            return np.zeros(self.victim_obs_dim, dtype=np.float32)

        candidates: List[Tuple[float, "BaseVictim"]] = []
        for v in victims:
            d = dist_2d(uav.pos, v.pos)
            if d > fov_r:
                continue
            if obstacles and not check_los_2d(uav.pos, v.pos, obstacles):
                continue
            candidates.append((d, v))

        candidates.sort(key=lambda x: x[0])

        result = np.zeros(self.victim_obs_dim, dtype=np.float32)
        for i, (d, v) in enumerate(candidates[:self._n_victims]):
            base             = i * self._v_dims
            result[base + 0] = (v.pos[0] - uav.pos[0]) / fov_r
            result[base + 1] = (v.pos[1] - uav.pos[1]) / fov_r
            result[base + 2] = d / fov_r
            result[base + 3] = float(v.urgency) / 5.0
            result[base + 4] = 1.0 if v.is_found else 0.0

        return result

    def scan_obstacles(
        self,
        uav:       "UAV",
        obstacles: list,
    ) -> np.ndarray:
        """Quét obstacles - KHÔNG có noise (obstacles là static, dễ thấy)."""
        fov_r = self.calculate_fov_radius(uav.pos[2])
        if fov_r <= 0:
            return np.zeros(self.obstacle_obs_dim, dtype=np.float32)

        from entities.obstacle import Debris, DangerZone

        candidates: List[Tuple[float, object, float]] = []
        for obs in obstacles:
            d          = dist_2d(uav.pos, obs.pos)
            obs_radius = obs._get_fallback_radius()
            if d > fov_r + obs_radius:
                continue
            type_id = 1.0 if isinstance(obs, DangerZone) else 0.0
            candidates.append((d, obs, type_id))

        candidates.sort(key=lambda x: x[0])

        result = np.zeros(self.obstacle_obs_dim, dtype=np.float32)
        for i, (d, obs, tid) in enumerate(candidates[:self._n_obstacles]):
            base             = i * self._o_dims
            result[base + 0] = np.clip(
                (obs.pos[0] - uav.pos[0]) / fov_r, -1.5, 1.5
            )
            result[base + 1] = np.clip(
                (obs.pos[1] - uav.pos[1]) / fov_r, -1.5, 1.5
            )
            result[base + 2] = tid

        return result