from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Reward v4.0 - Anti-exploit redesign
    
    CORE PHILOSOPHY:
        Reward phải PROPORTIONAL với task progress
        Coverage 28% → reward thấp
        Coverage 80% → reward cao
        Landing là NECESSARY ACTION, không phải goal
    
    EXPLOIT FIX:
        v3.x: landing = 20 + early_bonus(50) = +70 max
              → Agent farm landing thay vì explore
        
        v4.0: landing = 5 (fixed, no bonus)
              coverage_delta tăng mạnh để dominate
              Approach/hover reward giảm về 0
    
    
    """
    # ══════════════════════════════════════════════════════
    # COVERAGE - DOMINANT SIGNAL (tăng mạnh từ 15 → 30)
    # Lý do: Coverage phải là signal CHÍNH
    # 1% coverage → +0.30 per step
    # Agent explore toàn map → +3000 per episode
    # ══════════════════════════════════════════════════════
    r_coverage_delta: float = 30.0

    # ══════════════════════════════════════════════════════
    # VICTIM - IMPORTANT SIGNAL
    # urgency 5.0 → +50.0 (giữ nguyên)
    # urgency 1.0 → +10.0
    # ══════════════════════════════════════════════════════
    r_victim_base: float = 50.0

    # ══════════════════════════════════════════════════════
    # TERMINAL
    # ══════════════════════════════════════════════════════
    r_terminal_base:    float = 300.0
    terminal_bonus_cap: float = 200.0

    # ══════════════════════════════════════════════════════
    # BATTERY PENALTIES (giảm để không dominate)
    # Chỉ cần đủ để agent "feel pain", không quá lớn
    # ══════════════════════════════════════════════════════
    r_battery_20:   float = -0.5    # Nhẹ, chỉ warning
    r_battery_10:   float = -2.0    # Moderate
    r_battery_5:    float = -8.0    # Severe
    r_battery_dead: float = -50.0   # One-time, đau

    # ══════════════════════════════════════════════════════
    # COLLISION
    # ══════════════════════════════════════════════════════
    r_collision_obstacle: float = -15.0

    # ══════════════════════════════════════════════════════
    # PROXIMITY
    # ══════════════════════════════════════════════════════
    r_proximity_1m:        float = -1.0
    r_proximity_2m:        float = -0.3
    r_proximity_3m:        float = -0.05
    proximity_penalty_cap: float = -3.0

    # ══════════════════════════════════════════════════════
    # TIME
    # ══════════════════════════════════════════════════════
    r_time_penalty: float = -0.2

    # ══════════════════════════════════════════════════════
    # LANDING REWARDS - v4.0 REDESIGN
    #
    # BEFORE (v3.x):
    #   r_landing_success = 20 (hardcoded)
    #   early_bonus = up to +50 (EXPLOIT!)
    #   approach_weight = 0.5 (too high)
    #
    # AFTER (v4.0):
    #   r_landing_success = 5 (small, no bonus)
    #   early_bonus = REMOVED
    #   approach_weight = 0.05 (tiny, just nudge)
    #   hover_penalty = -2.0 (giữ anti-hover)
    #
    # RATIONALE:
    #   Landing là survival action, không phải goal
    #   Agent phải land VÌ pin thấp, không vì reward
    #   5 reward = 1/10 victim urgency avg (đúng proportion)
    # ══════════════════════════════════════════════════════
    r_landing_success: float = 5.0    # Giảm từ 20 → 5
    r_approach_weight: float = 0.05   # Giảm từ 0.5 → 0.05
    r_hover_penalty:   float = -2.0   # Giảm từ -3 → -2

    # ══════════════════════════════════════════════════════
    # CAPS
    # step_penalty_cap: -8.0 (nới rộng một chút)
    # step clip: giữ rộng để không miss signals
    # ══════════════════════════════════════════════════════
    step_penalty_cap: float = -8.0

    step_reward_clip_min: float = -30.0
    step_reward_clip_max: float = +200.0

    # ══════════════════════════════════════════════════════
    # SHAPING
    # distance_shaping_max giảm vì coverage đã tăng
    # ══════════════════════════════════════════════════════
    enable_distance_shaping:      bool  = True
    distance_shaping_max_per_uav: float = 2.0   # Giảm từ 3.0 → 2.0

    # ══════════════════════════════════════════════════════
    # FLEET (deprecated)
    # ══════════════════════════════════════════════════════
    r_fleet_deploy: float = 0.0
    r_fleet_recall: float = 0.0

    # Episode clip (logging only)
    episode_reward_clip_min: float = -5000.0
    episode_reward_clip_max: float = +10000.0