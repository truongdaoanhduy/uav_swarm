from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Reward function configuration for SAR task.

    REBALANCED v3 - Research-grade:
        Based on random policy baseline + multi-agent RL stability
        
        Design principles:
            1. Sparse signals (coverage, victim) > Dense penalties
            2. No reward saturation (clip bounds wide enough)
            3. Step penalty cap to prevent single-step collapse
            4. Multi-agent aware (proximity normalized by pair count)

    """
    # ══════════════════════════════════════════════════════════
    # 1. COVERAGE REWARD (dense, incremental)
    # ══════════════════════════════════════════════════════════
    r_coverage_delta: float = 6.0 
    # Reward per 1% coverage increase
    # Max per episode: +600 (100% coverage) - achievable but not trivial
    # Expected with random (55%): ~330

    # ══════════════════════════════════════════════════════════
    # 2. VICTIM DISCOVERY REWARD (sparse, high value)
    # ══════════════════════════════════════════════════════════
    r_victim_base: float = 30.0  # UNCHANGED
    # Base × (urgency / 5.0)
    # Range: +10 (urgency=1) to +50 (urgency=5)
    # Expected with random (53%, avg urgency=3): ~318

    # ══════════════════════════════════════════════════════════
    # 3. BATTERY PENALTIES (progressive)
    # ══════════════════════════════════════════════════════════
    r_battery_20: float = -1.0       # ✅ CHANGED: -0.5 → 0.0 (remove early penalty)
    r_battery_10: float = -5.0      # ✅ CHANGED: -1.5 → -1.0 (reduce)
    r_battery_5: float = -20.0       # UNCHANGED (critical zone)
    r_battery_dead: float = -100.0  # UNCHANGED (one-time)

    # Rationale:
    # - Remove penalty at 20% → encourage exploration
    # - Keep strong penalty at <10% → force return behavior
    # - Dead penalty prevents catastrophic failure

    # ══════════════════════════════════════════════════════════
    # 4. COLLISION PENALTY (one-time per obstacle)
    # ══════════════════════════════════════════════════════════
    r_collision_obstacle: float = -30.0  # UNCHANGED

    # ══════════════════════════════════════════════════════════
    # 5. DANGER ZONE PENALTY (per step inside zone)
    # ══════════════════════════════════════════════════════════
    # Applied via DangerZoneConfig.penalties (already rebalanced)

    # ══════════════════════════════════════════════════════════
    # 6. PROXIMITY PENALTY (multi-threshold, per UAV pair)
    # ══════════════════════════════════════════════════════════
    r_proximity_1m: float = -10.0   # UNCHANGED
    r_proximity_2m: float = -3.0    # UNCHANGED
    r_proximity_3m: float = -0.5    # UNCHANGED
    
    # ✅ NEW: Proximity cap (per step, total across all pairs)
    proximity_penalty_cap: float = -15.0
    # Prevents proximity spam from dominating signal
    # With 6 pairs: worst case -60 → capped at -15

    # ══════════════════════════════════════════════════════════
    # 7. FLEET MANAGEMENT INCENTIVES (deprecated - set to 0)
    # ══════════════════════════════════════════════════════════
    r_fleet_deploy: float = 0.0  
    r_fleet_recall: float = 0.0   
    
    # Rationale: Fleet behavior should emerge from RL, not hand-crafted

    # ══════════════════════════════════════════════════════════
    # 8. TIME PENALTY (gentle pressure)
    # ══════════════════════════════════════════════════════════
    r_time_penalty: float = -0.1   # ✅ CHANGED: -0.1 → -0.05
    # With 4 UAVs × 300 steps: -60 total (manageable)

    # ══════════════════════════════════════════════════════════
    # 9. TERMINAL BONUS (mission success)
    # ══════════════════════════════════════════════════════════
    r_terminal_base: float = 200.0  # UNCHANGED
    terminal_bonus_cap: float = 100.0  # ✅ CHANGED: 50 → 100 (more meaningful)

    # ══════════════════════════════════════════════════════════
    # 10. REWARD CLIPPING & CAPPING
    # ══════════════════════════════════════════════════════════
    
    # ✅ CRITICAL: Step penalty cap (applied BEFORE clipping)
    # Step penalty cap (applied BEFORE clipping)
    step_penalty_cap: float = -30.0
    
    # Step-level clip (WIDENED to prevent saturation)
    step_reward_clip_min: float = -100.0
    step_reward_clip_max: float = +100.0
    
    # ✅ FIX 3.4: Episode-level clip (FOR LOGGING/ANALYSIS ONLY - does NOT affect learning)
    # These bounds are used for:
    #   - Extreme episode detection logging
    #   - Result visualization scaling
    #   - Statistical outlier filtering
    # They are NOT enforced during training (would bias gradient)
    episode_reward_clip_min: float = -800.0
    episode_reward_clip_max: float = +600.0

    # ══════════════════════════════════════════════════════════
    # 11. DISTANCE SHAPING (sparse → dense bridge)
    # ══════════════════════════════════════════════════════════
    enable_distance_shaping: bool = True   # ✅ NEW: toggleable
    distance_shaping_max_per_uav: float = 1.0  # UNCHANGED
    
    # Future: implement distance-delta shaping to avoid local optimum
    # (requires state memory)