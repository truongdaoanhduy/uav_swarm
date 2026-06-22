def reward_func(factors):
       import math

       battery_pct           = factors[0]
       coverage_delta        = factors[1]
       local_cov             = factors[2]
       n_found               = factors[3]
       avg_urgency           = factors[4]
       dist_unfound          = factors[5]
       dist_station          = factors[6]
       is_charging           = factors[7]
       is_dead               = factors[8]
       n_nearby              = factors[9]
       time_ratio            = factors[10]
       n_active              = factors[11]
       high_urgency_unfound  = factors[12]
       dist_high_urgency     = factors[13]
       local_cov_delta       = factors[14]
       idle_steps            = factors[15]
       high_urgency_ratio    = factors[16]
       neighbor_coverage     = factors[17]

       reward = 0.0

       # 1. Victim discovery (urgency-scaled with time-decay)
       if n_found > 0:
           # Base reward scales with urgency: urg=1 -> ~30, urg=5 -> ~100
           urgency_factor = 1.0 + 0.8 * (avg_urgency / 5.0)
           # Time decay: early finds rewarded more, but high urgency decays slower
           time_decay = 1.0 + 0.5 * (1.0 - time_ratio) * (1.0 - 0.3 * (avg_urgency / 5.0))
           reward += 30.0 * urgency_factor * time_decay * n_found

       # 2. Coverage (time-aware with anti-redundancy)
       if coverage_delta > 0.0001:
           # Scale by remaining time: early game coverage is more valuable
           time_weight = 1.0 - 0.5 * time_ratio
           cov_reward = coverage_delta * 300.0 * time_weight
           # Penalize redundant scanning in over-explored zones
           if neighbor_coverage > 0.8:
               cov_reward *= 0.2
           reward += cov_reward

       # 3. Idle penalty (adaptive threshold)
       # Threshold adapts: early game allows more exploration (up to 15), late game strict (5)
       idle_threshold = 5.0 + 10.0 * (1.0 - time_ratio)
       if idle_steps > idle_threshold:
           # Progressive penalty, capped at -30
           idle_penalty = -2.0 * (idle_steps - idle_threshold)
           reward += max(idle_penalty, -30.0)

       # 4. Urgency priority (soft guidance)
       if high_urgency_unfound > 0:
           # Encourage moving toward critical victims
           # Normalize distance to [0,1], scale by count
           dist_norm = min(dist_high_urgency / 350.0, 1.0)
           urgency_guidance = -3.0 * dist_norm * min(high_urgency_unfound, 3.0) / 3.0
           reward += urgency_guidance

       # 5. Battery management
       if is_dead > 0.5:
           reward -= 200.0
       elif battery_pct < 30.0:
           # Potential-based shaping: smooth penalty increases as battery drops & distance grows
           battery_potential = -0.5 * (30.0 - battery_pct) * (dist_station / 200.0)
           reward += battery_potential
           # Small reward for landing when critically low
           if is_charging > 0.5:
               reward += 5.0

       # 6. Collision avoidance
       if n_nearby > 2:
           reward -= 5.0 * (n_nearby - 2)

       return float(reward)