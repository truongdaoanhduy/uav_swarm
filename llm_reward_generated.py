def reward_func(factors):
        # Extract factors
        battery = factors[0]
        explore_progress = factors[1]
        redundancy = factors[2]
        discovery = factors[3]
        importance = factors[4]
        dist_unexplored = factors[5]
        dist_charger = factors[6]
        charging = factors[7]
        failure = factors[8]
        neighbors = factors[9]
        mission_progress = factors[10]
        team_status = factors[11]

        # Normalize/Scale safely
        # Assume inputs might be raw, so apply safe scaling
        # I'll use np.clip and sigmoid-like transformations to keep things bounded
        # But the prompt says "Use normalized values whenever possible", so I'll assume they are roughly in [0,1] or reasonable ranges, but I'll add robust scaling.

        # A. Mission Progress
        r_mission = discovery * importance

        # B. Exploration & Coverage
        r_explore = explore_progress - redundancy

        # C. Cooperation (Spatial Diversity)
        # Assume neighbors factor represents proximity/density (higher = closer/more clustered)
        # Reward dispersion: 1 - neighbors (if normalized)
        r_coop = 1.0 - neighbors

        # D. Safety & Energy
        # Battery reward: encourage maintaining high battery
        r_battery = battery
        # Low battery penalty
        r_low_battery = -np.clip(battery - 0.3, -1, 0) # Penalty when < 0.3
        # Charging reward
        r_charging = charging
        # Incentive to move to charger when low
        r_recharge_incentive = (1.0 - dist_charger) * np.clip(0.3 - battery, 0, 1)
        # Failure penalty
        r_failure = -failure * 5.0

        # E. Time Efficiency
        # Small step penalty to encourage speed, scaled by team status
        r_time = -0.01 * team_status

        # F. Navigation/Direction shaping (optional but helps)
        # Encourage moving towards unexplored areas
        r_navigation = (1.0 - dist_unexplored) * 0.5

        # Combine with weights
        # Weights should be tuned but kept general
        w_mission = 2.0
        w_explore = 1.0
        w_coop = 0.5
        w_battery = 0.5
        w_low_battery = 1.0
        w_charging = 1.5
        w_recharge = 1.0
        w_failure = 5.0
        w_time = 0.05
        w_nav = 0.3

        total_reward = (w_mission * r_mission +
                        w_explore * r_explore +
                        w_coop * r_coop +
                        w_battery * r_battery +
                        w_low_battery * r_low_battery +
                        w_charging * r_charging +
                        w_recharge * r_recharge_incentive +
                        w_failure * r_failure +
                        w_time * r_time +
                        w_nav * r_navigation)

        # Scale by team operational status to reduce reward when team is degraded
        total_reward *= team_status

        return float(total_reward)