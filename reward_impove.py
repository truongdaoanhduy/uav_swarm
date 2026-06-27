import numpy as np 
def reward_func(factors):
       # factors[0]: current_coverage_ratio (0-1)
       # factors[1]: new_coverage_this_step (0-1)
       # factors[2]: dist_to_nearest_victim (float)
       # factors[3]: victims_discovered_this_step (float, cumulative or step)
       # factors[4]: battery_level (0-1)
       # factors[5]: is_charging (0 or 1)
       # factors[6]: dist_to_charger (float)
       # factors[7]: team_overlap_ratio (0-1)
       # factors[8]: battery_death_flag (0 or 1)
       # factors[9]: clustering_metric (0-1, higher = worse)
       # factors[10]: dist_to_zone_center (float)
       # factors[11]: collision_risk (0-1)

       # 1. Coverage & Search Progress
       r_coverage = np.clip(factors[0], 0.0, 1.0) * 0.2
       r_new_cov = np.clip(factors[1], 0.0, 1.0)

       # 2. Victim Discovery & Proximity
       r_victim_disc = np.clip(factors[3], 0.0, 1.0)
       r_victim_dist = np.clip(1.0 - factors[2] / 50.0, -1.0, 1.0)

       # 3. Battery & Charging Management
       r_battery = np.clip(factors[4], 0.0, 1.0) * 0.1
       r_charging = np.clip(factors[5] * (1.0 if factors[4] < 0.3 else 0.0), 0.0, 1.0)
       r_charger_dist = np.clip((1.0 - factors[6] / 50.0) * (1.0 if factors[4] < 0.3 else 0.0), -1.0, 1.0)
       r_death = -np.clip(factors[8], 0.0, 1.0)

       # 4. Cooperation & Spatial Distribution
       r_overlap = -np.clip(factors[7], 0.0, 1.0)
       r_cluster = -np.clip(factors[9], 0.0, 1.0)

       # 5. Zone Adherence & Safety
       r_zone = np.clip(1.0 - factors[10] / 50.0, -1.0, 1.0)
       r_collision = -np.clip(factors[11], 0.0, 1.0)

       # 6. Idle Penalty (Strict Definition)
       is_idle = (factors[1] == 0.0) and (factors[3] == 0.0) and (factors[5] == 0.0)
       r_idle = -0.5 if is_idle else 0.0

       # 7. Fixed Step Penalty for Time Efficiency
       r_step = -0.1

       # Sum and clip final reward
       reward = r_coverage + r_new_cov + r_victim_disc + r_victim_dist + \
                r_battery + r_charging + r_charger_dist + r_death + \
                r_overlap + r_cluster + r_zone + r_collision + r_idle + r_step

       reward = np.clip(reward, -5.0, 5.0)
       return float(reward)