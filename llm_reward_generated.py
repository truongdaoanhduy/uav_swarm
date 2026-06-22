import numpy as np

def reward_func(factors):
        try:
                battery = float(factors[0])
                new_explored = float(factors[1])
                nearby_explored = float(factors[2])
                victims_found = float(factors[3])
                avg_urgency = float(factors[4])
                dist_to_victim = float(factors[5])
                dist_to_charger = float(factors[6])
                is_charging = float(factors[7])
                battery_death = float(factors[8])
                nearby_teammates = float(factors[9])
                episode_progress = float(factors[10])
                active_uavs = float(factors[11])
        except Exception:
                return 0.0

        r_victim = victims_found * (1.0 + avg_urgency * 0.4)
        r_explore = new_explored * 15.0
        p_redundant = -nearby_explored * 8.0

        dist_v = min(max(dist_to_victim, 0.0), 250.0)
        r_approach = np.exp(-dist_v / 80.0) * 3.0

        p_cluster = -nearby_teammates * 2.5

        r_battery = battery * 0.5
        p_low_battery = -max(0.0, 0.25 - battery) * 20.0
        r_charging = is_charging * 4.0 if battery < 0.4 else 0.0

        dist_c = min(max(dist_to_charger, 0.0), 250.0)
        r_charger_nav = np.exp(-dist_c / 50.0) * 2.0 if battery < 0.3 else 0.0

        p_death = -battery_death * 100.0
        p_time = -min(episode_progress, 1.0) * 1.0
        p_team_loss = -(4.0 - active_uavs) * 15.0

        reward = (r_victim + r_explore + r_approach + r_battery + r_charging + r_charger_nav +
                        p_redundant + p_cluster + p_low_battery + p_death + p_time + p_team_loss)

        return float(np.clip(reward, -100.0, 100.0))
