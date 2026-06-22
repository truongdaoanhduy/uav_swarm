import numpy as np

def reward_func(factors):
    # Convert to numpy array for safe numerical operations
    f = np.asarray(factors, dtype=np.float32)
    
    # Extract factors
    battery = f[0]
    new_area = f[1]
    explored_ratio = f[2]
    victims_found = f[3]
    avg_urgency = f[4]
    dist_victim = f[5]
    dist_charger = f[6]
    charging = f[7]
    battery_death = f[8]
    teammates_nearby = f[9]
    episode_progress = f[10]
    active_uavs = f[11]
    
    # A. Search Efficiency
    # Strong reward for finding victims, scaled by urgency priority
    r_victim = victims_found * (2.0 + avg_urgency * 0.5)
    # Mild shaping to guide UAVs toward undiscovered victims
    r_guidance = -np.clip(dist_victim, 0.0, 250.0) * 0.02
    
    # B. Coverage Optimization
    # Reward novel exploration
    r_coverage = new_area * 0.5
    # Penalize redundant exploration of known areas
    r_redundancy = -explored_ratio * 0.5
    
    # C. Multi-UAV Cooperation
    # Penalize clustering to encourage spatial distribution
    r_spread = -teammates_nearby * 0.2
    
    # D. Safety & Battery Management
    # Penalize operating in dangerous low-battery zones
    r_low_batt = -np.maximum(0.0, 0.3 - battery) * 2.0
    # Reward charging when low, penalize charging when sufficient
    r_charging = charging * np.where(battery < 0.4, 0.5, -0.2)
    # Guide toward charger only when battery is critically low
    r_charger_guide = -np.clip(dist_charger, 0.0, 250.0) * 0.01 * np.where(battery < 0.3, 1.0, 0.0)
    # Heavy penalty for permanent UAV loss
    r_death = -battery_death * 15.0
    
    # E. Time Efficiency
    # Constant step penalty to discourage wasting time
    r_step = -0.05
    # Reward steady progress toward mission completion
    r_progress = episode_progress * 0.1
    
    # F. Team Survival
    # Continuous bonus for keeping all UAVs operational
    r_survival = active_uavs * 0.05
    
    # Aggregate reward
    total_reward = (r_victim + r_guidance + 
                    r_coverage + r_redundancy + 
                    r_spread + 
                    r_low_batt + r_charging + r_charger_guide + r_death + 
                    r_step + r_progress + 
                    r_survival)
    
    # Clip for training stability while preserving signal direction
    return float(np.clip(total_reward, -20.0, 20.0))