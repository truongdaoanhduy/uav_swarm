def reward_func(factors):
    # factors[0]: battery percentage. Used for survival and charging decisions.
    battery = factors[0]
    # factors[1]: new area covered this step. Used to measure exploration progress.
    new_coverage = factors[1]
    # factors[2]: nearby explored ratio. Used to detect wasted repeated exploration.
    explored_ratio = factors[2]
    # factors[3]: number of victims discovered. Used for rescue success reward.
    victims_found = factors[3]
    # factors[4]: victim urgency. Used to prioritize important victims.
    urgency = factors[4]
    # factors[5]: distance to nearest undiscovered victim. Used to encourage movement toward targets.
    dist_to_victim = factors[5]
    # factors[6]: distance to charging station. Used for battery planning.
    dist_to_charger = factors[6]
    # factors[7]: charging state. Used to evaluate charging behavior.
    charging_state = factors[7]
    # factors[8]: battery death event. Used as severe failure penalty.
    battery_death = factors[8]
    # factors[9]: nearby teammates. Used for cooperation and overlap avoidance.
    nearby_teammates = factors[9]
    # factors[10]: episode progress. Used for completion speed optimization.
    episode_progress = factors[10]
    # factors[11]: number of active UAVs. Used to evaluate team survival.
    active_uavs = factors[11]

    reward = 0.0

    # A. Victim discovery efficiency
    # factors[3] & factors[4]: Reward discovering victims, scaled by urgency to prioritize critical rescues
    reward += victims_found * urgency * 15.0

    # B. Exploration coverage
    # factors[1]: Reward new unique area coverage to encourage active searching
    reward += new_coverage * 8.0
    # factors[2]: Penalize redundant exploration in already explored regions to prevent wasting steps
    reward -= explored_ratio * 3.0

    # C. Active search behavior & Idle penalty
    # Check if UAV made meaningful progress this step
    made_progress = (new_coverage > 0) or (victims_found > 0)
    moving_toward_target = (dist_to_victim < 15.0)
    charging_when_needed = (charging_state == 1 and battery < 30.0)

    if not made_progress and not moving_toward_target and not charging_when_needed:
        # Strong penalty for idle/no-progress steps
        reward -= 2.0
        # Scale penalty with episode progress: longer unproductive periods yield heavier cumulative penalties
        reward -= episode_progress * 1.5

    # D. Multi-UAV cooperation
    # factors[9]: Penalize unnecessary clustering to encourage distributed search and avoid duplicated efforts
    reward -= max(0, nearby_teammates - 1) * 1.5

    # E. Battery and survival
    # factors[8]: Severe penalty for battery death to enforce survival priority
    if battery_death:
        reward -= 100.0
    # factors[7] & factors[0]: Reward correct charging decisions based on battery level
    if charging_state == 1:
        if battery < 30.0:
            reward += 3.0  # Charging when needed
        elif battery > 80.0:
            reward -= 2.0  # Wasting time charging when full
    # factors[0]: General survival bonus for maintaining operational battery
    reward += battery * 0.05

    # F. Time optimization
    # factors[10]: Penalize slow completion to encourage faster mission finish and reduce unnecessary steps
    reward -= episode_progress * 1.0

    # G. Safety & Navigation incentives
    # factors[5]: Reward moving closer to undiscovered victims to guide active navigation
    reward += max(0, 20.0 - dist_to_victim) * 0.2
    # factors[6]: Reward moving toward charger when battery is low to prevent unsafe low-battery behavior
    if battery < 30.0:
        reward += max(0, 20.0 - dist_to_charger) * 0.2

    # Team survival tracking
    # factors[11]: Penalize loss of UAVs to maintain team integrity and cooperative capability
    reward -= (4 - active_uavs) * 20.0

    return float(reward)