def reward_func(factors):
    reward = 0.0

    # Coverage reward scaled by unexploredness and episode progress
    coverage_weight = 1 - factors[10]
    reward += factors[1] * (1 - factors[2]) * 5.0 * coverage_weight

    # Victims reward scaled by urgency and episode progress
    victim_weight = 1 + factors[10]
    reward += factors[3] * factors[4] * 2.0 * victim_weight
    # Battery penalties
    if factors[8] == 1:  # Battery died
        reward -= 100.0
    else:
        if factors[0] < 20 and factors[7] == 0:  # Low battery and not charging
            reward -= factors[6] * 0.5  # Penalize distance to charging station

    # Redundancy penalty for nearby teammates
    reward -= factors[9] * 0.5

    return float(reward)