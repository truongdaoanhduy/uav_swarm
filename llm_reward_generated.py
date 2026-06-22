def reward_func(factors):
        # Unpack factors for clarity
        bat_pct = factors[0]
        new_area = factors[1]
        exp_ratio = factors[2]
        vic_found = factors[3]
        vic_urg = factors[4]
        dist_vic = factors[5]
        dist_chg = factors[6]
        chg_state = factors[7]
        bat_death = factors[8]
        near_team = factors[9]
        ep_prog = factors[10]
        active_uav = factors[11]

        r = 0.0

        # A. Victim search efficiency
        # factors[3]: Direct reward for finding victims
        r += vic_found * 15.0
        # factors[4]: Urgency weighting amplifies reward for critical victims
        r += vic_found * vic_urg * 10.0
        # factors[5]: Shaping reward for approaching undiscovered victims (inverse distance)
        r += 5.0 / (1.0 + dist_vic)

        # B. Coverage exploration
        # factors[1]: Reward for expanding search frontier
        r += new_area * 4.0
        # factors[2]: Penalty for redundant exploration in already mapped zones
        r -= exp_ratio * 6.0

        # C. Multi UAV cooperation & F. Collision avoidance
        # factors[9]: Penalize clustering/overlap to encourage spatial distribution
        r -= max(0.0, near_team - 1.0) * 5.0
        # factors[11]: Reward maintaining full team operational status
        r += active_uav * 3.0

        # D. Safety & Battery Management
        # factors[0]: Penalty for critically low battery to enforce survival
        if bat_pct < 25.0:
            r -= (25.0 - bat_pct) * 0.8
        # factors[6]: Dynamic penalty based on distance to charger when battery is low
        if bat_pct < 30.0:
            r -= (dist_chg / 100.0) * 4.0
        # factors[7]: Reward efficient charging behavior
        if chg_state == 1:
            r += 2.0 if bat_pct < 40.0 else -1.0
        # factors[8]: Severe penalty for battery depletion/crash
        if bat_death == 1:
            r -= 100.0

        # E. Mission completion speed
        # factors[10]: Time pressure penalty to discourage loitering
        r -= ep_prog * 2.0
        # Acceleration bonus: higher reward for victims found early in episode
        r += vic_found * (1.0 - ep_prog) * 5.0

        return r