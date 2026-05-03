ACTOR OBS (68 dims) - LOCAL ONLY:
┌─────────────────────────────────────────────┐
│ Part 1: Self State [0:11] 11 dims │
│ pos_x, pos_y, alt → [0,1] │
│ vel_x, vel_y, vel_z → [-1,1] │
│ battery → [0,1] │
│ state_onehot (4) → {0,1} │
├─────────────────────────────────────────────┤
│ Part 2: Stations [11:19] 8 dims │
│ 2 × [rel_x, rel_y, dist, occupancy] │
├─────────────────────────────────────────────┤
│ Part 3: Teammates [19:28] 9 dims │
│ 3 × [dist, bearing, rel_alt] │
├─────────────────────────────────────────────┤
│ Part 4: Obstacles FOV [28:40] 12 dims │
│ 4 × [rel_x, rel_y, type_id] │
├─────────────────────────────────────────────┤
│ Part 5: Victims FOV [40:65] 25 dims │
│ 5 × [rel_x, rel_y, dist, urgency, found] │
├─────────────────────────────────────────────┤
│ Part 6: Coverage [65:68] 3 dims │
│ local_15m, local_30m, time_remain │
└─────────────────────────────────────────────┘

CRITIC OBS: N_UAV × 68 + 7 global dims
