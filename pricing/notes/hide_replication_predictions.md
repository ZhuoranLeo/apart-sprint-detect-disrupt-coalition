# Visibility experiment, replication: predictions written before running

Written 2026-10-24, 23:18 New York time, before running `hide` at rounds 90, 150 and 180.

## What already happened (round-120 copies, condition c1, 4 games per arm)
- B cannot see A (b_blind): A changed its price inside the window in 4/4 games, mostly upward
  (exploratory tests), and returned to its pre-window price in round 124 in 4/4 games.
- A cannot see B (a_blind) and placebo: A did not change its price (one 0.03 move in one placebo game).
- No secret undercutting of the kind Stigler (1964) predicts was observed.

## Replication design
- Same 4 c1 main runs, copies saved at rounds 90, 150 and 180 (window = the 3 rounds after the copy point,
  then 10 more rounds). 3 arms each: b_blind, a_blind, placebo. 36 new copies.
- Limitation: the copies come from the same 4 games, so they are not fully independent samples.

## Predictions
1. Main: the share of copies where A changes its price inside the window is clearly higher in b_blind
   than in a_blind and placebo.
2. Main: in b_blind, A returns to (within 0.02 of) its pre-window price in the first round in which B can
   see again.
3. a_blind and placebo: A's price inside the window stays within 0.02 of its pre-window price in most copies.
4. Exploratory, no directional prediction: whether A's in-window moves are upward or downward.

All outcomes will be reported, whether or not they match these predictions.
