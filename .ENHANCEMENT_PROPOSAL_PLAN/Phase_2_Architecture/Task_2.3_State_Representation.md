# Task 2.3: Improve State Representation

## Objective
Optimize the game state representation for better learning.

## Problem Statement
Current state (`game.py:149-181`) includes:
- Bag tile counts (5) - potentially redundant
- Graveyard tile counts (5) - potentially redundant
- Plate contents (n_plates * 5)
- Central contents (5)
- First player token (1)
- Per-player states (~36 each)

**Issues**:
1. Recent fix removes bag/graveyard (10 features) - may lose useful info
2. State not normalized (counts range 0-20, penalties unbounded)
3. No feature engineering for strategic patterns

## Proposed Solution

### Normalization
```python
def normalize_state(state):
    # Tile counts: divide by max possible
    state[:n_colors] /= config.n_tile_per_color

    # Penalties: cap and normalize
    state[penalty_idx] = min(state[penalty_idx], 20) / 20

    return state
```

### Feature Engineering
Add derived features:
- `tiles_remaining_ratio`: total tiles left / initial tiles
- `round_progress`: current_round / max_rounds
- `score_difference`: current_player_score - max_opponent_score
- `row_completion_status`: how close each row is to completion

## Implementation Steps

### Step 2.3.1: Add state normalization function
**File**: `src/obj/game.py` or `src/utils.py`
**Action**: Create `normalize_state()` function

### Step 2.3.2: Normalize tile counts
**Action**: Divide by `config.n_tile_per_color` (20)

### Step 2.3.3: Normalize penalty values
**Action**: Cap and scale to [0, 1] range

### Step 2.3.4: Add strategic features
**File**: `src/obj/game.py`
**Action**: Append derived features to state vector

### Step 2.3.5: Update input_length calculation
**File**: `src/obj/bot_player.py`
**Action**: Adjust for new state size

### Step 2.3.6: Re-evaluate state slicing
**File**: `src/obj/bot_player.py`
**Action**: Decide if `start_input_index` is still appropriate

## Acceptance Criteria
- [ ] All state features normalized to [0, 1] or [-1, 1]
- [ ] No feature has unbounded range
- [ ] Strategic features improve learning signal

## Dependencies
- None (can be done in parallel)

## Estimated Complexity
Medium - requires understanding of all state components

## References
- Original code: `game.py:149-181`
- `bot_player.py:67-69` (start_input_index)
