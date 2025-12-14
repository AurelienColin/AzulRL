# Task 3.3: Comprehensive Training Metrics

## Objective
Implement proper metrics tracking to diagnose training progress.

## Problem Statement
Current implementation only saves loss history to PNG:
```python
# run_rl.py:70-138
# Only plots to history.png, no detailed metrics
```

**Issues**:
1. Limited visibility into training dynamics
2. No win rate tracking
3. No invalid action rate monitoring
4. Cannot diagnose what's working/failing

## Proposed Solution
Track comprehensive metrics:

```python
metrics = {
    'episode': [],
    'loss_choose': [],
    'loss_eor': [],
    'win_rate': [],           # vs random agent
    'avg_score': [],
    'invalid_action_rate': [],
    'exploration_rate': [],
    'learning_rate': [],
    'avg_reward': [],
    'game_length': [],        # number of rounds
}
```

## Implementation Steps

### Step 3.3.1: Create metrics dictionary
**File**: `src/scripts/run_rl.py`
**Action**: Initialize comprehensive metrics tracking

### Step 3.3.2: Track loss per model
**File**: `src/scripts/run_rl.py`
**Action**: Log loss for choose_model and end_of_round_model separately

### Step 3.3.3: Implement win rate evaluation
**File**: `src/scripts/run_rl.py`
**Action**: Every N episodes, play against random baseline

### Step 3.3.4: Track invalid action attempts
**File**: `src/obj/bot_player.py`
**Action**: Count attempted invalid actions per episode

### Step 3.3.5: Log exploration and learning rate
**File**: `src/scripts/run_rl.py`
**Action**: Record epsilon and LR at each episode

### Step 3.3.6: Create visualization dashboard
**File**: `src/scripts/run_rl.py`
**Action**: Multi-panel plot showing all metrics

### Step 3.3.7: Save metrics to JSON
**File**: `src/scripts/run_rl.py`
**Action**: Persist metrics for later analysis

## Acceptance Criteria
- [ ] All key metrics tracked during training
- [ ] Win rate shows improvement over episodes
- [ ] Invalid action rate decreases
- [ ] Visualization shows all metrics in subplot grid

## Dependencies
- None (can be implemented incrementally)

## Estimated Complexity
Medium - many components but straightforward

## References
- Original code: `run_rl.py:70-138`
- TensorBoard logging patterns
