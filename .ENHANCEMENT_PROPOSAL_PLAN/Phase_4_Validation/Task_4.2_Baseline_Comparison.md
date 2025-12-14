# Task 4.2: Baseline Agent Comparison

## Objective
Create baseline agents to measure RL training progress.

## Problem Statement
- No objective measure of learning progress
- Cannot distinguish "learning" from "lucky variance"
- Need comparison points for win rate

## Proposed Solution
Implement multiple baseline agents:

1. **Random Agent**: Selects uniformly random valid actions
2. **Greedy Agent**: Always selects action with highest immediate score
3. **Heuristic Agent**: Uses simple Azul strategy rules

```python
class RandomAgent(Player):
    def internal_choice(self, game_state):
        valid_actions = get_valid_actions(game_state)
        return random.choice(valid_actions)

class GreedyAgent(Player):
    def internal_choice(self, game_state):
        # Try each action, return one with best immediate reward
        best_action = None
        best_score = -np.inf
        for action in get_valid_actions(game_state):
            score = evaluate_action(game_state, action)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
```

## Implementation Steps

### Step 4.2.1: Create RandomAgent class
**File**: `src/obj/random_agent.py` (new file)
**Action**: Implement agent that selects random valid actions

### Step 4.2.2: Create GreedyAgent class
**File**: `src/obj/greedy_agent.py` (new file)
**Action**: Implement agent that maximizes immediate score

### Step 4.2.3: Create evaluation harness
**File**: `src/scripts/evaluate.py` (new file)
**Action**: Run N games between agents, compute statistics

### Step 4.2.4: Add evaluation to training loop
**File**: `src/scripts/run_rl.py`
**Action**: Periodically evaluate vs baselines during training

### Step 4.2.5: Track win rate over training
**File**: `src/scripts/run_rl.py`
**Action**: Plot win rate vs each baseline

### Step 4.2.6: Create performance report
**Action**: Generate summary of model vs baseline performance

## Acceptance Criteria
- [ ] Random agent works correctly
- [ ] Greedy agent beats random agent
- [ ] RL agent win rate vs random increases over training
- [ ] RL agent eventually beats greedy agent

## Expected Performance Hierarchy
After successful training:
1. RL Agent > Greedy Agent > Random Agent
2. RL Agent win rate vs random > 80%
3. RL Agent win rate vs greedy > 60%

## Dependencies
- Phase 3 tasks (training metrics)

## Estimated Complexity
Medium - requires implementing agent variants
