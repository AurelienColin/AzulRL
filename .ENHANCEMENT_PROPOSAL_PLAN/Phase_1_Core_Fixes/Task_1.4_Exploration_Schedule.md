# Task 1.4: Implement Exploration Decay Schedule

## Objective
Replace fixed 10% exploration with a decaying schedule to enable convergence.

## Problem Statement
Current implementation (`bot_player.py:86`):
```python
if np.random.rand() < 0.1:  # Always 10% random
    # Random action
```

**Issues**:
1. Fixed exploration prevents policy convergence
2. Model never fully exploits learned behavior
3. No relationship between training progress and exploration

## Proposed Solution
Implement **epsilon-greedy with exponential decay**:

```python
class BotPlayer:
    def __init__(self, ...):
        self.epsilon_start = 1.0      # Initial exploration
        self.epsilon_end = 0.01       # Final exploration
        self.epsilon_decay = 0.995    # Decay per episode
        self.epsilon = self.epsilon_start

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_end,
                          self.epsilon * self.epsilon_decay)

    def internal_choice(self, game_state):
        if np.random.rand() < self.epsilon:
            return self._random_action()
        else:
            return self._policy_action(game_state)
```

## Implementation Steps

### Step 1.4.1: Add exploration parameters to config
**File**: `src/config.py`
**Action**: Add `epsilon_start`, `epsilon_end`, `epsilon_decay` constants

### Step 1.4.2: Add epsilon attribute to BotPlayer
**File**: `src/obj/bot_player.py`
**Action**: Initialize `self.epsilon` in `__init__`

### Step 1.4.3: Implement decay method
**File**: `src/obj/bot_player.py`
**Action**: Add `decay_exploration()` method

### Step 1.4.4: Modify internal_choice
**File**: `src/obj/bot_player.py`
**Action**: Use `self.epsilon` instead of hardcoded 0.1

### Step 1.4.5: Call decay in training loop
**File**: `src/scripts/run_rl.py`
**Action**: Call `player.decay_exploration()` after each episode

### Step 1.4.6: Log exploration rate
**File**: `src/scripts/run_rl.py`
**Action**: Track and visualize epsilon over training

## Acceptance Criteria
- [ ] Exploration rate decreases over training
- [ ] Final exploration rate reaches `epsilon_end`
- [ ] Model behavior becomes more deterministic over time

## Dependencies
None (can be done in parallel with other Phase 1 tasks)

## Estimated Complexity
Low - parameter addition and simple logic change

## References
- Original code: `bot_player.py:86`
- Epsilon-greedy exploration strategy
