# Task 1.1: Fix Reward Function

## Objective
Replace the diluted reward signal with a proper credit assignment mechanism.

## Problem Statement
Current implementation (`run_rl.py:55-58`):
```python
reward = score_increase * choice_one_hot / n_turns
```

This divides the total score gain equally across all turns, producing near-zero rewards that prevent learning.

## Proposed Solution
Implement **Policy Gradient with Discounted Returns**:

```python
# Calculate discounted returns for each timestep
gamma = 0.99  # Discount factor
returns = []
G = final_score_increase
for t in reversed(range(n_turns)):
    returns.insert(0, G)
    G = gamma * G

# Normalize returns (advantage estimation)
returns = np.array(returns)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

## Implementation Steps

### Step 1.1.1: Define discount factor in config
**File**: `src/config.py`
**Action**: Add `gamma: float = 0.99` parameter

### Step 1.1.2: Refactor reward calculation
**File**: `src/scripts/run_rl.py`
**Action**: Replace reward averaging with discounted returns

### Step 1.1.3: Add advantage normalization
**File**: `src/scripts/run_rl.py`
**Action**: Normalize returns by subtracting mean and dividing by std

### Step 1.1.4: Test reward magnitude
**Action**: Verify rewards have meaningful magnitude (not near-zero)

## Acceptance Criteria
- [ ] Rewards vary significantly between good and bad actions
- [ ] Training loss shows decreasing trend
- [ ] No division by n_turns in reward calculation

## Dependencies
None (first task in Phase 1)

## Estimated Complexity
Medium - core algorithm change

## References
- Original code: `run_rl.py:55-58`
- Policy gradient theory: Sutton & Barto Ch. 13
