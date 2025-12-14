# Task 2.1: Increase Network Capacity

## Objective
Expand network architecture to handle 100+ dimensional state space.

## Problem Statement
Current implementation (`bot_player.py:22-24`):
```python
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
```

**Issues**:
1. Only 2 hidden layers with decreasing width
2. 64→32 bottleneck for 100+ input features
3. Insufficient capacity for complex state-action mapping

## Proposed Solution
Expand to deeper, wider network with skip connections:

```python
# Option A: Wider network
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)

# Option B: Residual blocks
def residual_block(x, units):
    shortcut = x
    x = layers.Dense(units, activation='relu')(x)
    x = layers.Dense(units)(x)
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units)(shortcut)
    return layers.Add()([x, shortcut])
```

## Implementation Steps

### Step 2.1.1: Define architecture hyperparameters in config
**File**: `src/config.py`
**Action**: Add `hidden_layers: list = [256, 128, 64]`

### Step 2.1.2: Refactor choose_model architecture
**File**: `src/obj/bot_player.py`
**Action**: Use configurable layer sizes with optional residual connections

### Step 2.1.3: Refactor end_of_round_model architecture
**File**: `src/obj/bot_player.py`
**Action**: Match architecture improvements

### Step 2.1.4: Add dropout for regularization
**File**: `src/obj/bot_player.py`
**Action**: Add `layers.Dropout(0.2)` between dense layers

### Step 2.1.5: Add batch normalization
**File**: `src/obj/bot_player.py`
**Action**: Add `layers.BatchNormalization()` after activations

### Step 2.1.6: Test parameter count
**Action**: Verify total parameters are reasonable (~100k-1M)

## Acceptance Criteria
- [ ] Network has at least 3 hidden layers
- [ ] First hidden layer width >= input dimension
- [ ] Regularization (dropout) prevents overfitting
- [ ] Model trains without exploding/vanishing gradients

## Dependencies
- Phase 1 tasks should be complete first (architecture changes are secondary)

## Estimated Complexity
Medium - architecture redesign

## References
- Original code: `bot_player.py:22-38`
- Deep RL architectures (DQN, PPO)
