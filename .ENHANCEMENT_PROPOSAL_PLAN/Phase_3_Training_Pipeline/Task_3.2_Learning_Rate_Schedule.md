# Task 3.2: Implement Learning Rate Schedule

## Objective
Add adaptive learning rate scheduling for stable convergence.

## Problem Statement
Current implementation (`run_rl.py:83-86`):
```python
model.compile(optimizer=AdamW(learning_rate=learning_rate), ...)
```

**Issues**:
1. Fixed learning rate (0.001) throughout training
2. May be too high early (instability) or too late (slow convergence)
3. No adaptation to training dynamics

## Proposed Solution
Implement **cosine annealing with warm restarts**:

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def cosine_schedule(epoch, total_epochs, lr_initial, lr_min):
    return lr_min + 0.5 * (lr_initial - lr_min) * (
        1 + np.cos(np.pi * epoch / total_epochs)
    )

# Or use built-in scheduler
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    alpha=0.01  # Minimum LR ratio
)
optimizer = AdamW(learning_rate=lr_schedule)
```

## Implementation Steps

### Step 3.2.1: Define LR schedule parameters in config
**File**: `src/config.py`
**Action**: Add `lr_initial`, `lr_min`, `lr_decay_steps`

### Step 3.2.2: Implement cosine decay schedule
**File**: `src/scripts/run_rl.py`
**Action**: Create LR schedule using `tf.keras.optimizers.schedules`

### Step 3.2.3: Apply schedule to optimizer
**File**: `src/scripts/run_rl.py`
**Action**: Pass schedule to AdamW optimizer

### Step 3.2.4: Log learning rate over training
**File**: `src/scripts/run_rl.py`
**Action**: Track LR alongside loss for visualization

### Step 3.2.5: Add warmup period (optional)
**File**: `src/scripts/run_rl.py`
**Action**: Linear warmup for first N steps

## Acceptance Criteria
- [ ] Learning rate decreases over training
- [ ] Training remains stable (no loss explosions)
- [ ] Final LR reaches configured minimum

## Dependencies
- Phase 1 tasks should be complete first

## Estimated Complexity
Low - standard Keras functionality

## References
- Original code: `run_rl.py:83-86`
- Cosine annealing paper (Loshchilov & Hutter, 2017)
