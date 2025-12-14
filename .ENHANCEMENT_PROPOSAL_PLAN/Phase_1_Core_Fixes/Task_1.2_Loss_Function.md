# Task 1.2: Fix Loss Function

## Objective
Replace the masked MAE loss with appropriate loss functions for policy learning.

## Problem Statement
Current implementation (`run_rl.py:12-19`):
```python
def masked_mae(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    mask = mask + 1E-8  # Creates non-zero gradient on zeros
    mae = tf.abs(y_true - y_pred) * mask
    return tf.reduce_sum(mae) / tf.reduce_sum(mask)
```

**Issues**:
1. MAE is inappropriate for probability outputs
2. Epsilon addition causes gradient leakage
3. Sparse targets (one-hot) with MAE = poor signal

## Proposed Solution
Use **Policy Gradient Loss** (negative log-likelihood weighted by advantage):

```python
def policy_loss(y_true, y_pred):
    """
    y_true: one-hot action + advantage in last position
    y_pred: log probabilities from model
    """
    action = y_true[:, :-1]  # One-hot action
    advantage = y_true[:, -1:]  # Advantage value

    # Categorical cross-entropy weighted by advantage
    log_prob = tf.reduce_sum(action * y_pred, axis=-1)
    loss = -tf.reduce_mean(log_prob * advantage)
    return loss
```

## Implementation Steps

### Step 1.2.1: Remove masked_mae function
**File**: `src/scripts/run_rl.py`
**Action**: Delete `masked_mae` function

### Step 1.2.2: Implement policy gradient loss
**File**: `src/scripts/run_rl.py`
**Action**: Add `policy_loss` function with advantage weighting

### Step 1.2.3: Update model compilation
**File**: `src/scripts/run_rl.py`
**Action**: Change loss from `masked_mae` to `policy_loss`

### Step 1.2.4: Modify training data format
**File**: `src/scripts/run_rl.py`
**Action**: Append advantage value to one-hot encoded targets

## Acceptance Criteria
- [ ] Loss function uses log probabilities
- [ ] Advantages weight the gradients
- [ ] No epsilon hack in loss calculation
- [ ] Loss decreases during training

## Dependencies
- Task 1.1 (Reward Function) - needs advantage values

## Estimated Complexity
Medium - requires coordinated changes to loss and data format

## References
- Original code: `run_rl.py:12-19`
- REINFORCE algorithm
