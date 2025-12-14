# Task 1.3: Add Softmax to Model Outputs

## Objective
Convert model outputs from raw logits to proper probability distributions.

## Problem Statement
Current implementation (`bot_player.py:26-38`):
```python
plate_head = layers.Dense(n_plates + 1)(x)  # Raw logits
color_head = layers.Dense(config.n_colors)(x)  # Raw logits
row_head = layers.Dense(config.n_colors)(x)  # Raw logits
```

**Issues**:
1. Raw logits don't represent probabilities
2. Training targets are one-hot (probability distributions)
3. Mismatch between output space and target space

## Proposed Solution
Add softmax activation to output heads:

```python
plate_head = layers.Dense(n_plates + 1, activation='softmax')(x)
color_head = layers.Dense(config.n_colors, activation='softmax')(x)
row_head = layers.Dense(config.n_colors, activation='softmax')(x)
```

**Alternative**: Use log-softmax for numerical stability with policy gradient:
```python
plate_head = layers.Dense(n_plates + 1)(x)
plate_head = tf.nn.log_softmax(plate_head)
```

## Implementation Steps

### Step 1.3.1: Add softmax to choose_model
**File**: `src/obj/bot_player.py`
**Action**: Add softmax activation to plate_head, color_head, row_head

### Step 1.3.2: Add softmax to end_of_round_model
**File**: `src/obj/bot_player.py`
**Action**: Add softmax activation to each of the 5 output heads

### Step 1.3.3: Remove inference-time softmax
**File**: `src/obj/bot_player.py`
**Action**: Remove `softmax()` calls in `internal_choice()` if model outputs are now probabilities

### Step 1.3.4: Verify probability sums
**Action**: Assert that output probabilities sum to 1.0

## Acceptance Criteria
- [ ] All model output heads have softmax activation
- [ ] Output probabilities sum to 1.0 (within tolerance)
- [ ] Inference still works correctly with argmax

## Dependencies
None (can be done in parallel with Task 1.1 and 1.2)

## Estimated Complexity
Low - straightforward architecture change

## References
- Original code: `bot_player.py:26-38`
- Softmax definition for classification
