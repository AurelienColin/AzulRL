# Task 2.2: Implement Action Masking

## Objective
Prevent the model from selecting invalid actions during training and inference.

## Problem Statement
Current implementation applies penalty at inference (`bot_player.py:92-104`):
```python
if color_count == 0:  # Invalid color selection
    best_plate_output[i_plate] = -1 * config.taboo_penalty
```

**Issues**:
1. Invalid actions are penalized only after selection
2. Model still trains on invalid action sequences
3. No mechanism to mask invalid actions before selection

## Proposed Solution
Implement **action masking** before softmax:

```python
def get_valid_action_mask(game_state, plates, n_colors):
    """Returns mask of valid (plate, color) combinations."""
    mask = np.zeros((len(plates), n_colors))
    for i, plate in enumerate(plates):
        for color_idx in range(n_colors):
            if plate.content[color_idx] > 0:
                mask[i, color_idx] = 1.0
    return mask

def internal_choice(self, game_state):
    # Get model logits
    plate_logits, color_logits, row_logits = self.choose_model(state)

    # Get valid action mask
    mask = get_valid_action_mask(game_state, plates, n_colors)

    # Apply mask: invalid actions get -inf before softmax
    masked_logits = plate_logits + (1 - mask) * (-1e9)

    # Now softmax gives zero probability to invalid actions
    probs = tf.nn.softmax(masked_logits)
```

## Implementation Steps

### Step 2.2.1: Create action mask generator
**File**: `src/obj/bot_player.py`
**Action**: Add `get_valid_action_mask()` function

### Step 2.2.2: Modify choose_model to accept mask
**File**: `src/obj/bot_player.py`
**Action**: Add mask parameter to model inference

### Step 2.2.3: Apply mask before softmax
**File**: `src/obj/bot_player.py`
**Action**: Mask logits with -inf for invalid actions

### Step 2.2.4: Handle row validity
**File**: `src/obj/bot_player.py`
**Action**: Create row mask based on player's left panel state

### Step 2.2.5: Update end_of_round action masking
**File**: `src/obj/bot_player.py`
**Action**: Mask invalid column placements

### Step 2.2.6: Remove taboo_penalty logic
**File**: `src/obj/bot_player.py`
**Action**: Remove penalty-based invalid action handling

## Acceptance Criteria
- [ ] Model never selects invalid actions
- [ ] Valid action mask correctly identifies legal moves
- [ ] Masking happens before probability calculation
- [ ] `taboo_penalty` no longer needed

## Dependencies
- Task 1.3 (Model Outputs) - softmax must be in place

## Estimated Complexity
Medium - requires understanding of game state to generate masks

## References
- Original code: `bot_player.py:89-104`
- Action masking in masked self-attention
