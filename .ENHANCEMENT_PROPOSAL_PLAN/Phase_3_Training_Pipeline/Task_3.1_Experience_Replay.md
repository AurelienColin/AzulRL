# Task 3.1: Implement Experience Replay

## Objective
Add experience replay buffer to stabilize training and improve sample efficiency.

## Problem Statement
Current implementation trains immediately on each round:
```python
# In train_on_round()
model.train_on_batch(state, reward)  # Single sample training
```

**Issues**:
1. High correlation between consecutive samples
2. Catastrophic forgetting of earlier experiences
3. Inefficient use of collected data (each sample used once)

## Proposed Solution
Implement **replay buffer** with prioritized sampling:

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)
```

## Implementation Steps

### Step 3.1.1: Create ReplayBuffer class
**File**: `src/obj/replay_buffer.py` (new file)
**Action**: Implement basic replay buffer with push/sample methods

### Step 3.1.2: Add buffer to training script
**File**: `src/scripts/run_rl.py`
**Action**: Initialize replay buffer before training loop

### Step 3.1.3: Store experiences in buffer
**File**: `src/scripts/run_rl.py`
**Action**: Push (state, action, reward) tuples to buffer after each turn

### Step 3.1.4: Sample batches for training
**File**: `src/scripts/run_rl.py`
**Action**: Replace single-sample training with batch sampling

### Step 3.1.5: Add minimum buffer size check
**File**: `src/scripts/run_rl.py`
**Action**: Only start training after buffer has minimum samples

### Step 3.1.6: Configure buffer parameters
**File**: `src/config.py`
**Action**: Add `replay_buffer_size`, `batch_size`, `min_buffer_size`

## Acceptance Criteria
- [x] Replay buffer stores experiences correctly
- [x] Training uses random batches from buffer
- [x] Correlation between training samples reduced
- [x] Sample efficiency improved (measure avg training loss)

## Implementation Notes (2025-12-14)
- Created `src/obj/replay_buffer.py` with `ReplayBuffer` class
- Added experience replay parameters to `config.py`: `replay_buffer_size`, `batch_size`, `min_buffer_size`, `train_every_n_rounds`
- Refactored `train_on_round()` to `collect_round_experiences()` for separation of concerns
- Added `train_from_buffer()` function for batch sampling and training
- Each model trains independently when its buffer reaches minimum size
- Fixed `compute_discounted_returns()` to handle 2D reward arrays
- Fixed model weights file extension to use `.weights.h5` (Keras 3 requirement)

## Dependencies
- Phase 1 and Phase 2 tasks should be complete

## Estimated Complexity
Medium - new component but standard implementation

## References
- DQN paper (Mnih et al., 2015)
- Experience replay best practices
