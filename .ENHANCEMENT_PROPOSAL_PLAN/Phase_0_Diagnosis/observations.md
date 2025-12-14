# Phase 0: Diagnosis - Codebase Observations

## Executive Summary
The AzulRL project implements the Azul board game with neural network-based bots. The RL training is failing because:
1. **Reward signal is too weak/sparse** - rewards distributed across turns dilute learning signal
2. **Loss function masking issues** - masked MAE doesn't properly propagate gradients
3. **No action validity enforcement** during training
4. **Fixed exploration rate** prevents convergence

---

## 1. Project Structure Analysis

### Directory Layout
```
AzulRL/
├── src/
│   ├── config.py              # Global configuration constants
│   ├── utils.py               # Utility functions (one-hot encoding)
│   ├── obj/                   # Core game objects
│   │   ├── bag.py             # Tile bag management
│   │   ├── central.py         # Central plate with first-player token
│   │   ├── container.py       # Base container class (5 colors)
│   │   ├── plate.py           # Individual plates
│   │   ├── player.py          # Human player (base class)
│   │   ├── bot_player.py      # RL bot player
│   │   └── game.py            # Main game loop
│   └── scripts/
│       ├── run_rl.py          # RL training script
│       └── run_manual.py      # Manual gameplay
├── res/models/                # Model weights storage
└── test/                      # Empty test directory
```

### Code Quality Assessment
| Aspect | Status | Notes |
|--------|--------|-------|
| Type hints | Partial | Some functions lack type annotations |
| Documentation | Minimal | Few docstrings |
| Test coverage | None | Empty test directory |
| Code organization | Good | Clean class hierarchy |

---

## 2. Game Implementation Review

### Game Mechanics (Correct)
- **5 colors**: black, white, red, blue, yellow
- **Tile distribution**: 20 per color (100 total)
- **Player boards**: Left panel (staging) + Right panel (5x5 scoring grid)
- **Scoring**: Points for placement + bonuses for complete rows/columns/colors

### Game State Representation
**File**: `src/obj/game.py:149-181`

State vector components:
| Index Range | Content | Size |
|-------------|---------|------|
| 0-4 | Bag tile counts | 5 |
| 5-9 | Graveyard tile counts | 5 |
| 10-(10+n_plates*5) | Plate tile counts | n_plates * 5 |
| After plates | Central tile counts | 5 |
| +1 | First player token | 1 |
| Per player | Left panel (count+color per row) | n_colors * 2 |
| Per player | Right panel (5x5 grid) | n_colors^2 |
| Per player | Penalties | 1 |
| End | Current player one-hot | n_players |

**Total state size**: ~100+ features (depends on player count)

### Issues Found in Game Logic
1. **Hard round limit**: `game.py:122` limits to 15 rounds, potentially truncating games
2. **No game end detection**: Missing natural termination when players complete boards

---

## 3. RL Training Pipeline Analysis

### Model Architecture
**File**: `src/obj/bot_player.py:16-43`

**Model 1 (choose_model)**:
```
Input → Dense(64, ReLU) → Dense(32, ReLU) →
├─ plate_head: Dense(n_plates+1)  [no softmax]
├─ color_head: Dense(5)           [no softmax]
└─ row_head: Dense(5)             [no softmax]
```

**Model 2 (end_of_round_model)**:
```
Input → Dense(64, ReLU) → Dense(32, ReLU) →
└─ 5 parallel heads: Dense(5) each [no softmax]
```

### Training Loop
**File**: `src/scripts/run_rl.py:22-67`

```python
def train_on_round(game, choose_model, end_of_round_model):
    # Collect choices and states during round
    choices = {player: [] for player in game.players}
    game_states = {player: [] for player in game.players}

    # Play round, collecting data
    game.round()
    game.end_of_round()

    # Calculate rewards (ISSUE: sparse signal)
    for player in game.players:
        score_increase = player.score - player.prev_score
        reward = score_increase * choice_one_hot / n_turns  # Diluted!
        model.train_on_batch(state, reward)
```

### Loss Function
**File**: `src/scripts/run_rl.py:12-19`

```python
def masked_mae(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    mask = mask + 1E-8  # Avoid division by zero
    mae = tf.abs(y_true - y_pred) * mask
    return tf.reduce_sum(mae) / tf.reduce_sum(mask)
```

---

## 4. Critical Issues Identified

### ISSUE 1: Reward Signal Dilution (CRITICAL)
**Location**: `run_rl.py:55-58`
**Severity**: Critical

The reward for each action is calculated as:
```python
reward = score_increase * choice_one_hot / n_turns
```

**Problem**: A player's score increase happens ONCE at round end, then gets divided by total turns. If a player scores +10 in 20 turns, each turn gets ~0.5 reward. The model cannot distinguish which specific actions led to good outcomes.

**Impact**: Near-zero gradient signal, models converge to outputting zeros.

### ISSUE 2: Masked Loss Function
**Location**: `run_rl.py:12-19`
**Severity**: High

The `masked_mae` function:
- Adds epsilon (1E-8) to mask, causing minimal non-zero gradients even for zero targets
- Uses MAE on sparse one-hot targets, which is suboptimal
- Should use cross-entropy for classification-like outputs

### ISSUE 3: No Softmax on Model Outputs
**Location**: `bot_player.py:26-38`
**Severity**: High

Model heads output raw logits without softmax activation:
```python
plate_head = layers.Dense(n_plates + 1)(x)  # No softmax!
```

**Problem**: Raw logits don't represent probabilities. The argmax works for inference, but training on one-hot targets without proper probability output causes poor learning.

### ISSUE 4: Fixed Exploration Rate
**Location**: `bot_player.py:86`
**Severity**: Medium

```python
if np.random.rand() < 0.1:  # Always 10% random
```

**Problem**: No exploration decay. Models always see 10% random actions, preventing convergence to optimal policy.

### ISSUE 5: No Action Validity During Training
**Location**: `bot_player.py:89-104`
**Severity**: Medium

Invalid actions are penalized at inference (`taboo_penalty = -1000`) but training doesn't account for action validity. Model trains equally on valid and invalid action sequences.

### ISSUE 6: Insufficient Network Capacity
**Location**: `bot_player.py:22-24`
**Severity**: Medium

Only 2 hidden layers (64→32) for 100+ input features:
```python
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
```

**Problem**: Bottleneck may prevent learning complex state-action mappings.

### ISSUE 7: No Reward Normalization
**Location**: `run_rl.py:55-58`
**Severity**: Medium

Rewards not normalized by:
- Running mean/std (baseline subtraction)
- Episode returns
- Advantage estimation

**Impact**: High variance in training signal.

### ISSUE 8: State Input Slicing
**Location**: `bot_player.py:67-69`
**Severity**: Low (recent fix)

```python
start_input_index = 2 * config.n_colors  # Removes bag+graveyard
```

Recent fix removes first 10 elements (bag + graveyard info), which may be intentional but reduces available information.

### ISSUE 9: No Learning Rate Schedule
**Location**: `run_rl.py:83-86`
**Severity**: Low

Fixed learning rate (0.001) throughout training:
```python
model.compile(optimizer=AdamW(learning_rate=learning_rate), ...)
```

### ISSUE 10: Missing Test Suite
**Location**: `test/` (empty)
**Severity**: Low (for training, high for maintenance)

No unit tests to verify game logic or training correctness.

---

## 5. Metrics and Baseline

### Current Training Output
From `res/models/history.png`: Training history shows flat or oscillating loss, indicating no learning progression.

### Expected Behavior
A learning agent should show:
1. Decreasing loss over epochs
2. Increasing win rate vs random player
3. Decreasing invalid action rate

---

## 6. Recommended Fix Priority

| Priority | Issue | Estimated Impact |
|----------|-------|------------------|
| 1 | Reward signal dilution | Critical - blocks all learning |
| 2 | Masked loss function | High - weak gradients |
| 3 | Missing softmax | High - probability mismatch |
| 4 | Fixed exploration | Medium - prevents convergence |
| 5 | Action validity training | Medium - learns bad actions |
| 6 | Network capacity | Medium - limited representation |
| 7 | Reward normalization | Medium - training stability |
| 8 | Learning rate schedule | Low - fine-tuning |
| 9 | Test suite | Low - maintenance |

---

## 7. File Reference Index

| File | Lines | Critical Content |
|------|-------|------------------|
| `run_rl.py` | 12-19 | `masked_mae` loss function |
| `run_rl.py` | 22-67 | `train_on_round()` training loop |
| `run_rl.py` | 70-138 | `train()` main function |
| `bot_player.py` | 16-43 | Model architectures |
| `bot_player.py` | 84-104 | `internal_choice()` action selection |
| `bot_player.py` | 106-130 | `end_of_round()` tile placement |
| `game.py` | 71-126 | `round()` game loop |
| `game.py` | 149-181 | `get_state()` state representation |
| `player.py` | 114-125 | `count_score()` scoring logic |
| `config.py` | All | Configuration constants |
