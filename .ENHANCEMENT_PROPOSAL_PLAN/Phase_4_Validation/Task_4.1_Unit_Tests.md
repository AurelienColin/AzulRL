# Task 4.1: Create Unit Test Suite

## Objective
Establish test coverage for game logic and training components.

## Problem Statement
- `test/` directory is empty
- No automated testing
- Bug risks high during refactoring

## Proposed Solution
Create comprehensive pytest test suite:

```
test/
├── conftest.py              # Shared fixtures
├── test_game.py             # Game logic tests
├── test_player.py           # Player mechanics tests
├── test_bot_player.py       # Bot decision tests
├── test_training.py         # Training pipeline tests
└── test_state.py            # State representation tests
```

## Implementation Steps

### Step 4.1.1: Set up pytest configuration
**File**: `pytest.ini` or `pyproject.toml`
**Action**: Configure pytest with appropriate settings

### Step 4.1.2: Create shared fixtures
**File**: `test/conftest.py`
**Action**: Define reusable game, player, model fixtures

### Step 4.1.3: Test game initialization
**File**: `test/test_game.py`
**Action**: Test game setup, plate distribution, state creation

### Step 4.1.4: Test player mechanics
**File**: `test/test_player.py`
**Action**: Test tile placement, scoring, penalty calculation

### Step 4.1.5: Test bot decision logic
**File**: `test/test_bot_player.py`
**Action**: Test action selection, action masking, exploration

### Step 4.1.6: Test training loop
**File**: `test/test_training.py`
**Action**: Test reward calculation, loss function, batch training

### Step 4.1.7: Test state representation
**File**: `test/test_state.py`
**Action**: Test state vector construction, normalization

### Step 4.1.8: Add CI integration
**File**: `.gitlab-ci.yml` or GitHub Actions
**Action**: Run tests on commit

## Acceptance Criteria
- [x] All core game logic has test coverage
- [x] Training pipeline components are tested
- [x] Tests pass consistently (153 tests passing)
- [x] Coverage > 70% (CI configured with coverage reporting)

## Dependencies
- All implementation phases should be complete

## Estimated Complexity
Medium - many tests but straightforward patterns

## Test Cases to Include

### Game Logic
- `test_bag_initialization`: Bag has correct tile counts
- `test_plate_distribution`: Plates receive 4 tiles each
- `test_tile_selection`: Selecting tiles moves them correctly
- `test_scoring`: Correct points for placements
- `test_penalty_calculation`: Penalties applied correctly

### Training
- `test_reward_calculation`: Rewards have correct magnitude
- `test_loss_function`: Loss produces non-zero gradients
- `test_action_masking`: Invalid actions get zero probability
- `test_exploration_decay`: Epsilon decreases correctly
