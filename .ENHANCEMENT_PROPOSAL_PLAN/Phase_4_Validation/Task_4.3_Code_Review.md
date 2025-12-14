# Task 4.3: Code Review and Cleanup

## Objective
Review entire codebase for consistency, remove duplications, ensure quality.

## Scope

### Files to Review
| File | Focus Areas |
|------|-------------|
| `src/config.py` | Parameter organization, naming |
| `src/utils.py` | Utility function completeness |
| `src/obj/container.py` | Base class design |
| `src/obj/bag.py` | Tile management correctness |
| `src/obj/plate.py` | Plate mechanics |
| `src/obj/central.py` | Central plate logic |
| `src/obj/player.py` | Player state management |
| `src/obj/bot_player.py` | Model architecture, decision logic |
| `src/obj/game.py` | Game loop, state representation |
| `src/scripts/run_rl.py` | Training pipeline |

### Review Checklist
- [ ] Type hints on all function signatures
- [ ] Docstrings for public methods
- [ ] No code duplication
- [ ] Consistent naming conventions
- [ ] No dead code
- [ ] No hardcoded magic numbers
- [ ] Proper error handling

## Implementation Steps

### Step 4.3.1: Add missing type hints
**Action**: Review all functions, add type annotations

### Step 4.3.2: Add docstrings
**Action**: Document all public classes and methods

### Step 4.3.3: Remove duplicated code
**Action**: Identify and refactor duplications

### Step 4.3.4: Centralize magic numbers
**Action**: Move hardcoded values to config.py

### Step 4.3.5: Clean up imports
**Action**: Remove unused imports, organize import order

### Step 4.3.6: Run linters
**Action**: Execute pylint, mypy, and fix issues

## Known Issues to Address

### Code Duplication Candidates
1. State slicing logic appears in multiple places
2. Model compilation repeated for both models
3. Action selection logic similar across methods

### Naming Inconsistencies
1. `n_colors` vs `n_tile_per_color` - clarify distinction
2. `internal_choice` vs `end_of_round` - standardize pattern

### Missing Type Hints
1. `game.py:get_state()` return type
2. `bot_player.py:internal_choice()` return type

## Acceptance Criteria
- [ ] All functions have type hints
- [ ] No pylint errors (severity > warning)
- [ ] mypy type checking passes
- [ ] Code coverage doesn't decrease

## Dependencies
- After all implementation phases complete

## Estimated Complexity
Low-Medium - tedious but straightforward
