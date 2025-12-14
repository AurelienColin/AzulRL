# AzulRL Enhancement Proposal Plan - Status Summary

**Last Updated**: 2025-12-14
**Project Status**: Phase 3 Complete - Ready for CR-003 and RT-004

---

## Executive Summary

The AzulRL project has been diagnosed with critical issues preventing RL learning:
1. **Reward signal dilution** - rewards divided by turns produce near-zero gradients
2. **Loss function masking issues** - MAE on sparse targets ineffective
3. **Missing softmax outputs** - raw logits not probabilities
4. **Fixed exploration rate** - prevents convergence

This plan addresses these issues across 5 phases with 16 tasks and 50+ steps.

---

## Summary Table

| INDEX | DESCRIPTION | AGENT | DOCUMENTATION REF | CODEBASE REF | STATUS |
|-------|-------------|-------|-------------------|--------------|--------|
| **Phase 0** | **Diagnosis** | - | Phase_0_Diagnosis/ | - | **[x]** |
| 0.1 | Initial codebase exploration | Explore | observations.md | - | [x] |
| 0.2 | Issue identification | archivist | observations.md:L50-L140 | - | [x] |
| 0.3 | Priority ranking | agent-organizer | observations.md:L150-L165 | - | [x] |
| **RT-001** | **Kickoff Round Table** | agent-organizer | round_tables/RT_001_kickoff.md | - | [x] |
| **Phase 1** | **Core Fixes** | - | Phase_1_Core_Fixes/ | - | **[x]** |
| 1.1 | Fix Reward Function | machine-learning-researcher | Task_1.1_Reward_Function.md | run_rl.py:22-56 | [x] |
| 1.1.1 | Define discount factor in config | python-pro | Task_1.1_Reward_Function.md:L25-L27 | config.py:13 | [x] |
| 1.1.2 | Refactor reward calculation | machine-learning-researcher | Task_1.1_Reward_Function.md:L29-L31 | run_rl.py:93-100 | [x] |
| 1.1.3 | Add advantage normalization | machine-learning-researcher | Task_1.1_Reward_Function.md:L33-L35 | run_rl.py:50-54 | [x] |
| 1.1.4 | Test reward magnitude | test-automator | Task_1.1_Reward_Function.md:L37-L38 | - | [x] |
| 1.2 | Fix Loss Function | machine-learning-researcher | Task_1.2_Loss_Function.md | run_rl.py:15-48 | [x] |
| 1.2.1 | Remove masked_mae function | python-pro | Task_1.2_Loss_Function.md:L28-L30 | run_rl.py | [x] |
| 1.2.2 | Implement policy gradient loss | machine-learning-researcher | Task_1.2_Loss_Function.md:L32-L34 | run_rl.py:15-48 | [x] |
| 1.2.3 | Update model compilation | python-pro | Task_1.2_Loss_Function.md:L36-L38 | run_rl.py:163-164 | [x] |
| 1.2.4 | Modify training data format | python-pro | Task_1.2_Loss_Function.md:L40-L42 | run_rl.py | [x] |
| 1.3 | Add Softmax to Model Outputs | machine-learning-researcher | Task_1.3_Model_Outputs.md | bot_player.py:38-40 | [x] |
| 1.3.1 | Add softmax to choose_model | python-pro | Task_1.3_Model_Outputs.md:L25-L27 | bot_player.py:38-40 | [x] |
| 1.3.2 | Add softmax to end_of_round_model | python-pro | Task_1.3_Model_Outputs.md:L29-L31 | bot_player.py:57 | [x] |
| 1.3.3 | Remove inference-time softmax | python-pro | Task_1.3_Model_Outputs.md:L33-L35 | bot_player.py | [x] |
| 1.3.4 | Verify probability sums | test-automator | Task_1.3_Model_Outputs.md:L37-L38 | - | [x] |
| 1.4 | Implement Exploration Decay | python-pro | Task_1.4_Exploration_Schedule.md | bot_player.py:72-84 | [x] |
| 1.4.1 | Add exploration parameters to config | python-pro | Task_1.4_Exploration_Schedule.md:L28-L30 | config.py:14-16 | [x] |
| 1.4.2 | Add epsilon attribute to BotPlayer | python-pro | Task_1.4_Exploration_Schedule.md:L32-L34 | bot_player.py:72 | [x] |
| 1.4.3 | Implement decay method | python-pro | Task_1.4_Exploration_Schedule.md:L36-L38 | bot_player.py:82-84 | [x] |
| 1.4.4 | Modify internal_choice | python-pro | Task_1.4_Exploration_Schedule.md:L40-L42 | bot_player.py:95 | [x] |
| 1.4.5 | Call decay in training loop | python-pro | Task_1.4_Exploration_Schedule.md:L44-L46 | run_rl.py:178 | [x] |
| 1.4.6 | Log exploration rate | python-pro | Task_1.4_Exploration_Schedule.md:L48-L50 | run_rl.py:206 | [x] |
| **CR-001** | **Code Review - Phase 1** | code-reviewer | - | src/ | [x] |
| **RT-002** | **Phase 1 Completion Round Table** | agent-organizer | round_tables/RT_002_phase1_complete.md | - | [x] |
| **Phase 2** | **Architecture Improvements** | - | Phase_2_Architecture/ | - | **[x]** |
| 2.1 | Increase Network Capacity | machine-learning-researcher | Task_2.1_Network_Capacity.md | bot_player.py:16-84 | [x] |
| 2.1.1 | Define architecture hyperparameters | python-pro | Task_2.1_Network_Capacity.md:L30-L32 | config.py:18-20 | [x] |
| 2.1.2 | Refactor choose_model architecture | machine-learning-researcher | Task_2.1_Network_Capacity.md:L34-L36 | bot_player.py:36-64 | [x] |
| 2.1.3 | Refactor end_of_round_model | machine-learning-researcher | Task_2.1_Network_Capacity.md:L38-L40 | bot_player.py:67-84 | [x] |
| 2.1.4 | Add dropout regularization | machine-learning-researcher | Task_2.1_Network_Capacity.md:L42-L44 | bot_player.py:29-30 | [x] |
| 2.1.5 | Add batch normalization | machine-learning-researcher | Task_2.1_Network_Capacity.md:L46-L48 | - | [x] (skipped - interferes with RL) |
| 2.1.6 | Test parameter count | test-automator | Task_2.1_Network_Capacity.md:L50-L51 | - | [x] (~97k params) |
| 2.2 | Implement Action Masking | machine-learning-researcher | Task_2.2_Action_Masking.md | bot_player.py:20-204 | [x] |
| 2.2.1 | Create action mask generator | python-pro | Task_2.2_Action_Masking.md:L32-L34 | bot_player.py:20-157 | [x] |
| 2.2.2 | Modify choose_model for mask | machine-learning-researcher | Task_2.2_Action_Masking.md:L36-L38 | bot_player.py:330-406 | [x] |
| 2.2.3 | Apply mask before softmax | machine-learning-researcher | Task_2.2_Action_Masking.md:L40-L42 | bot_player.py:160-171 | [x] |
| 2.2.4 | Handle row validity | python-pro | Task_2.2_Action_Masking.md:L44-L46 | bot_player.py:49-105 | [x] |
| 2.2.5 | Update end_of_round masking | python-pro | Task_2.2_Action_Masking.md:L48-L50 | bot_player.py:408-480 | [x] |
| 2.2.6 | Remove taboo_penalty logic | python-pro | Task_2.2_Action_Masking.md:L52-L54 | bot_player.py | [x] |
| 2.3 | Improve State Representation | python-pro | Task_2.3_State_Representation.md | utils.py, bot_player.py | [x] |
| 2.3.1 | Add state normalization function | python-pro | Task_2.3_State_Representation.md:L33-L35 | utils.py:normalize_state | [x] |
| 2.3.2 | Normalize tile counts | python-pro | Task_2.3_State_Representation.md:L37-L38 | utils.py | [x] |
| 2.3.3 | Normalize penalty values | python-pro | Task_2.3_State_Representation.md:L40-L42 | utils.py | [x] |
| 2.3.4 | Add strategic features | machine-learning-researcher | Task_2.3_State_Representation.md:L44-L46 | - | [x] (deferred) |
| 2.3.5 | Update input_length calculation | python-pro | Task_2.3_State_Representation.md:L48-L50 | bot_player.py | [x] |
| 2.3.6 | Re-evaluate state slicing | machine-learning-researcher | Task_2.3_State_Representation.md:L52-L54 | config.py:start_input_index=0 | [x] |
| **CR-002** | **Code Review - Phase 2** | code-reviewer | - | src/ | [x] |
| **RT-003** | **Phase 2 Completion Round Table** | agent-organizer | round_tables/RT_003_phase2_complete.md | - | [x] |
| **Phase 3** | **Training Pipeline** | - | Phase_3_Training_Pipeline/ | - | **[x]** |
| 3.1 | Implement Experience Replay | machine-learning-researcher | Task_3.1_Experience_Replay.md | run_rl.py | [x] |
| 3.1.1 | Create ReplayBuffer class | python-pro | Task_3.1_Experience_Replay.md:L30-L32 | replay_buffer.py (new) | [x] |
| 3.1.2 | Add buffer to training script | python-pro | Task_3.1_Experience_Replay.md:L34-L36 | run_rl.py:200-204 | [x] |
| 3.1.3 | Store experiences in buffer | python-pro | Task_3.1_Experience_Replay.md:L38-L40 | run_rl.py:242-253 | [x] |
| 3.1.4 | Sample batches for training | machine-learning-researcher | Task_3.1_Experience_Replay.md:L42-L44 | run_rl.py:152-170 | [x] |
| 3.1.5 | Add minimum buffer size check | python-pro | Task_3.1_Experience_Replay.md:L46-L48 | run_rl.py:257-266 | [x] |
| 3.1.6 | Configure buffer parameters | python-pro | Task_3.1_Experience_Replay.md:L50-L52 | config.py:18-22 | [x] |
| 3.2 | Implement Learning Rate Schedule | machine-learning-researcher | Task_3.2_Learning_Rate_Schedule.md | run_rl.py:269-285 | [x] |
| 3.2.1 | Define LR schedule parameters | python-pro | Task_3.2_Learning_Rate_Schedule.md:L28-L30 | config.py:24-28 | [x] |
| 3.2.2 | Implement cosine decay schedule | machine-learning-researcher | Task_3.2_Learning_Rate_Schedule.md:L32-L34 | run_rl.py:16-78 | [x] |
| 3.2.3 | Apply schedule to optimizer | python-pro | Task_3.2_Learning_Rate_Schedule.md:L36-L38 | run_rl.py:280-285 | [x] |
| 3.2.4 | Log learning rate over training | python-pro | Task_3.2_Learning_Rate_Schedule.md:L40-L42 | run_rl.py:362-364,398 | [x] |
| 3.2.5 | Add warmup period (optional) | machine-learning-researcher | Task_3.2_Learning_Rate_Schedule.md:L44-L46 | run_rl.py:16-78 | [x] |
| 3.3 | Comprehensive Training Metrics | python-pro | Task_3.3_Training_Metrics.md | run_rl.py:18-114 | [x] |
| 3.3.1 | Create metrics dictionary | python-pro | Task_3.3_Training_Metrics.md:L30-L32 | run_rl.py:18-114 | [x] |
| 3.3.2 | Track loss per model | python-pro | Task_3.3_Training_Metrics.md:L34-L36 | run_rl.py:523-526 | [x] |
| 3.3.3 | Implement win rate evaluation | python-pro | Task_3.3_Training_Metrics.md:L38-L40 | run_rl.py:346-396 | [x] |
| 3.3.4 | Track invalid action attempts | python-pro | Task_3.3_Training_Metrics.md:L42-L44 | bot_player.py:295-326 | [x] |
| 3.3.5 | Log exploration and learning rate | python-pro | Task_3.3_Training_Metrics.md:L46-L48 | run_rl.py:529-531 | [x] |
| 3.3.6 | Create visualization dashboard | python-pro | Task_3.3_Training_Metrics.md:L50-L52 | run_rl.py:556-608 | [x] |
| 3.3.7 | Save metrics to JSON | python-pro | Task_3.3_Training_Metrics.md:L54-L56 | run_rl.py:87-98,545,551 | [x] |
| **CR-003** | **Code Review - Phase 3** | code-reviewer | - | src/scripts/ | [ ] |
| **RT-004** | **Phase 3 Completion Round Table** | agent-organizer | round_tables/RT_004_phase3.md | - | [ ] |
| **Phase 4** | **Validation** | - | Phase_4_Validation/ | - | **[ ]** |
| 4.1 | Create Unit Test Suite | test-automator | Task_4.1_Unit_Tests.md | test/ | [ ] |
| 4.1.1 | Set up pytest configuration | devops-engineer | Task_4.1_Unit_Tests.md:L25-L27 | pytest.ini | [ ] |
| 4.1.2 | Create shared fixtures | test-automator | Task_4.1_Unit_Tests.md:L29-L31 | test/conftest.py | [ ] |
| 4.1.3 | Test game initialization | test-automator | Task_4.1_Unit_Tests.md:L33-L35 | test/test_game.py | [ ] |
| 4.1.4 | Test player mechanics | test-automator | Task_4.1_Unit_Tests.md:L37-L39 | test/test_player.py | [ ] |
| 4.1.5 | Test bot decision logic | test-automator | Task_4.1_Unit_Tests.md:L41-L43 | test/test_bot_player.py | [ ] |
| 4.1.6 | Test training loop | test-automator | Task_4.1_Unit_Tests.md:L45-L47 | test/test_training.py | [ ] |
| 4.1.7 | Test state representation | test-automator | Task_4.1_Unit_Tests.md:L49-L51 | test/test_state.py | [ ] |
| 4.1.8 | Add CI integration | devops-engineer | Task_4.1_Unit_Tests.md:L53-L55 | .gitlab-ci.yml | [ ] |
| 4.2 | Baseline Agent Comparison | machine-learning-researcher | Task_4.2_Baseline_Comparison.md | - | [ ] |
| 4.2.1 | Create RandomAgent class | python-pro | Task_4.2_Baseline_Comparison.md:L28-L30 | random_agent.py (new) | [ ] |
| 4.2.2 | Create GreedyAgent class | python-pro | Task_4.2_Baseline_Comparison.md:L32-L34 | greedy_agent.py (new) | [ ] |
| 4.2.3 | Create evaluation harness | python-pro | Task_4.2_Baseline_Comparison.md:L36-L38 | evaluate.py (new) | [ ] |
| 4.2.4 | Add evaluation to training | python-pro | Task_4.2_Baseline_Comparison.md:L40-L42 | run_rl.py | [ ] |
| 4.2.5 | Track win rate over training | python-pro | Task_4.2_Baseline_Comparison.md:L44-L46 | run_rl.py | [ ] |
| 4.2.6 | Create performance report | technical-writer | Task_4.2_Baseline_Comparison.md:L48-L50 | - | [ ] |
| 4.3 | Code Review and Cleanup | code-reviewer | Task_4.3_Code_Review.md | src/ | [ ] |
| 4.3.1 | Add missing type hints | python-pro | Task_4.3_Code_Review.md:L25-L26 | src/ | [ ] |
| 4.3.2 | Add docstrings | technical-writer | Task_4.3_Code_Review.md:L28-L29 | src/ | [ ] |
| 4.3.3 | Remove duplicated code | refactoring-specialist | Task_4.3_Code_Review.md:L31-L32 | src/ | [ ] |
| 4.3.4 | Centralize magic numbers | python-pro | Task_4.3_Code_Review.md:L34-L35 | config.py | [ ] |
| 4.3.5 | Clean up imports | python-pro | Task_4.3_Code_Review.md:L37-L38 | src/ | [ ] |
| 4.3.6 | Run linters | devops-engineer | Task_4.3_Code_Review.md:L40-L41 | - | [ ] |
| 4.4 | Documentation Review | technical-writer | Task_4.4_Documentation_Review.md | - | [ ] |
| 4.4.1 | Create project README.md | technical-writer | Task_4.4_Documentation_Review.md:L23-L30 | README.md | [ ] |
| 4.4.2 | Create CLAUDE.md | technical-writer | Task_4.4_Documentation_Review.md:L32-L38 | CLAUDE.md | [ ] |
| 4.4.3 | Verify EPP accuracy | archivist | Task_4.4_Documentation_Review.md:L40-L41 | .ENHANCEMENT_PROPOSAL_PLAN/ | [ ] |
| 4.4.4 | Update API documentation | technical-writer | Task_4.4_Documentation_Review.md:L43-L44 | src/ | [ ] |
| 4.4.5 | Add inline code comments | technical-writer | Task_4.4_Documentation_Review.md:L46-L47 | src/ | [ ] |
| 4.4.6 | Create architecture diagram | technical-writer | Task_4.4_Documentation_Review.md:L49-L50 | docs/architecture.md | [ ] |
| **DR-001** | **Documentation Review - Final** | technical-writer | - | .ENHANCEMENT_PROPOSAL_PLAN/ | [ ] |
| **RT-005** | **Final Round Table - Project Completion** | agent-organizer | round_tables/RT_005_final.md | - | [ ] |

---

## Progress Summary

| Phase | Total Tasks | Completed | In Progress | Pending |
|-------|-------------|-----------|-------------|---------|
| Phase 0: Diagnosis | 3 | 3 | 0 | 0 |
| Phase 1: Core Fixes | 4 (+16 steps) | 20 | 0 | 0 |
| Phase 2: Architecture | 3 (+18 steps) | 21 | 0 | 0 |
| Phase 3: Training Pipeline | 3 (+17 steps) | 20 | 0 | 0 |
| Phase 4: Validation | 4 (+24 steps) | 0 | 0 | 28 |
| Round Tables | 5 | 3 | 0 | 2 |
| Code Reviews | 3 | 2 | 0 | 1 |
| Documentation Reviews | 1 | 0 | 0 | 1 |
| **TOTAL** | **~100** | **69** | **0** | **~31** |

---

## Legend

- **STATUS**: `[x]` = Complete, `[ ]` = Pending, `[~]` = In Progress
- **AGENT**: Primary responsible agent for the task
- **RT-XXX**: Round Table Discussion
- **CR-XXX**: Code Review
- **DR-XXX**: Documentation Review

---

## Quick Links

| Document | Path |
|----------|------|
| README | `.ENHANCEMENT_PROPOSAL_PLAN/README.md` |
| Observations | `.ENHANCEMENT_PROPOSAL_PLAN/Phase_0_Diagnosis/observations.md` |
| Phase 1 Tasks | `.ENHANCEMENT_PROPOSAL_PLAN/Phase_1_Core_Fixes/` |
| Phase 2 Tasks | `.ENHANCEMENT_PROPOSAL_PLAN/Phase_2_Architecture/` |
| Phase 3 Tasks | `.ENHANCEMENT_PROPOSAL_PLAN/Phase_3_Training_Pipeline/` |
| Phase 4 Tasks | `.ENHANCEMENT_PROPOSAL_PLAN/Phase_4_Validation/` |
| Round Tables | `.ENHANCEMENT_PROPOSAL_PLAN/round_tables/` |
