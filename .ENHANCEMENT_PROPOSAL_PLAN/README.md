# AzulRL Enhancement Proposal Plan (EPP)

## Objective
Diagnose and fix the Reinforcement Learning training pipeline for the AzulRL project. The RL model is currently not learning to play Azul effectively.

## Project Overview
AzulRL implements the tabletop game "Azul" with neural network-based bot players trained via reinforcement learning. The system uses two models:
1. **choose_model**: Selects (plate, color, row) during gameplay
2. **end_of_round_model**: Determines tile placement at round end

## EPP Structure

```
.ENHANCEMENT_PROPOSAL_PLAN/
├── README.md                    # This file
├── STATUS.md                    # Summary table of all Phases/Tasks/Steps
├── Phase_0_Diagnosis/           # Initial codebase review and issue identification
├── Phase_1_Core_Fixes/          # Critical reward and loss function fixes
├── Phase_2_Architecture/        # Model architecture improvements
├── Phase_3_Training_Pipeline/   # Training loop and hyperparameter optimization
├── Phase_4_Validation/          # Testing and validation of improvements
├── round_tables/                # Round table discussion logs
└── questions/                   # User input requests
```

## Timeline
- **Created**: 2025-12-08
- **Last Updated**: 2025-12-08

## Key Documents
| Document | Purpose |
|----------|---------|
| `STATUS.md` | Master tracking table for all tasks |
| `Phase_0_Diagnosis/observations.md` | Initial codebase analysis |
| `round_tables/RT_001_kickoff.md` | Initial planning discussion |
