"""Reinforcement learning package for search zone generation and evaluation.

This package implements a reinforcement learning system for generating and
evaluating search zones in missing-person cases. It combines movement models,
risk maps, and reward functions to optimize search zone placement across
multiple time windows.

Key components:
- movement_model: Probabilistic movement model with Markov chain propagation
- zone_rl: Zone selection and scoring workflows
- rewards: Hierarchical reward computation (zone, window, episode levels)
- rl_env: RL environment for zone evaluation
- build_rl_zones: Script for processing cases and generating zones
"""


