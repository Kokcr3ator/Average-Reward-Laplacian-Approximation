"""
rlgrid: A research codebase for average-reward reinforcement learning in gridworld environments.

This package provides tools for:
- Parsing 2D grid environments from text files
- Computing transition kernels for deterministic MDPs
- Defining and evaluating policies
- Simulating trajectories in the average-reward setting
"""

from rlgrid.grid_env import GridWorldEnv
from rlgrid.policies import (
    uniform_policy,
    random_policy,
    random_deterministic_policy,
    policy_induced_transition_matrix,
)
from rlgrid.utils import (
    simulate,
    estimate_average_reward,
    visualize_grid,
)

__version__ = "0.1.0"
__all__ = [
    "GridWorldEnv",
    "uniform_policy",
    "random_policy",
    "random_deterministic_policy",
    "policy_induced_transition_matrix",
    "simulate",
    "estimate_average_reward",
    "visualize_grid",
]
