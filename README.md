# Average Reward Laplacian Approximation

A research codebase for testing theoretical bounds in **average-reward reinforcement learning** using deterministic gridworld environments.

## Overview

This project provides tools for:

- **Parsing 2D grid environments** from simple text files
- **Computing transition kernels** for deterministic MDPs with finite state spaces
- **Defining and evaluating policies**: uniform, random stochastic, and random deterministic
- **Simulating trajectories** in the average-reward (continuing) setting
- **Computing exact and estimated average rewards** under various policies

The environment operates in a **continuing task** setting where:
- Entering the goal state yields reward **+1**
- All other transitions yield reward **-1**
- Upon reaching the goal, the agent is **teleported** to a random non-wall, non-goal cell

This design ensures the MDP is ergodic, enabling proper average-reward analysis.

## Project Structure

```
Average-Reward-Laplacian-Approximation/
├── envs/                          # Grid environment definitions
│   ├── simple_5x5.txt             # Simple 5×5 room
│   ├── maze_8x5.txt               # Small maze with internal walls
│   ├── larger_maze.txt            # Larger maze environment
│   └── minimal.txt                # Minimal single-cell environment
├── rlgrid/                        # Main Python package
│   ├── __init__.py                # Package exports
│   ├── grid_env.py                # GridWorldEnv class
│   ├── policies.py                # Policy definitions
│   └── utils.py                   # Simulation and visualization helpers
├── notebooks/
│   └── explore_env.ipynb          # Interactive exploration notebook
├── tests/                         # Unit tests
│   ├── test_grid_env.py           # Environment tests
│   ├── test_policies.py           # Policy tests
│   └── test_utils.py              # Utility function tests
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Installation

### Requirements

- Python 3.9+
- NumPy
- Matplotlib
- Jupyter (for notebooks)

### Install

```bash
# Clone the repository
git clone <repository-url>
cd Average-Reward-Laplacian-Approximation

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Defining Environments

Grid environments are defined in text files using a simple format:

| Character | Meaning |
|-----------|---------|
| `.` | Free cell (traversable) |
| `#` | Wall (non-traversable) |
| `G` | Goal cell (unique, required) |

### Example: Simple 5×5 Room

```
#####
#...#
#.G.#
#...#
#####
```

### Example: Maze with Internal Walls

```
########
#......#
#.##G#.#
#......#
########
```

### Parsing Rules

- All rows must have the same length
- Exactly one `G` (goal) cell is required
- `#` tiles are walls and cannot be entered
- Non-wall cells are mapped to state indices 0, 1, ..., S-1

## Quick Start

### Loading an Environment

```python
from rlgrid import GridWorldEnv

# Load from file
env = GridWorldEnv.from_txt("envs/simple_5x5.txt")

print(f"States: {env.n_states}")
print(f"Actions: {env.n_actions}")
print(f"Goal state: {env.goal_state}")
print(env)  # Display grid
```

### Defining Policies

```python
from rlgrid import uniform_policy, random_policy, random_deterministic_policy
import numpy as np

rng = np.random.default_rng(42)

# Uniform policy: equal probability for all actions
pi_uniform = uniform_policy(env)

# Random stochastic policy: random distribution over actions per state
pi_random = random_policy(env, rng=rng)

# Random deterministic policy: one action per state, chosen randomly
pi_det = random_deterministic_policy(env, rng=rng)
```

### Computing Transition Matrices

```python
from rlgrid import policy_induced_transition_matrix

# Get environment transition kernel P[s, a, s']
P = env.get_transition_kernel()

# Compute policy-induced transition matrix P_π[s, s']
P_pi = policy_induced_transition_matrix(env, pi_uniform)
```

### Simulating Trajectories

```python
from rlgrid import simulate, estimate_average_reward

# Run simulation
states, actions, rewards = simulate(env, pi_uniform, T=10000, rng=rng)

# Estimate average reward
avg_reward = estimate_average_reward(rewards, burn_in=1000)
print(f"Estimated average reward: {avg_reward:.4f}")
```

### Visualizing the Environment

```python
from rlgrid import visualize_grid
import matplotlib.pyplot as plt

ax = visualize_grid(env, show_state_indices=True, title="My GridWorld")
plt.show()
```

## Running the Notebook

The interactive notebook provides comprehensive exploration:

```bash
cd notebooks
jupyter notebook explore_env.ipynb
```

The notebook demonstrates:
1. Loading and visualizing environments
2. Inspecting the transition kernel
3. Defining and comparing policies
4. Running simulations with trajectory visualization
5. Computing and comparing average rewards

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_grid_env.py

# Run with coverage
pytest --cov=rlgrid
```

## API Reference

### GridWorldEnv

```python
class GridWorldEnv:
    """Deterministic gridworld environment for average-reward RL."""
    
    # Class attributes
    ACTIONS = {0: "up", 1: "right", 2: "down", 3: "left"}
    
    # Instance attributes
    grid: List[List[str]]      # 2D character grid
    height: int                 # Grid height
    width: int                  # Grid width
    n_states: int              # Number of non-wall states
    n_actions: int             # Always 4
    goal_state: int            # State index of goal
    goal_pos: Tuple[int, int]  # (row, col) of goal
    P: np.ndarray              # Transition kernel (n_states, n_actions, n_states)
    
    # Methods
    @classmethod
    def from_txt(path: str) -> GridWorldEnv
    def step(s: int, a: int) -> Tuple[int, float]
    def reward(s: int, a: int, s_next: int) -> float
    def get_transition_kernel() -> np.ndarray
    def get_reward_matrix() -> np.ndarray
    def get_expected_reward() -> np.ndarray
```

### Policy Functions

```python
def uniform_policy(env: GridWorldEnv) -> np.ndarray
def random_policy(env: GridWorldEnv, rng=None) -> np.ndarray
def random_deterministic_policy(env: GridWorldEnv, rng=None) -> np.ndarray
def policy_induced_transition_matrix(env: GridWorldEnv, policy: np.ndarray) -> np.ndarray
```

$$\rho(\pi) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} r_t$$

Key properties:
- **No terminal states**: The task continues indefinitely
- **Ergodicity**: Teleportation ensures all states are reachable under any policy
- **Exact computation**: The stationary distribution enables closed-form average reward calculation
