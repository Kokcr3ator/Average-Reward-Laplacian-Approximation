"""
Utility Module

This module provides helper functions for simulation, visualization,
and analysis of GridWorld environments for average-reward reinforcement learning.

Key functions:
    - simulate: Run a trajectory under a given policy
    - estimate_average_reward: Estimate the long-run average reward
    - visualize_grid: Create a matplotlib visualization of the grid
    - print_transition_info: Display transition information for debugging
"""

from __future__ import annotations

from typing import Tuple, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rlgrid.grid_env import GridWorldEnv


def simulate(
    env: "GridWorldEnv",
    policy: np.ndarray,
    T: int,
    start_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a trajectory in the environment under a given policy.
    
    Starting from an initial state, samples actions according to the policy
    and executes them in the environment for T steps.
    
    Args:
        env: The GridWorld environment.
        policy: Policy array of shape (n_states, n_actions).
        T: Number of timesteps to simulate.
        start_state: Initial state. If None, samples uniformly from non-goal states.
        rng: Random number generator. If None, uses numpy's default.
    
    Returns:
        Tuple of (states, actions, rewards):
            - states: Array of shape (T+1,) with state trajectory (includes start).
            - actions: Array of shape (T,) with actions taken.
            - rewards: Array of shape (T,) with rewards received.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> states, actions, rewards = simulate(env, pi, T=1000)
        >>> print(f"Average reward: {rewards.mean():.4f}")
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if policy.shape != (env.n_states, env.n_actions):
        raise ValueError(
            f"Policy shape {policy.shape} doesn't match expected "
            f"({env.n_states}, {env.n_actions})"
        )
    
    # Initialize arrays
    states = np.zeros(T + 1, dtype=np.int64)
    actions = np.zeros(T, dtype=np.int64)
    rewards = np.zeros(T, dtype=np.float64)
    
    # Set initial state
    if start_state is None:
        if len(env._non_goal_states) > 0:
            states[0] = rng.choice(env._non_goal_states)
        else:
            states[0] = 0
    else:
        if start_state < 0 or start_state >= env.n_states:
            raise ValueError(f"Invalid start_state {start_state}")
        states[0] = start_state
    
    # Simulate trajectory
    for t in range(T):
        s = states[t]
        
        # Sample action from policy
        a = rng.choice(env.n_actions, p=policy[s])
        actions[t] = a
        
        # Take step in environment
        s_next, r = env.step(s, a)
        states[t + 1] = s_next
        rewards[t] = r
    
    return states, actions, rewards


def estimate_average_reward(rewards: np.ndarray, burn_in: int = 0) -> float:
    """
    Estimate the average reward from a trajectory of rewards.
    
    Computes the sample mean of rewards, optionally discarding initial
    burn-in samples to reduce bias from initial transient.
    
    Args:
        rewards: Array of rewards from a trajectory.
        burn_in: Number of initial samples to discard. Default 0.
    
    Returns:
        Estimated average reward.
    
    Example:
        >>> rewards = np.array([-1, -1, 1, -1, -1, -1, 1, -1])
        >>> avg = estimate_average_reward(rewards)
        >>> print(f"Average reward: {avg:.4f}")
    """
    if burn_in >= len(rewards):
        raise ValueError(
            f"burn_in ({burn_in}) must be less than length of rewards ({len(rewards)})"
        )
    
    return float(np.mean(rewards[burn_in:]))


def estimate_average_reward_running(rewards: np.ndarray) -> np.ndarray:
    """
    Compute running average reward over a trajectory.
    
    Returns an array where element t is the average reward up to time t.
    Useful for visualizing convergence.
    
    Args:
        rewards: Array of rewards from a trajectory.
    
    Returns:
        Array of shape (len(rewards),) with running averages.
    
    Example:
        >>> rewards = np.array([-1, -1, 1, -1, -1, 1])
        >>> running_avg = estimate_average_reward_running(rewards)
    """
    return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)


def visualize_grid(
    env: "GridWorldEnv",
    ax=None,
    show_state_indices: bool = True,
    title: Optional[str] = None,
    current_state: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 8)
):
    """
    Visualize the grid environment using matplotlib.
    
    Creates a grid visualization where:
        - Walls are black
        - Free cells are white
        - Goal cell is green
        - Optionally shows state indices on each cell
        - Optionally highlights current state
    
    Args:
        env: The GridWorld environment.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        show_state_indices: Whether to annotate state indices on cells.
        title: Title for the plot.
        current_state: State to highlight with a red marker.
        figsize: Figure size if creating new figure.
    
    Returns:
        Matplotlib axes object.
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> ax = visualize_grid(env, title="My GridWorld")
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create numeric grid: 0=free, 1=wall, 2=goal
    grid_numeric = np.zeros((env.height, env.width), dtype=np.int32)
    for row in range(env.height):
        for col in range(env.width):
            char = env.grid[row][col]
            if char == '#':
                grid_numeric[row, col] = 1
            elif char == 'G':
                grid_numeric[row, col] = 2
            else:
                grid_numeric[row, col] = 0
    
    # Colors: white (free), black (wall), green (goal)
    cmap = ListedColormap(['white', 'black', 'limegreen'])
    
    # Plot grid
    ax.imshow(grid_numeric, cmap=cmap, origin='upper', aspect='equal')
    
    # Draw grid lines
    for x in range(env.width + 1):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5)
    for y in range(env.height + 1):
        ax.axhline(y - 0.5, color='gray', linewidth=0.5)
    
    # Annotate state indices
    if show_state_indices:
        for state, (row, col) in env.state_to_pos.items():
            ax.text(
                col, row, str(state),
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='black' if env.grid[row][col] != 'G' else 'white'
            )
    
    # Highlight current state if provided
    if current_state is not None and current_state in env.state_to_pos:
        row, col = env.state_to_pos[current_state]
        ax.plot(col, row, 'ro', markersize=20, alpha=0.6)
    
    # Labels and title
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    if title:
        ax.set_title(title)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='gray', label='Free cell'),
        mpatches.Patch(facecolor='black', edgecolor='gray', label='Wall'),
        mpatches.Patch(facecolor='limegreen', edgecolor='gray', label='Goal'),
    ]
    if current_state is not None:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, label='Current state', alpha=0.6)
        )
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    return ax


def visualize_trajectory(
    env: "GridWorldEnv",
    states: np.ndarray,
    max_steps: int = 50,
    figsize: Tuple[float, float] = (10, 8)
):
    """
    Visualize a trajectory on the grid.
    
    Shows the grid with arrows indicating the path taken.
    
    Args:
        env: The GridWorld environment.
        states: Array of state indices representing the trajectory.
        max_steps: Maximum number of steps to visualize.
        figsize: Figure size.
    
    Returns:
        Matplotlib axes object.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    visualize_grid(env, ax=ax, show_state_indices=True)
    
    # Plot trajectory
    n_steps = min(len(states) - 1, max_steps)
    
    for t in range(n_steps):
        s = states[t]
        s_next = states[t + 1]
        
        row1, col1 = env.state_to_pos[s]
        row2, col2 = env.state_to_pos[s_next]
        
        # Small offset to see overlapping arrows
        offset = 0.1 * (t / max(n_steps, 1))
        
        # Draw arrow
        ax.annotate(
            '', xy=(col2 + offset, row2 + offset),
            xytext=(col1 + offset, row1 + offset),
            arrowprops=dict(
                arrowstyle='->', color='blue',
                alpha=0.5, lw=1.5
            )
        )
    
    # Mark start and end
    if len(states) > 0:
        start_row, start_col = env.state_to_pos[states[0]]
        ax.plot(start_col, start_row, 'gs', markersize=15, alpha=0.7, label='Start')
        
        end_idx = min(len(states) - 1, max_steps)
        end_row, end_col = env.state_to_pos[states[end_idx]]
        ax.plot(end_col, end_row, 'rs', markersize=15, alpha=0.7, label='End')
    
    ax.set_title(f'Trajectory (first {n_steps} steps)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.5))
    
    return ax


def print_transition_info(env: "GridWorldEnv", state: int) -> None:
    """
    Print transition information for a given state.
    
    Shows the next state for each action and whether it hits a wall.
    
    Args:
        env: The GridWorld environment.
        state: State index to inspect.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> print_transition_info(env, 0)
    """
    print(f"Transitions from state {state} at position {env.state_to_pos[state]}:")
    print("-" * 50)
    
    for a in range(env.n_actions):
        action_name = env.ACTIONS[a]
        next_state = int(np.argmax(env.P[state, a]))
        next_pos = env.state_to_pos[next_state]
        
        stays = (next_state == state)
        is_goal = (next_state == env.goal_state)
        
        status = ""
        if stays:
            status = "(stays in place - wall/boundary)"
        if is_goal:
            status = "(GOAL! +1 reward, teleport)"
        
        print(f"  {action_name:6s} -> state {next_state} at {next_pos} {status}")


def print_policy_info(env: "GridWorldEnv", policy: np.ndarray) -> None:
    """
    Print policy information for each state.
    
    Args:
        env: The GridWorld environment.
        policy: Policy array of shape (n_states, n_actions).
    """
    print("Policy:")
    print("-" * 70)
    print(f"{'State':>6} {'Position':>10} | {'Up':>8} {'Right':>8} {'Down':>8} {'Left':>8}")
    print("-" * 70)
    
    for s in range(env.n_states):
        pos = env.state_to_pos[s]
        probs = policy[s]
        
        marker = " *" if s == env.goal_state else ""
        print(
            f"{s:>6} {str(pos):>10}{marker} | "
            f"{probs[0]:>8.3f} {probs[1]:>8.3f} {probs[2]:>8.3f} {probs[3]:>8.3f}"
        )
    
    print("-" * 70)
    print("* = goal state")


def compute_stationary_distribution(
    P_pi: np.ndarray,
    max_iter: int = 10000,
    tol: float = 1e-10
) -> np.ndarray:
    """
    Compute the stationary distribution of a Markov chain.
    
    Uses eigenvalue decomposition to find the stationary distribution.
    Falls back to power iteration if needed.
    
    Args:
        P_pi: Transition matrix of shape (n_states, n_states).
        max_iter: Maximum number of iterations for power method fallback.
        tol: Convergence tolerance.
    
    Returns:
        Stationary distribution array of shape (n_states,).
    
    Raises:
        ValueError: If the algorithm doesn't converge.
    
    Example:
        >>> P = policy_induced_transition_matrix(env, policy)
        >>> mu = compute_stationary_distribution(P)
        >>> assert np.allclose(mu @ P, mu)
    """
    n_states = P_pi.shape[0]
    
    # Try eigenvalue method first (works for all ergodic chains)
    try:
        # Left eigenvectors of P correspond to right eigenvectors of P.T
        eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        # (there should be exactly one for an ergodic chain)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        if np.abs(eigenvalues[idx] - 1.0) < 1e-8:
            mu = np.real(eigenvectors[:, idx])
            # Normalize to get probability distribution
            mu = np.abs(mu)  # Ensure non-negative
            mu = mu / mu.sum()
            
            # Verify it's actually stationary
            if np.allclose(mu @ P_pi, mu, atol=1e-6):
                return mu
    except np.linalg.LinAlgError:
        pass
    
    # Fallback: power iteration with averaging (handles periodic chains)
    mu = np.ones(n_states) / n_states
    mu_avg = mu.copy()
    
    for i in range(1, max_iter + 1):
        mu_new = mu @ P_pi
        
        # Running average to handle periodicity
        mu_avg = (mu_avg * i + mu_new) / (i + 1)
        
        if i > 100 and np.max(np.abs(mu_avg @ P_pi - mu_avg)) < tol:
            return mu_avg / mu_avg.sum()
        
        mu = mu_new
    
    # If we haven't converged, return the running average anyway
    # (for periodic chains, the average is still valid)
    mu_avg = mu_avg / mu_avg.sum()
    if np.allclose(mu_avg @ P_pi, mu_avg, atol=1e-4):
        return mu_avg
    
    raise ValueError(f"Stationary distribution did not converge in {max_iter} iterations")


def compute_average_reward_exact(
    env: "GridWorldEnv",
    policy: np.ndarray
) -> float:
    """
    Compute the exact average reward under a policy.
    
    Uses the stationary distribution to compute:
        ρ(π) = Σ_s μ(s) * Σ_a π(s,a) * Σ_{s'} P(s'|s,a) * R(s,a,s')
    
    For an ergodic MDP, this is the long-run average reward.
    
    Args:
        env: The GridWorld environment.
        policy: Policy array of shape (n_states, n_actions).
    
    Returns:
        Exact average reward under the policy.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> avg_reward = compute_average_reward_exact(env, pi)
    """
    from rlgrid.policies import policy_induced_transition_matrix, policy_expected_reward
    
    # Get policy-induced transition matrix
    P_pi = policy_induced_transition_matrix(env, policy)
    
    # Compute stationary distribution
    mu = compute_stationary_distribution(P_pi)
    
    # Get expected reward per state under policy
    r_pi = policy_expected_reward(env, policy)
    
    # Compute average reward
    return float(np.dot(mu, r_pi))
