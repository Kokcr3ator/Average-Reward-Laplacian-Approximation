"""
Policy Module

This module provides functions for defining and working with policies
in the GridWorld environment for average-reward reinforcement learning.

A policy π is represented as a numpy array of shape (n_states, n_actions)
where π[s, a] is the probability of taking action a in state s.
Each row sums to 1.

Available policy constructors:
    - uniform_policy: Equal probability for all actions in each state
    - random_policy: Random stochastic policy (via Dirichlet distribution)
    - random_deterministic_policy: Random but deterministic (one action per state)

Utility functions:
    - policy_induced_transition_matrix: Compute P_π from P and π
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rlgrid.grid_env import GridWorldEnv


def uniform_policy(env: "GridWorldEnv") -> np.ndarray:
    """
    Create a uniform policy that assigns equal probability to all actions.
    
    For each state s, the policy assigns probability 1/n_actions to each action.
    
    Args:
        env: The GridWorld environment.
    
    Returns:
        Policy array of shape (n_states, n_actions) where each row sums to 1
        and all entries are 1/n_actions.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> print(pi[0])  # [0.25, 0.25, 0.25, 0.25]
    """
    policy = np.ones((env.n_states, env.n_actions), dtype=np.float64)
    policy /= env.n_actions
    return policy


def random_policy(
    env: "GridWorldEnv",
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Create a random stochastic policy.
    
    For each state s, samples a random probability distribution over actions
    using a symmetric Dirichlet distribution with concentration parameter 1.
    This produces a uniformly random point on the probability simplex.
    
    Args:
        env: The GridWorld environment.
        rng: Random number generator. If None, uses numpy's default.
    
    Returns:
        Policy array of shape (n_states, n_actions) where each row sums to 1
        and represents a random probability distribution over actions.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> rng = np.random.default_rng(42)
        >>> pi = random_policy(env, rng=rng)
        >>> assert np.allclose(pi.sum(axis=1), 1.0)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample from symmetric Dirichlet(1, 1, ..., 1) = uniform on simplex
    policy = rng.dirichlet(np.ones(env.n_actions), size=env.n_states)
    return policy.astype(np.float64)


def random_deterministic_policy(
    env: "GridWorldEnv",
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Create a random deterministic policy.
    
    For each state s, randomly chooses one action uniformly at random
    and assigns probability 1 to that action, 0 to all others.
    
    Args:
        env: The GridWorld environment.
        rng: Random number generator. If None, uses numpy's default.
    
    Returns:
        Policy array of shape (n_states, n_actions) where each row has
        exactly one entry equal to 1 and all others equal to 0.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> rng = np.random.default_rng(42)
        >>> pi = random_deterministic_policy(env, rng=rng)
        >>> assert all(np.max(pi[s]) == 1.0 for s in range(env.n_states))
    """
    if rng is None:
        rng = np.random.default_rng()
    
    policy = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    
    for s in range(env.n_states):
        chosen_action = rng.integers(0, env.n_actions)
        policy[s, chosen_action] = 1.0
    
    return policy


def policy_induced_transition_matrix(
    env: "GridWorldEnv",
    policy: np.ndarray
) -> np.ndarray:
    """
    Compute the policy-induced transition matrix P_π.
    
    Given the environment's transition kernel P[s, a, s'] and a policy π[s, a],
    computes the state-to-state transition probabilities under the policy:
    
        P_π[s, s'] = Σ_a π[s, a] * P[s, a, s']
    
    This gives the probability of transitioning from state s to state s'
    when following policy π.
    
    Args:
        env: The GridWorld environment with transition kernel P.
        policy: Policy array of shape (n_states, n_actions).
    
    Returns:
        Transition matrix of shape (n_states, n_states) where P_π[s, s']
        is the probability of going from s to s' under policy π.
        Each row sums to 1.
    
    Raises:
        ValueError: If policy shape doesn't match environment dimensions.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> P_pi = policy_induced_transition_matrix(env, pi)
        >>> assert P_pi.shape == (env.n_states, env.n_states)
        >>> assert np.allclose(P_pi.sum(axis=1), 1.0)
    """
    if policy.shape != (env.n_states, env.n_actions):
        raise ValueError(
            f"Policy shape {policy.shape} doesn't match expected "
            f"({env.n_states}, {env.n_actions})"
        )
    
    # P_pi[s, s'] = sum_a pi[s, a] * P[s, a, s']
    # Using einsum for clarity and efficiency
    # s=current state, a=action, t=next state (s')
    P_pi = np.einsum('sa,sat->st', policy, env.P)
    
    return P_pi


def policy_expected_reward(
    env: "GridWorldEnv",
    policy: np.ndarray
) -> np.ndarray:
    """
    Compute the expected reward for each state under a given policy.
    
    For each state s, computes:
        r_π(s) = Σ_a π[s, a] * Σ_{s'} P[s, a, s'] * R(s, a, s')
    
    Args:
        env: The GridWorld environment.
        policy: Policy array of shape (n_states, n_actions).
    
    Returns:
        Array of shape (n_states,) with expected reward for each state.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> r_pi = policy_expected_reward(env, pi)
    """
    if policy.shape != (env.n_states, env.n_actions):
        raise ValueError(
            f"Policy shape {policy.shape} doesn't match expected "
            f"({env.n_states}, {env.n_actions})"
        )
    
    # Get expected reward for each (s, a) pair
    R_sa = env.get_expected_reward()  # shape (n_states, n_actions)
    
    # Compute r_pi(s) = sum_a pi[s, a] * R[s, a]
    r_pi = np.sum(policy * R_sa, axis=1)
    
    return r_pi


def validate_policy(policy: np.ndarray, n_states: int, n_actions: int) -> bool:
    """
    Validate that a policy array is well-formed.
    
    Checks:
        - Shape is (n_states, n_actions)
        - All entries are non-negative
        - Each row sums to 1 (within tolerance)
    
    Args:
        policy: Policy array to validate.
        n_states: Expected number of states.
        n_actions: Expected number of actions.
    
    Returns:
        True if valid, False otherwise.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = uniform_policy(env)
        >>> assert validate_policy(pi, env.n_states, env.n_actions)
    """
    if policy.shape != (n_states, n_actions):
        return False
    
    if np.any(policy < 0):
        return False
    
    row_sums = policy.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        return False
    
    return True


def is_deterministic_policy(policy: np.ndarray) -> bool:
    """
    Check if a policy is deterministic.
    
    A deterministic policy has exactly one action with probability 1
    in each state.
    
    Args:
        policy: Policy array of shape (n_states, n_actions).
    
    Returns:
        True if deterministic, False otherwise.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi_det = random_deterministic_policy(env)
        >>> assert is_deterministic_policy(pi_det)
        >>> pi_stoch = uniform_policy(env)
        >>> assert not is_deterministic_policy(pi_stoch)
    """
    # Check if each row has exactly one 1 and rest 0s
    for s in range(policy.shape[0]):
        row = policy[s]
        if not (np.max(row) == 1.0 and np.sum(row) == 1.0 and np.count_nonzero(row) == 1):
            return False
    return True


def get_deterministic_actions(policy: np.ndarray) -> np.ndarray:
    """
    Extract the action for each state from a deterministic policy.
    
    Args:
        policy: Deterministic policy array of shape (n_states, n_actions).
    
    Returns:
        Array of shape (n_states,) with the chosen action for each state.
    
    Raises:
        ValueError: If policy is not deterministic.
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> pi = random_deterministic_policy(env)
        >>> actions = get_deterministic_actions(pi)
    """
    if not is_deterministic_policy(policy):
        raise ValueError("Policy is not deterministic")
    
    return np.argmax(policy, axis=1)
