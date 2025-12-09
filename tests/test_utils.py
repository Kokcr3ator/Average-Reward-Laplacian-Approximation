"""
Unit Tests for Utils Module

Tests simulation, average reward estimation, and other utility functions.
"""

import pytest
import numpy as np

from rlgrid.grid_env import GridWorldEnv
from rlgrid.policies import uniform_policy, random_deterministic_policy
from rlgrid.utils import (
    simulate,
    estimate_average_reward,
    estimate_average_reward_running,
    compute_stationary_distribution,
    compute_average_reward_exact,
)
from rlgrid.policies import policy_induced_transition_matrix


@pytest.fixture
def simple_env():
    """Create a simple 5x5 environment."""
    grid = [
        list("#####"),
        list("#...#"),
        list("#.G.#"),
        list("#...#"),
        list("#####"),
    ]
    rng = np.random.default_rng(42)
    return GridWorldEnv(grid, rng=rng)


@pytest.fixture
def maze_env():
    """Create a maze environment."""
    grid = [
        list("########"),
        list("#......#"),
        list("#.##G#.#"),
        list("#......#"),
        list("########"),
    ]
    rng = np.random.default_rng(42)
    return GridWorldEnv(grid, rng=rng)


class TestSimulate:
    """Tests for simulate function."""
    
    def test_output_shapes(self, simple_env):
        """Test output arrays have correct shapes."""
        pi = uniform_policy(simple_env)
        T = 100
        rng = np.random.default_rng(42)
        
        states, actions, rewards = simulate(simple_env, pi, T=T, rng=rng)
        
        assert states.shape == (T + 1,)
        assert actions.shape == (T,)
        assert rewards.shape == (T,)
    
    def test_states_valid(self, simple_env):
        """Test all states are valid."""
        pi = uniform_policy(simple_env)
        T = 100
        rng = np.random.default_rng(42)
        
        states, _, _ = simulate(simple_env, pi, T=T, rng=rng)
        
        assert np.all(states >= 0)
        assert np.all(states < simple_env.n_states)
    
    def test_actions_valid(self, simple_env):
        """Test all actions are valid."""
        pi = uniform_policy(simple_env)
        T = 100
        rng = np.random.default_rng(42)
        
        _, actions, _ = simulate(simple_env, pi, T=T, rng=rng)
        
        assert np.all(actions >= 0)
        assert np.all(actions < simple_env.n_actions)
    
    def test_rewards_valid(self, simple_env):
        """Test all rewards are -1 or +1."""
        pi = uniform_policy(simple_env)
        T = 100
        rng = np.random.default_rng(42)
        
        _, _, rewards = simulate(simple_env, pi, T=T, rng=rng)
        
        assert np.all((rewards == -1.0) | (rewards == 1.0))
    
    def test_start_state_used(self, simple_env):
        """Test specified start state is used."""
        pi = uniform_policy(simple_env)
        T = 10
        rng = np.random.default_rng(42)
        start = 5
        
        states, _, _ = simulate(simple_env, pi, T=T, start_state=start, rng=rng)
        
        assert states[0] == start
    
    def test_invalid_start_state_error(self, simple_env):
        """Test error on invalid start state."""
        pi = uniform_policy(simple_env)
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="Invalid start_state"):
            simulate(simple_env, pi, T=10, start_state=-1, rng=rng)
        
        with pytest.raises(ValueError, match="Invalid start_state"):
            simulate(simple_env, pi, T=10, start_state=simple_env.n_states, rng=rng)
    
    def test_wrong_policy_shape_error(self, simple_env):
        """Test error when policy has wrong shape."""
        wrong_pi = np.ones((simple_env.n_states + 1, simple_env.n_actions))
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="doesn't match"):
            simulate(simple_env, wrong_pi, T=10, rng=rng)
    
    def test_reproducible_with_rng(self, simple_env):
        """Test simulation is reproducible with same RNG and same env RNG."""
        pi = uniform_policy(simple_env)
        T = 50
        
        # Create two separate environments with same RNG seed
        grid = [
            list("#####"),
            list("#...#"),
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        env1 = GridWorldEnv(grid, rng=np.random.default_rng(42))
        env2 = GridWorldEnv(grid, rng=np.random.default_rng(42))
        
        states1, actions1, rewards1 = simulate(
            env1, pi, T=T, start_state=0, rng=np.random.default_rng(100)
        )
        states2, actions2, rewards2 = simulate(
            env2, pi, T=T, start_state=0, rng=np.random.default_rng(100)
        )
        
        assert np.array_equal(states1, states2)
        assert np.array_equal(actions1, actions2)
        assert np.array_equal(rewards1, rewards2)
    
    def test_goal_reached(self, simple_env):
        """Test that goal is reached in long simulation."""
        pi = uniform_policy(simple_env)
        T = 1000
        rng = np.random.default_rng(42)
        
        _, _, rewards = simulate(simple_env, pi, T=T, rng=rng)
        
        # Should reach goal at least once in 1000 steps
        assert np.any(rewards == 1.0)
    
    def test_teleportation_works(self, simple_env):
        """Test that teleportation happens when leaving goal."""
        pi = uniform_policy(simple_env)
        T = 1000
        rng = np.random.default_rng(42)
        
        states, _, rewards = simulate(simple_env, pi, T=T, rng=rng)
        
        # After a +1 reward, the next state should BE the goal
        # Then after -1 reward from goal, next state should NOT be goal (teleported)
        for t in range(len(rewards) - 1):
            if rewards[t] == 1.0:
                # After +1 reward, we should be at the goal
                assert states[t + 1] == simple_env.goal_state
                # Next step from goal should teleport (reward -1)
                assert rewards[t + 1] == -1.0
                # And we should no longer be at goal
                assert states[t + 2] != simple_env.goal_state

class TestComputeStationaryDistribution:
    """Tests for compute_stationary_distribution function."""
    
    def test_sums_to_one(self, simple_env):
        """Test stationary distribution sums to 1."""
        pi = uniform_policy(simple_env)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        mu = compute_stationary_distribution(P_pi)
        
        assert np.isclose(mu.sum(), 1.0)
    
    def test_all_non_negative(self, simple_env):
        """Test all probabilities are non-negative."""
        pi = uniform_policy(simple_env)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        mu = compute_stationary_distribution(P_pi)
        
        assert np.all(mu >= 0)
    
    def test_is_stationary(self, simple_env):
        """Test mu @ P = mu (stationarity condition)."""
        pi = uniform_policy(simple_env)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        mu = compute_stationary_distribution(P_pi)
        
        # mu @ P should equal mu
        mu_next = mu @ P_pi
        assert np.allclose(mu, mu_next)
    
    def test_correct_shape(self, simple_env):
        """Test output has correct shape."""
        pi = uniform_policy(simple_env)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        mu = compute_stationary_distribution(P_pi)
        
        assert mu.shape == (simple_env.n_states,)
    
    def test_with_deterministic_policy(self, simple_env):
        """Test with deterministic policy."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        mu = compute_stationary_distribution(P_pi)
        
        assert np.isclose(mu.sum(), 1.0)
        assert np.all(mu >= 0)
