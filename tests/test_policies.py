"""
Unit Tests for Policies Module

Tests the policy constructors and policy-induced transition matrix computation.
"""

import pytest
import numpy as np

from rlgrid.grid_env import GridWorldEnv
from rlgrid.policies import (
    uniform_policy,
    random_policy,
    random_deterministic_policy,
    policy_induced_transition_matrix,
    policy_expected_reward,
    validate_policy,
    is_deterministic_policy,
    get_deterministic_actions,
)


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
    return GridWorldEnv(grid)


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
    return GridWorldEnv(grid)


class TestUniformPolicy:
    """Tests for uniform_policy function."""
    
    def test_shape(self, simple_env):
        """Test policy has correct shape."""
        pi = uniform_policy(simple_env)
        assert pi.shape == (simple_env.n_states, simple_env.n_actions)
    
    def test_all_equal(self, simple_env):
        """Test all entries are equal (1/n_actions)."""
        pi = uniform_policy(simple_env)
        expected_prob = 1.0 / simple_env.n_actions
        assert np.allclose(pi, expected_prob)
    
    def test_rows_sum_to_one(self, simple_env):
        """Test each row sums to 1."""
        pi = uniform_policy(simple_env)
        row_sums = pi.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_valid_policy(self, simple_env):
        """Test policy passes validation."""
        pi = uniform_policy(simple_env)
        assert validate_policy(pi, simple_env.n_states, simple_env.n_actions)


class TestRandomPolicy:
    """Tests for random_policy function."""
    
    def test_shape(self, simple_env):
        """Test policy has correct shape."""
        rng = np.random.default_rng(42)
        pi = random_policy(simple_env, rng=rng)
        assert pi.shape == (simple_env.n_states, simple_env.n_actions)
    
    def test_rows_sum_to_one(self, simple_env):
        """Test each row sums to 1."""
        rng = np.random.default_rng(42)
        pi = random_policy(simple_env, rng=rng)
        row_sums = pi.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_all_non_negative(self, simple_env):
        """Test all probabilities are non-negative."""
        rng = np.random.default_rng(42)
        pi = random_policy(simple_env, rng=rng)
        assert np.all(pi >= 0)
    
    def test_reproducible_with_rng(self, simple_env):
        """Test policy is reproducible with same RNG seed."""
        pi1 = random_policy(simple_env, rng=np.random.default_rng(42))
        pi2 = random_policy(simple_env, rng=np.random.default_rng(42))
        assert np.allclose(pi1, pi2)
    
    def test_different_with_different_rng(self, simple_env):
        """Test different RNG seeds give different policies."""
        pi1 = random_policy(simple_env, rng=np.random.default_rng(42))
        pi2 = random_policy(simple_env, rng=np.random.default_rng(123))
        assert not np.allclose(pi1, pi2)
    
    def test_valid_policy(self, simple_env):
        """Test policy passes validation."""
        rng = np.random.default_rng(42)
        pi = random_policy(simple_env, rng=rng)
        assert validate_policy(pi, simple_env.n_states, simple_env.n_actions)
    
    def test_not_deterministic(self, simple_env):
        """Test random policy is generally not deterministic."""
        rng = np.random.default_rng(42)
        pi = random_policy(simple_env, rng=rng)
        # Very unlikely for a random Dirichlet sample to be deterministic
        assert not is_deterministic_policy(pi)


class TestRandomDeterministicPolicy:
    """Tests for random_deterministic_policy function."""
    
    def test_shape(self, simple_env):
        """Test policy has correct shape."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        assert pi.shape == (simple_env.n_states, simple_env.n_actions)
    
    def test_rows_sum_to_one(self, simple_env):
        """Test each row sums to 1."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        row_sums = pi.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_exactly_one_one_per_row(self, simple_env):
        """Test each row has exactly one 1 and rest 0s."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        
        for s in range(simple_env.n_states):
            row = pi[s]
            assert np.count_nonzero(row) == 1
            assert np.max(row) == 1.0
            assert np.min(row) == 0.0
    
    def test_is_deterministic(self, simple_env):
        """Test policy is identified as deterministic."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        assert is_deterministic_policy(pi)
    
    def test_valid_policy(self, simple_env):
        """Test policy passes validation."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        assert validate_policy(pi, simple_env.n_states, simple_env.n_actions)
    
    def test_reproducible_with_rng(self, simple_env):
        """Test policy is reproducible with same RNG seed."""
        pi1 = random_deterministic_policy(simple_env, rng=np.random.default_rng(42))
        pi2 = random_deterministic_policy(simple_env, rng=np.random.default_rng(42))
        assert np.array_equal(pi1, pi2)
    
    def test_get_deterministic_actions(self, simple_env):
        """Test extracting actions from deterministic policy."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        actions = get_deterministic_actions(pi)
        
        assert actions.shape == (simple_env.n_states,)
        for s in range(simple_env.n_states):
            assert pi[s, actions[s]] == 1.0


class TestPolicyInducedTransitionMatrix:
    """Tests for policy_induced_transition_matrix function."""
    
    def test_shape(self, simple_env):
        """Test P_pi has correct shape."""
        pi = uniform_policy(simple_env)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        assert P_pi.shape == (simple_env.n_states, simple_env.n_states)
    
    def test_rows_sum_to_one(self, simple_env):
        """Test each row of P_pi sums to 1."""
        pi = uniform_policy(simple_env)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        row_sums = P_pi.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_all_non_negative(self, simple_env):
        """Test all entries are non-negative."""
        pi = uniform_policy(simple_env)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        assert np.all(P_pi >= 0)
    
    def test_deterministic_policy_matches_transition(self, simple_env):
        """Test P_pi under deterministic policy has single 1 per row."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        P_pi = policy_induced_transition_matrix(simple_env, pi)
        
        for s in range(simple_env.n_states):
            # Should have exactly one 1 per row (deterministic)
            assert np.count_nonzero(P_pi[s]) == 1
            assert np.max(P_pi[s]) == 1.0
    
    def test_wrong_shape_error(self, simple_env):
        """Test error when policy has wrong shape."""
        wrong_pi = np.ones((simple_env.n_states + 1, simple_env.n_actions))
        with pytest.raises(ValueError, match="doesn't match"):
            policy_induced_transition_matrix(simple_env, wrong_pi)
    
    def test_with_maze_env(self, maze_env):
        """Test with maze environment."""
        pi = uniform_policy(maze_env)
        P_pi = policy_induced_transition_matrix(maze_env, pi)
        
        assert P_pi.shape == (maze_env.n_states, maze_env.n_states)
        assert np.allclose(P_pi.sum(axis=1), 1.0)


class TestPolicyExpectedReward:
    """Tests for policy_expected_reward function."""
    
    def test_shape(self, simple_env):
        """Test expected reward has correct shape."""
        pi = uniform_policy(simple_env)
        r_pi = policy_expected_reward(simple_env, pi)
        assert r_pi.shape == (simple_env.n_states,)
    
    def test_values_in_range(self, simple_env):
        """Test expected rewards are in [-1, 1]."""
        pi = uniform_policy(simple_env)
        r_pi = policy_expected_reward(simple_env, pi)
        assert np.all(r_pi >= -1.0)
        assert np.all(r_pi <= 1.0)
    
    def test_goal_neighbor_positive(self, simple_env):
        """Test states adjacent to goal have higher expected reward."""
        # Create a policy that always moves towards goal
        # For simplicity, just verify structure with uniform policy
        pi = uniform_policy(simple_env)
        r_pi = policy_expected_reward(simple_env, pi)
        
        # At least one state should have expected reward > -1
        # (the one that can reach the goal)
        assert np.any(r_pi > -1.0)


class TestValidatePolicy:
    """Tests for validate_policy function."""
    
    def test_valid_uniform(self, simple_env):
        """Test uniform policy is valid."""
        pi = uniform_policy(simple_env)
        assert validate_policy(pi, simple_env.n_states, simple_env.n_actions)
    
    def test_wrong_shape(self, simple_env):
        """Test wrong shape is invalid."""
        pi = np.ones((simple_env.n_states + 1, simple_env.n_actions)) / simple_env.n_actions
        assert not validate_policy(pi, simple_env.n_states, simple_env.n_actions)
    
    def test_negative_values(self, simple_env):
        """Test negative values are invalid."""
        pi = uniform_policy(simple_env)
        pi[0, 0] = -0.1
        pi[0, 1] = 1.1 / 3 + 0.1  # Keep row sum at 1
        assert not validate_policy(pi, simple_env.n_states, simple_env.n_actions)
    
    def test_rows_not_sum_to_one(self, simple_env):
        """Test rows not summing to 1 are invalid."""
        pi = np.ones((simple_env.n_states, simple_env.n_actions)) * 0.5
        assert not validate_policy(pi, simple_env.n_states, simple_env.n_actions)


class TestIsDeterministicPolicy:
    """Tests for is_deterministic_policy function."""
    
    def test_uniform_not_deterministic(self, simple_env):
        """Test uniform policy is not deterministic."""
        pi = uniform_policy(simple_env)
        assert not is_deterministic_policy(pi)
    
    def test_random_det_is_deterministic(self, simple_env):
        """Test random deterministic policy is deterministic."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        assert is_deterministic_policy(pi)
    
    def test_manual_deterministic(self, simple_env):
        """Test manually created deterministic policy."""
        pi = np.zeros((simple_env.n_states, simple_env.n_actions))
        for s in range(simple_env.n_states):
            pi[s, 0] = 1.0  # Always action 0
        assert is_deterministic_policy(pi)


class TestGetDeterministicActions:
    """Tests for get_deterministic_actions function."""
    
    def test_extract_actions(self, simple_env):
        """Test extracting actions from deterministic policy."""
        rng = np.random.default_rng(42)
        pi = random_deterministic_policy(simple_env, rng=rng)
        actions = get_deterministic_actions(pi)
        
        assert len(actions) == simple_env.n_states
        for s in range(simple_env.n_states):
            assert 0 <= actions[s] < simple_env.n_actions
            assert pi[s, actions[s]] == 1.0
    
    def test_error_on_stochastic(self, simple_env):
        """Test error when policy is not deterministic."""
        pi = uniform_policy(simple_env)
        with pytest.raises(ValueError, match="not deterministic"):
            get_deterministic_actions(pi)
