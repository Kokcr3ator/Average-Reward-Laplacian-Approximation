"""
Unit Tests for GridWorldEnv

Tests the grid environment parsing, state mappings, transition kernel,
reward function, and teleportation logic.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from rlgrid.grid_env import GridWorldEnv


class TestGridParsing:
    """Tests for grid file parsing and validation."""
    
    def test_parse_simple_grid(self):
        """Test parsing a simple 5x5 grid."""
        grid_str = "#####\n#...#\n#.G.#\n#...#\n#####"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(grid_str)
            f.flush()
            
            env = GridWorldEnv.from_txt(f.name)
            
        os.unlink(f.name)
        
        assert env.height == 5
        assert env.width == 5
        assert env.n_states == 9  # 3x3 inner grid
        assert env.n_actions == 4
    
    def test_parse_grid_with_internal_walls(self):
        """Test parsing grid with internal walls."""
        grid_str = "########\n#......#\n#.##G#.#\n#......#\n########"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(grid_str)
            f.flush()
            
            env = GridWorldEnv.from_txt(f.name)
            
        os.unlink(f.name)
        
        assert env.height == 5
        assert env.width == 8
        # Count non-wall cells: 6 + 3 + 6 = 15 free cells (middle row has 3 walls)
        assert env.n_states == 15
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            GridWorldEnv.from_txt("nonexistent_file.txt")
    
    def test_empty_grid_error(self):
        """Test error on empty grid."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GridWorldEnv([])
    
    def test_no_goal_error(self):
        """Test error when grid has no goal."""
        grid = [
            list("#####"),
            list("#...#"),
            list("#...#"),
            list("#...#"),
            list("#####"),
        ]
        with pytest.raises(ValueError, match="exactly one goal"):
            GridWorldEnv(grid)
    
    def test_multiple_goals_error(self):
        """Test error when grid has multiple goals."""
        grid = [
            list("#####"),
            list("#G..#"),
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        with pytest.raises(ValueError, match="exactly one goal"):
            GridWorldEnv(grid)
    
    def test_invalid_character_error(self):
        """Test error on invalid grid character."""
        grid = [
            list("#####"),
            list("#.X.#"),  # X is invalid
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        with pytest.raises(ValueError, match="Invalid character"):
            GridWorldEnv(grid)
    
    def test_unequal_row_lengths_error(self):
        """Test error when rows have different lengths."""
        grid = [
            list("####"),
            list("#.G.#"),  # Different length
            list("####"),
        ]
        with pytest.raises(ValueError, match="same length"):
            GridWorldEnv(grid)


class TestStateMappings:
    """Tests for state-to-position mappings."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple 5x5 environment."""
        grid = [
            list("#####"),
            list("#...#"),
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        return GridWorldEnv(grid)
    
    def test_correct_number_of_states(self, simple_env):
        """Test that number of states matches non-wall cells."""
        assert simple_env.n_states == 9
    
    def test_state_to_pos_consistency(self, simple_env):
        """Test state_to_pos and pos_to_state are inverse mappings."""
        for state, pos in simple_env.state_to_pos.items():
            assert simple_env.pos_to_state[pos] == state
    
    def test_all_states_mapped(self, simple_env):
        """Test all states from 0 to n_states-1 are mapped."""
        states = set(simple_env.state_to_pos.keys())
        expected = set(range(simple_env.n_states))
        assert states == expected
    
    def test_goal_state_identified(self, simple_env):
        """Test goal state is correctly identified."""
        assert simple_env.goal_state is not None
        goal_pos = simple_env.state_to_pos[simple_env.goal_state]
        assert simple_env.grid[goal_pos[0]][goal_pos[1]] == 'G'
    
    def test_goal_position_matches(self, simple_env):
        """Test goal_pos matches state_to_pos for goal_state."""
        assert simple_env.goal_pos == simple_env.state_to_pos[simple_env.goal_state]


class TestTransitionKernel:
    """Tests for the transition kernel."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple 5x5 environment."""
        grid = [
            list("#####"),
            list("#...#"),
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        return GridWorldEnv(grid)
    
    def test_transition_kernel_shape(self, simple_env):
        """Test transition kernel has correct shape."""
        P = simple_env.get_transition_kernel()
        expected_shape = (simple_env.n_states, simple_env.n_actions, simple_env.n_states)
        assert P.shape == expected_shape
    
    def test_deterministic_transitions(self, simple_env):
        """Test each (s, a) has exactly one next state with prob 1."""
        P = simple_env.get_transition_kernel()
        
        for s in range(simple_env.n_states):
            for a in range(simple_env.n_actions):
                # Should have exactly one 1 and rest 0s
                assert np.sum(P[s, a, :]) == 1.0
                assert np.max(P[s, a, :]) == 1.0
                assert np.count_nonzero(P[s, a, :]) == 1
    
    def test_transition_probabilities_valid(self, simple_env):
        """Test all transition probabilities are 0 or 1."""
        P = simple_env.get_transition_kernel()
        assert np.all((P == 0) | (P == 1))
    
    def test_wall_collision_stays_in_place(self, simple_env):
        """Test moving into wall keeps agent in place."""
        # Find a state adjacent to a wall and verify it stays in place
        # State 0 is at (1,1) in the simple 5x5 grid
        # Up action (0) would go to (0,1) which is a wall
        
        P = simple_env.get_transition_kernel()
        
        # State at (1,1), action up should stay in place
        state_1_1 = simple_env.pos_to_state[(1, 1)]
        action_up = 0
        next_state = np.argmax(P[state_1_1, action_up])
        
        assert next_state == state_1_1  # Should stay in place
    
    def test_valid_move_changes_state(self, simple_env):
        """Test valid move transitions to correct state."""
        P = simple_env.get_transition_kernel()
        
        # State at (1,1), action right should go to (1,2)
        state_1_1 = simple_env.pos_to_state[(1, 1)]
        state_1_2 = simple_env.pos_to_state[(1, 2)]
        action_right = 1
        
        next_state = np.argmax(P[state_1_1, action_right])
        assert next_state == state_1_2


class TestRewardFunction:
    """Tests for the reward function."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple 5x5 environment."""
        grid = [
            list("#####"),
            list("#...#"),
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        return GridWorldEnv(grid)
    
    def test_goal_reward_positive(self, simple_env):
        """Test entering goal gives +1 reward."""
        goal_state = simple_env.goal_state
        for s in range(simple_env.n_states):
            for a in range(simple_env.n_actions):
                r = simple_env.reward(s, a, goal_state)
                assert r == 1.0
    
    def test_non_goal_reward_negative(self, simple_env):
        """Test non-goal transitions give -1 reward."""
        goal_state = simple_env.goal_state
        for s in range(simple_env.n_states):
            for a in range(simple_env.n_actions):
                for s_next in range(simple_env.n_states):
                    if s_next != goal_state:
                        r = simple_env.reward(s, a, s_next)
                        assert r == -1.0
    
    def test_reward_matrix_shape(self, simple_env):
        """Test reward matrix has correct shape."""
        R = simple_env.get_reward_matrix()
        expected_shape = (simple_env.n_states, simple_env.n_actions, simple_env.n_states)
        assert R.shape == expected_shape
    
    def test_expected_reward_shape(self, simple_env):
        """Test expected reward has correct shape."""
        R_expected = simple_env.get_expected_reward()
        expected_shape = (simple_env.n_states, simple_env.n_actions)
        assert R_expected.shape == expected_shape


class TestStepFunction:
    """Tests for the step function and teleportation."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple 5x5 environment with fixed RNG."""
        grid = [
            list("#####"),
            list("#...#"),
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        rng = np.random.default_rng(42)
        return GridWorldEnv(grid, rng=rng)
    
    def test_step_returns_tuple(self, simple_env):
        """Test step returns (next_state, reward) tuple."""
        result = simple_env.step(0, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_step_non_goal_reward(self, simple_env):
        """Test step gives -1 reward when not reaching goal."""
        # From state 0, any action that doesn't reach goal
        state = 0
        for a in range(simple_env.n_actions):
            # Check if this action leads to goal
            P = simple_env.get_transition_kernel()
            next_deterministic = np.argmax(P[state, a])
            
            if next_deterministic != simple_env.goal_state:
                _, reward = simple_env.step(state, a)
                assert reward == -1.0
    
    def test_step_goal_reward(self, simple_env):
        """Test step gives +1 reward when reaching goal."""
        # Find a state adjacent to goal
        goal_row, goal_col = simple_env.goal_pos
        
        # Check each neighbor of the goal
        for action, (dr, dc) in simple_env.ACTION_DELTAS.items():
            neighbor_pos = (goal_row - dr, goal_col - dc)  # Reverse delta
            if neighbor_pos in simple_env.pos_to_state:
                state = simple_env.pos_to_state[neighbor_pos]
                
                # Check if action from neighbor leads to goal
                P = simple_env.get_transition_kernel()
                if np.argmax(P[state, action]) == simple_env.goal_state:
                    _, reward = simple_env.step(state, action)
                    assert reward == 1.0
    
    def test_teleportation_on_goal(self, simple_env):
        """Test teleportation to non-goal state after reaching goal."""
        # Find a state adjacent to goal and action that leads to goal
        goal_row, goal_col = simple_env.goal_pos
        P = simple_env.get_transition_kernel()
        
        for state in range(simple_env.n_states):
            for action in range(simple_env.n_actions):
                if np.argmax(P[state, action]) == simple_env.goal_state:
                    # This (state, action) leads to goal
                    # After teleportation, should be in non-goal state
                    for _ in range(10):  # Test multiple times due to randomness
                        next_state, reward = simple_env.step(state, action)
                        assert reward == 1.0
                        assert next_state != simple_env.goal_state
                        assert next_state in simple_env._non_goal_states
                    return  # Test passed
        
        pytest.fail("No state-action pair leads to goal")
    
    def test_invalid_state_error(self, simple_env):
        """Test error on invalid state."""
        with pytest.raises(ValueError, match="Invalid state"):
            simple_env.step(-1, 0)
        
        with pytest.raises(ValueError, match="Invalid state"):
            simple_env.step(simple_env.n_states, 0)
    
    def test_invalid_action_error(self, simple_env):
        """Test error on invalid action."""
        with pytest.raises(ValueError, match="Invalid action"):
            simple_env.step(0, -1)
        
        with pytest.raises(ValueError, match="Invalid action"):
            simple_env.step(0, simple_env.n_actions)


class TestMinimalEnvironment:
    """Tests for edge cases with minimal environments."""
    
    def test_single_cell_goal(self):
        """Test environment with only goal cell."""
        grid = [
            list("###"),
            list("#G#"),
            list("###"),
        ]
        env = GridWorldEnv(grid)
        
        assert env.n_states == 1
        assert env.goal_state == 0
        
        # All actions should stay in place (no non-goal states to teleport to)
        for a in range(env.n_actions):
            next_state, reward = env.step(0, a)
            assert reward == 1.0  # Always reaching goal
            assert next_state == 0  # No other state to teleport to
    
    def test_two_cell_environment(self):
        """Test environment with two cells."""
        grid = [
            list("###"),
            list("#.#"),
            list("#G#"),
            list("###"),
        ]
        env = GridWorldEnv(grid)
        
        assert env.n_states == 2


class TestEnvironmentRepr:
    """Tests for string representations."""
    
    @pytest.fixture
    def simple_env(self):
        grid = [
            list("#####"),
            list("#...#"),
            list("#.G.#"),
            list("#...#"),
            list("#####"),
        ]
        return GridWorldEnv(grid)
    
    def test_repr(self, simple_env):
        """Test __repr__ returns valid string."""
        r = repr(simple_env)
        assert "GridWorldEnv" in r
        assert str(simple_env.n_states) in r
    
    def test_str(self, simple_env):
        """Test __str__ returns grid."""
        s = str(simple_env)
        assert "#####" in s
        assert "G" in s
    
    def test_get_grid_string(self, simple_env):
        """Test get_grid_string method."""
        s = simple_env.get_grid_string()
        lines = s.split('\n')
        assert len(lines) == 5
        assert all(len(line) == 5 for line in lines)
