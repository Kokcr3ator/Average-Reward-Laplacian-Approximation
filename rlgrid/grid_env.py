"""
GridWorld Environment Module

This module provides the GridWorldEnv class for creating and managing
deterministic gridworld environments for average-reward reinforcement learning.

The environment is parsed from text files where:
    - '.' represents free cells
    - '#' represents walls (non-traversable)
    - 'G' represents the unique goal cell

The MDP operates in a continuing (average-reward) setting:
    - Entering the goal from a non-goal state yields +1 reward, next state is goal
    - Taking any action from the goal state teleports to a random non-goal state, reward -1
    - All other transitions yield -1 reward
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


class GridWorldEnv:
    """
    A deterministic gridworld environment for average-reward reinforcement learning.
    
    The environment supports four actions: up, right, down, left.
    Transitions are deterministic - if an action would lead into a wall or
    out of bounds, the agent stays in place.
    
    This is a continuing MDP with teleportation from the goal state to
    enable average-reward analysis.
    
    Attributes:
        grid: 2D list of characters representing the grid.
        height: Number of rows in the grid.
        width: Number of columns in the grid.
        n_states: Number of non-wall states.
        n_actions: Number of actions (always 4).
        state_to_pos: Mapping from state index to (row, col) position.
        pos_to_state: Mapping from (row, col) position to state index.
        goal_state: The state index of the goal cell.
        goal_pos: The (row, col) position of the goal cell.
        P: Transition kernel of shape (n_states, n_actions, n_states).
        ACTIONS: Dictionary mapping action indices to action names.
        ACTION_DELTAS: Dictionary mapping action indices to (delta_row, delta_col).
    
    Example:
        >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        >>> print(f"States: {env.n_states}, Goal: {env.goal_state}")
        >>> next_state, reward = env.step(0, 1)  # Take action 'right' from state 0
    """
    
    # Action definitions
    ACTIONS: Dict[int, str] = {
        0: "up",
        1: "right", 
        2: "down",
        3: "left"
    }
    
    # Movement deltas: (delta_row, delta_col) for each action
    ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),   # up: decrease row
        1: (0, 1),    # right: increase col
        2: (1, 0),    # down: increase row
        3: (0, -1)    # left: decrease col
    }
    
    def __init__(
        self,
        grid: List[List[str]],
        rng: Optional[np.random.Generator] = None
    ) -> None:
        """
        Initialize the GridWorldEnv from a parsed grid.
        
        Args:
            grid: 2D list of characters ('.', '#', 'G').
            rng: Random number generator for teleportation. If None, uses default.
        
        Raises:
            ValueError: If grid is invalid (no goal, multiple goals, empty, etc.)
        """
        self._validate_grid(grid)
        
        self.grid: List[List[str]] = grid
        self.height: int = len(grid)
        self.width: int = len(grid[0])
        self.n_actions: int = 4
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        
        # Build state mappings
        self._build_state_mappings()
        
        # Build transition kernel
        self.P: np.ndarray = self._build_transition_kernel()
    
    @classmethod
    def from_txt(
        cls,
        path: str | Path,
        rng: Optional[np.random.Generator] = None
    ) -> "GridWorldEnv":
        """
        Load and build a GridWorldEnv from a .txt grid file.
        
        Args:
            path: Path to the .txt file containing the grid definition.
            rng: Random number generator for teleportation.
        
        Returns:
            A configured GridWorldEnv instance.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains an invalid grid.
        
        Example:
            >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Grid file not found: {path}")
        
        with open(path, 'r') as f:
            lines = f.read().strip().split('\n')
        
        grid = [list(line) for line in lines]
        return cls(grid, rng=rng)
    
    def _validate_grid(self, grid: List[List[str]]) -> None:
        """
        Validate that the grid is well-formed.
        
        Args:
            grid: 2D list of characters.
        
        Raises:
            ValueError: If validation fails.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        
        width = len(grid[0])
        goal_count = 0
        valid_chars = {'.', '#', 'G'}
        
        for row_idx, row in enumerate(grid):
            if len(row) != width:
                raise ValueError(
                    f"Row {row_idx} has length {len(row)}, expected {width}. "
                    "All rows must have the same length."
                )
            for col_idx, char in enumerate(row):
                if char not in valid_chars:
                    raise ValueError(
                        f"Invalid character '{char}' at position ({row_idx}, {col_idx}). "
                        f"Valid characters are: {valid_chars}"
                    )
                if char == 'G':
                    goal_count += 1
        
        if goal_count == 0:
            raise ValueError("Grid must contain exactly one goal 'G'. Found 0.")
        if goal_count > 1:
            raise ValueError(f"Grid must contain exactly one goal 'G'. Found {goal_count}.")
    
    def _build_state_mappings(self) -> None:
        """
        Build mappings between state indices and grid positions.
        
        Assigns sequential state indices to all non-wall cells,
        scanning row by row from top-left.
        """
        self.state_to_pos: Dict[int, Tuple[int, int]] = {}
        self.pos_to_state: Dict[Tuple[int, int], int] = {}
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.goal_state: Optional[int] = None
        
        state_idx = 0
        for row in range(self.height):
            for col in range(self.width):
                char = self.grid[row][col]
                if char != '#':  # Non-wall cell
                    pos = (row, col)
                    self.state_to_pos[state_idx] = pos
                    self.pos_to_state[pos] = state_idx
                    
                    if char == 'G':
                        self.goal_pos = pos
                        self.goal_state = state_idx
                    
                    state_idx += 1
        
        self.n_states: int = state_idx
        
        # Build list of non-goal states for teleportation
        self._non_goal_states: List[int] = [
            s for s in range(self.n_states) if s != self.goal_state
        ]
    
    def _build_transition_kernel(self) -> np.ndarray:
        """
        Build the deterministic transition kernel P(s' | s, a).
        
        For each state s and action a, determines the resulting state s'.
        If the action would lead to a wall or out of bounds, the agent stays in s.
        
        Returns:
            Numpy array of shape (n_states, n_actions, n_states) where
            P[s, a, s'] = 1 if taking action a in state s leads to s', else 0.
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        
        for s in range(self.n_states):
            row, col = self.state_to_pos[s]
            
            for a in range(self.n_actions):
                delta_row, delta_col = self.ACTION_DELTAS[a]
                new_row = row + delta_row
                new_col = col + delta_col
                
                # Check if new position is valid (in bounds and not a wall)
                if self._is_valid_position(new_row, new_col):
                    s_next = self.pos_to_state[(new_row, new_col)]
                else:
                    # Stay in place if move is invalid
                    s_next = s
                
                P[s, a, s_next] = 1.0
        
        return P
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """
        Check if a position is valid (in bounds and not a wall).
        
        Args:
            row: Row index.
            col: Column index.
        
        Returns:
            True if position is valid, False otherwise.
        """
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return self.grid[row][col] != '#'
    
    def reward(self, s: int, a: int, s_next: int) -> float:
        """
        Compute the reward for a transition.
        
        Args:
            s: Current state index.
            a: Action taken.
            s_next: Next state index (before teleportation).
        
        Returns:
            +1.0 if s_next is the goal state, -1.0 otherwise.
        """
        if s_next == self.goal_state:
            return 1.0
        return -1.0
    
    def step(self, s: int, a: int) -> Tuple[int, float]:
        """
        Execute one step in the environment.
        
        The agent takes action a in state s. The transition is deterministic.
        - If the agent is at the goal state: teleport to a random non-goal state, reward is -1
        - If the resulting state is the goal: move to goal, reward is +1
        - Otherwise: move to the new state, reward is -1
        
        Args:
            s: Current state index (0 to n_states-1).
            a: Action index (0 to n_actions-1).
        
        Returns:
            Tuple of (next_state, reward).
        
        Raises:
            ValueError: If state or action is out of bounds.
        
        Example:
            >>> env = GridWorldEnv.from_txt("envs/simple_5x5.txt")
            >>> next_state, reward = env.step(0, 1)  # Take 'right' from state 0
        """
        if s < 0 or s >= self.n_states:
            raise ValueError(f"Invalid state {s}. Must be in [0, {self.n_states - 1}]")
        if a < 0 or a >= self.n_actions:
            raise ValueError(f"Invalid action {a}. Must be in [0, {self.n_actions - 1}]")
        
        # If at goal state, teleport to random non-goal state
        if s == self.goal_state:
            if len(self._non_goal_states) > 0:
                s_next = self.rng.choice(self._non_goal_states)
            else:
                # If there are no non-goal states (single cell grid), stay at goal
                s_next = self.goal_state
            r = -1.0  # Reward for leaving goal (not entering it)
            return s_next, r
        
        # Get next state from transition kernel
        s_next = int(np.argmax(self.P[s, a]))
        
        # Compute reward based on reaching goal
        r = self.reward(s, a, s_next)
        
        return s_next, r
    
    def get_transition_kernel(self) -> np.ndarray:
        """
        Get the transition kernel P(s' | s, a).
        
        Returns:
            Numpy array of shape (n_states, n_actions, n_states).
        """
        return self.P.copy()
    
    def get_reward_matrix(self) -> np.ndarray:
        """
        Get the reward matrix R(s, a, s').
        
        Returns:
            Numpy array of shape (n_states, n_actions, n_states) where
            R[s, a, s'] is the reward for transition (s, a) -> s'.
        """
        R = np.full((self.n_states, self.n_actions, self.n_states), -1.0)
        # Reward +1 for entering the goal state
        R[:, :, self.goal_state] = 1.0
        return R
    
    def get_expected_reward(self) -> np.ndarray:
        """
        Get the expected reward for each (state, action) pair.
        
        For deterministic environments, this is simply R(s, a, s') where
        s' is the unique next state.
        
        Returns:
            Numpy array of shape (n_states, n_actions).
        """
        R_full = self.get_reward_matrix()
        # Sum over s' weighted by P (for deterministic, equivalent to just picking the one s')
        return np.sum(self.P * R_full, axis=2)
    
    def get_grid_string(self) -> str:
        """
        Get a string representation of the grid.
        
        Returns:
            Multi-line string showing the grid.
        """
        return '\n'.join(''.join(row) for row in self.grid)
    
    def __repr__(self) -> str:
        """Return a string representation of the environment."""
        return (
            f"GridWorldEnv(height={self.height}, width={self.width}, "
            f"n_states={self.n_states}, goal_state={self.goal_state})"
        )
    
    def __str__(self) -> str:
        """Return the grid as a string."""
        return self.get_grid_string()
