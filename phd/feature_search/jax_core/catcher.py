from typing import Optional, Tuple, Dict, Any, List
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import numpy as np

class CatchEnvironment(eqx.Module):
    """
    An implementation of the Catch environment with additional features like hot bit,
    reset bit, catch bit, miss bit, plus bit, and minus bit.
    """
    
    # Static parameters (configuration)
    rows: int = eqx.field(static=True)
    cols: int = eqx.field(static=True)
    hot_prob: float
    reset_prob: float
    reward_indicator_duration_min: int
    reward_indicator_duration_max: int
    
    # Dynamic parameters (state)
    rng: random.PRNGKey
    ball_row: jax.Array  # Current row of the ball
    ball_col: jax.Array  # Current column of the ball
    paddle_col: jax.Array  # Current column of the paddle
    in_reset: jax.Array  # Whether the ball is in reset state
    is_hot: jax.Array  # Whether the board is hot
    catch_bit: jax.Array  # Whether the ball was just caught
    miss_bit: jax.Array  # Whether the ball was just missed
    plus_bit: jax.Array  # Whether a positive reward is forthcoming
    minus_bit: jax.Array  # Whether a negative reward is forthcoming
    reward_countdown: jax.Array  # Countdown for reward after plus/minus bit activation
    
    def __init__(
        self,
        rows: int = 10,
        cols: int = 5,
        hot_prob: float = 0.3,
        reset_prob: float = 0.2,
        reward_indicator_duration_min: int = 1,
        reward_indicator_duration_max: int = 3,
        seed: Optional[int] = None,
    ):
        """Initialize the Catch environment with the given parameters."""
        super().__init__()
        
        # Store static configuration
        self.rows = rows
        self.cols = cols
        self.hot_prob = hot_prob
        self.reset_prob = reset_prob
        self.reward_indicator_duration_min = reward_indicator_duration_min
        self.reward_indicator_duration_max = reward_indicator_duration_max
        
        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 1000000000)
        key = random.PRNGKey(seed)
        
        # Initialize dynamic state
        key, subkey = random.split(key)
        self.rng = key
        
        # Start with ball in reset state
        self.ball_row = jnp.array(-1)  # -1 represents reset state
        self.ball_col = jnp.array(0)
        
        # Start with paddle in middle position
        self.paddle_col = jnp.array(cols // 2)
        
        # Initialize all bits as inactive
        self.in_reset = jnp.array(True)
        self.is_hot = jnp.array(False)
        self.catch_bit = jnp.array(False)
        self.miss_bit = jnp.array(False)
        self.plus_bit = jnp.array(False)
        self.minus_bit = jnp.array(False)
        self.reward_countdown = jnp.array(0)
    
    def _get_observation(self) -> jax.Array:
        """
        Construct the observation vector based on the current state.
        The observation is a 1D array of 50 bits for the board, plus 6 bits for
        hot, reset, catch, miss, plus, and minus bits.
        """
        # Initialize empty board (10x5 = 50 elements)
        board = jnp.zeros((self.rows, self.cols), dtype=jnp.int32)
        
        # Place ball on board if it's not in reset
        valid_ball_pos = ~self.in_reset & (self.ball_row >= 0) & (self.ball_row < self.rows)
        board = board.at[self.ball_row, self.ball_col].set(
            jnp.where(valid_ball_pos, 1, 0)
        )
        
        # Place paddle on board (always on bottom row)
        board = board.at[self.rows - 1, self.paddle_col].set(1)
        
        # Flatten board and append special bits
        flat_board = board.flatten()
        
        # Append special bits: [hot, reset, catch, miss, plus, minus]
        special_bits = jnp.array([
            self.is_hot, 
            self.in_reset, 
            self.catch_bit, 
            self.miss_bit, 
            self.plus_bit, 
            self.minus_bit
        ], dtype=jnp.int32)
        
        # Concatenate board and special bits
        observation = jnp.concatenate([flat_board, special_bits])
        
        return observation
    
    def step(self, action: int) -> Tuple[Any, jax.Array, jax.Array, Dict]:
        """
        Take a step in the environment based on the provided action.
        
        Args:
            action: 0 (left), 1 (stay), or 2 (right)
            
        Returns:
            Tuple of (new_state, observation, reward, info)
        """
        # Split RNG for different operations
        key, subkey = random.split(self.rng)
        
        # Update paddle position based on action
        # 0: left, 1: stay, 2: right
        paddle_col = jnp.clip(
            self.paddle_col + jnp.array([-1, 0, 1])[action],
            0,
            self.cols - 1
        )
        
        # Reset catch/miss bits each step if they were on
        catch_bit = jnp.array(False)
        miss_bit = jnp.array(False)
        
        # Initialize reward to zero
        reward = jnp.array(0.0)
        
        # Handle reward countdown for plus/minus bits
        plus_bit = self.plus_bit
        minus_bit = self.minus_bit
        reward_countdown = self.reward_countdown
        
        # If reward countdown is active and reaching zero, issue reward
        reward = jnp.where(
            (reward_countdown == 1) & plus_bit,
            1.0,
            reward
        )
        reward = jnp.where(
            (reward_countdown == 1) & minus_bit,
            -1.0,
            reward
        )
        
        # If countdown reaches 0, deactivate plus/minus bits and reset ball
        reset_after_reward = (reward_countdown == 1) & (plus_bit | minus_bit)
        plus_bit = jnp.where(reset_after_reward, False, plus_bit)
        minus_bit = jnp.where(reset_after_reward, False, minus_bit)
        
        # Decrement reward countdown if active
        reward_countdown = jnp.where(
            reward_countdown > 0,
            reward_countdown - 1,
            0
        )
        
        # Continue normal ball movement if not in special states
        ball_row = self.ball_row
        ball_col = self.ball_col
        in_reset = self.in_reset
        is_hot = self.is_hot
        
        # == HANDLE BALL IN RESET STATE ==
        key, subkey1 = random.split(key)
        will_enter_board = random.uniform(subkey1) < self.reset_prob
        
        # When in reset, ball may enter the board with probability reset_prob
        ball_enters = in_reset & will_enter_board
        
        # When ball enters, determine if board becomes hot
        key, subkey2 = random.split(key)
        becomes_hot = random.uniform(subkey2) < self.hot_prob
        is_hot = jnp.where(ball_enters, becomes_hot, is_hot)
        
        # Place ball in top row at random column when entering
        key, subkey3 = random.split(key)
        new_col = random.randint(subkey3, (), 0, self.cols)
        ball_row = jnp.where(ball_enters, jnp.array(0), ball_row)
        ball_col = jnp.where(ball_enters, new_col, ball_col)
        in_reset = jnp.where(ball_enters, False, in_reset)
        
        # == HANDLE BALL HITTING BOTTOM ROW ==
        ball_at_bottom = (ball_row == self.rows - 1) & ~in_reset
        
        # Determine if the ball is caught or missed
        is_caught = ball_at_bottom & (ball_col == paddle_col)
        is_missed = ball_at_bottom & (ball_col != paddle_col)
        
        # Update catch/miss bits
        catch_bit = jnp.where(is_caught, True, catch_bit)
        miss_bit = jnp.where(is_missed, True, miss_bit)
        
        # Activate plus/minus bits if hot, and set reward countdown
        key, subkey4 = random.split(key)
        duration = random.randint(
            subkey4, 
            (), 
            self.reward_indicator_duration_min,
            self.reward_indicator_duration_max + 1
        )
        
        plus_bit = jnp.where(is_caught & is_hot, True, plus_bit)
        minus_bit = jnp.where(is_missed & is_hot, True, minus_bit)
        
        # Set countdown when plus/minus bit is activated
        reward_countdown = jnp.where(
            (is_caught | is_missed) & is_hot & ~(plus_bit & minus_bit),
            duration,
            reward_countdown
        )
        
        # Reset ball after hitting bottom
        in_reset = jnp.where(ball_at_bottom, True, in_reset)
        ball_row = jnp.where(ball_at_bottom, jnp.array(-1), ball_row)
        
        # Advance ball if not in reset and not at bottom
        normal_fall = ~in_reset & ~ball_at_bottom & ~reset_after_reward
        ball_row = jnp.where(normal_fall, ball_row + 1, ball_row)
        
        # Reset after reward is delivered
        in_reset = jnp.where(reset_after_reward, True, in_reset)
        ball_row = jnp.where(reset_after_reward, jnp.array(-1), ball_row)
        
        # Construct new state
        new_state = eqx.tree_at(
            lambda t: (
                t.rng, t.ball_row, t.ball_col, t.paddle_col, t.in_reset, 
                t.is_hot, t.catch_bit, t.miss_bit, t.plus_bit, t.minus_bit,
                t.reward_countdown
            ),
            self,
            (
                key, ball_row, ball_col, paddle_col, in_reset, 
                is_hot, catch_bit, miss_bit, plus_bit, minus_bit,
                reward_countdown
            )
        )
        
        # Get observation
        observation = new_state._get_observation()
        
        # Info dictionary
        info = {
            "ball_row": ball_row,
            "ball_col": ball_col,
            "paddle_col": paddle_col,
            "in_reset": in_reset,
            "is_hot": is_hot,
            "catch_bit": catch_bit,
            "miss_bit": miss_bit,
            "plus_bit": plus_bit,
            "minus_bit": minus_bit,
            "reward_countdown": reward_countdown,
        }
        
        return new_state, observation, reward, info
    
    def reset(self, key: Optional[random.PRNGKey] = None) -> Tuple[Any, jax.Array]:
        """Reset the environment to an initial state."""
        if key is None:
            key, subkey = random.split(self.rng)
        else:
            key, subkey = random.split(key)
        
        # Create a fresh environment with the same static parameters
        new_env = CatchEnvironment(
            rows=self.rows,
            cols=self.cols,
            hot_prob=self.hot_prob,
            reset_prob=self.reset_prob,
            reward_indicator_duration_min=self.reward_indicator_duration_min,
            reward_indicator_duration_max=self.reward_indicator_duration_max,
            seed=random.randint(subkey, (), 0, 1000000000).item()
        )
        
        # Get observation
        observation = new_env._get_observation()
        
        return new_env, observation
    
    @property
    def observation_space_size(self) -> int:
        """Return the size of the observation space."""
        return self.rows * self.cols + 6  # board + 6 special bits
    
    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return 3  # left, stay, right


# Sanity check
def main():
    """Run a simple sanity check on the environment."""
    import matplotlib.pyplot as plt
    
    # Create environment
    env = CatchEnvironment(seed=42)
    
    # Reset the environment
    env, obs = env.reset()
    
    # Run a few steps
    total_reward = 0
    n_steps = 100
    
    # Create a function to print the board
    def print_board(obs, env):
        """Print the current state of the board."""
        board = obs[:env.rows * env.cols].reshape(env.rows, env.cols)
        special_bits = obs[env.rows * env.cols:]
        
        plt.figure(figsize=(5, 10))
        plt.imshow(board, cmap='Blues')
        
        # Add text for special bits
        special_bit_names = ['Hot', 'Reset', 'Catch', 'Miss', 'Plus', 'Minus']
        special_bit_status = ['ON' if bit else 'OFF' for bit in special_bits]
        
        for i, (name, status) in enumerate(zip(special_bit_names, special_bit_status)):
            plt.text(
                -1.5, env.rows + i * 0.5, 
                f"{name}: {status}", 
                fontsize=10
            )
        
        plt.title(f"Step: {step}")
        plt.tight_layout()
        plt.savefig(f"catch_step_{step}.png")
        plt.close()
    
    # Execute random actions
    for step in range(n_steps):
        # Choose a random action
        action = np.random.randint(0, 3)
        
        # Take a step
        env, obs, reward, info = env.step(action)
        total_reward += reward.item()
        
        # Print info for significant events
        if info["catch_bit"] or info["miss_bit"] or abs(reward) > 0:
            print(f"Step {step}:")
            print(f"  Action: {['left', 'stay', 'right'][action]}")
            print(f"  Reward: {reward.item()}")
            print(f"  Hot: {info['is_hot'].item()}")
            print(f"  Catch: {info['catch_bit'].item()}")
            print(f"  Miss: {info['miss_bit'].item()}")
            print(f"  Plus: {info['plus_bit'].item()}")
            print(f"  Minus: {info['minus_bit'].item()}")
            print(f"  Reward countdown: {info['reward_countdown'].item()}")
            print()
            
            # Visualize board
            print_board(obs, env)
    
    print(f"Total reward after {n_steps} steps: {total_reward}")
    
    # Test jitting the step function
    print("Testing JIT compilation...")
    import time
    
    def run_steps(env, n_steps=1000):
        """Run steps in the environment."""
        for _ in range(n_steps):
            action = 1  # Always stay
            env, _, _, _ = env.step(action)
        return env
    
    # Create a fresh environment
    env = CatchEnvironment(seed=123)
    
    # Time normal execution
    start = time.time()
    env = run_steps(env)
    normal_time = time.time() - start
    
    # Time jitted execution
    jitted_run_steps = jax.jit(
        lambda env: run_steps(env)
    )
    jitted_run_steps(env).block_until_ready()

    start = time.time()
    env = jitted_run_steps(env).block_until_ready()
    jitted_time = time.time() - start
    
    print(f"Normal execution time: {normal_time:.4f}s")
    print(f"Jitted execution time: {jitted_time:.4f}s")
    print(f"Speedup: {normal_time / jitted_time:.2f}x")


if __name__ == "__main__":
    main()
