from Game2048 import Game2048
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os
import time


class DQNSolver:
    def __init__(self, game, model_path=None):
        self.game = game
        self.directions = ['left', 'right', 'up', 'down']
        self.move_functions = {
            'left': self.game.move_left,
            'right': self.game.move_right,
            'up': self.game.move_up,
            'down': self.game.move_down
        }
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self._valid_moves_cache = None
        self._valid_moves_grid = None

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = keras.models.load_model(model_path)
            self.epsilon = self.epsilon_min  # Use minimum epsilon for pretrained models
        else:
            self.model = self._build_model()

    def _build_model(self):
        """Smaller, faster neural net for deep Q learning"""
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(4, 4)),
            keras.layers.Dense(64, activation='relu'),  # Smaller network
            keras.layers.Dense(4, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def _preprocess_state(self, grid):
        """Convert grid to features usable by the neural network"""
        # Log transform to reduce the range of values (vectorized)
        grid_log = np.zeros_like(grid, dtype=np.float32)
        mask = grid > 0
        grid_log[mask] = np.log2(grid[mask])
        return grid_log / 16.0  # Normalize

    def memorize(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        action_idx = self.directions.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def get_valid_moves(self):
        """Get list of valid moves for current game state - with caching"""
        # If cache exists and grid hasn't changed, use cached valid moves
        if self._valid_moves_cache is not None and np.array_equal(self._valid_moves_grid, self.game.grid):
            return self._valid_moves_cache

        valid_moves = []
        for direction in self.directions:
            test_grid = np.copy(self.game.grid)
            try:
                new_grid, _ = self.move_functions[direction](test_grid, self.game.score)
                if not np.array_equal(test_grid, new_grid):
                    valid_moves.append(direction)
            except:
                continue

        # Update cache
        self._valid_moves_cache = valid_moves
        self._valid_moves_grid = np.copy(self.game.grid)
        return valid_moves

    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose random valid move
            valid_moves = self.get_valid_moves()
            if not valid_moves:
                return np.random.choice(self.directions)
            return np.random.choice(valid_moves)

        # Exploitation: choose best predicted action
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)

        # Try actions in order of predicted value
        action_indices = np.argsort(-act_values[0])
        for action_idx in action_indices:
            direction = self.directions[action_idx]
            test_grid = np.copy(self.game.grid)
            try:
                new_grid, _ = self.move_functions[direction](test_grid, self.game.score)
                if not np.array_equal(test_grid, new_grid):
                    return direction
            except:
                continue

        # If no valid moves found, choose randomly
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return np.random.choice(self.directions)
        return np.random.choice(valid_moves)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Process predictions in batches
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Single batch prediction calls
        next_q_values = self.model.predict(next_states, verbose=0)
        target_q_values = self.model.predict(states, verbose=0)

        # Vectorized update
        targets = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)

        for i, action in enumerate(actions):
            target_q_values[i][action] = targets[i]

        # Single fit call with larger batch
        self.model.fit(states, target_q_values, epochs=1, verbose=0, batch_size=batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(self, old_score, new_score, old_grid, new_grid):
        """Calculate reward based on game state changes"""
        # Base reward from score change
        score_reward = (new_score - old_score) / 100.0

        # Reward for more empty cells
        old_empty = np.count_nonzero(old_grid == 0)
        new_empty = np.count_nonzero(new_grid == 0)
        empty_reward = (new_empty - old_empty) * 2.0

        # Reward for higher max tile
        max_tile_reward = 0
        new_max = np.max(new_grid)
        old_max = np.max(old_grid)
        if new_max > old_max:
            max_tile_reward = 10.0

        # Discourage moves that don't change the grid
        same_grid_penalty = -5.0 if np.array_equal(old_grid, new_grid) else 0.0

        return score_reward + empty_reward + max_tile_reward + same_grid_penalty

    def train(self, num_episodes=1000, batch_size=256, save_path="dqn_model.h5",
              save_interval=100, update_frequency=4):
        """Train the model through reinforcement learning"""
        scores = []
        max_tiles = []
        step_counter = 0
        start_time = time.time()

        for e in range(num_episodes):
            episode_start = time.time()
            self.game.new_game()
            state = self._preprocess_state(self.game.grid)
            done = False
            total_reward = 0
            moves_in_episode = 0

            # Play one game
            while not done:
                action = self.get_action(state)
                old_score = self.game.score
                old_grid = np.copy(self.game.grid)

                try:
                    self.game.move(action)
                    moves_in_episode += 1
                    next_state = self._preprocess_state(self.game.grid)
                    reward = self.calculate_reward(old_score, self.game.score, old_grid, self.game.grid)
                    total_reward += reward

                    # Add experience to memory
                    self.memorize(state, action, reward, next_state, False)
                    state = next_state

                    # Train less frequently for better speed
                    step_counter += 1
                    if step_counter % update_frequency == 0:
                        self.replay(batch_size)

                except RuntimeError as inst:
                    if str(inst) in ["GO", "WIN"]:
                        if str(inst) == "WIN":
                            reward = 100.0  # Big reward for winning
                        else:
                            reward = -50.0  # Penalty for game over

                        next_state = self._preprocess_state(self.game.grid)
                        self.memorize(state, action, reward, next_state, True)
                        self.replay(batch_size)
                        done = True

            # Record game results
            max_tile = np.max(self.game.grid)
            scores.append(self.game.score)
            max_tiles.append(max_tile)

            episode_time = time.time() - episode_start
            total_time = time.time() - start_time
            remaining = ((total_time / (e + 1)) * (num_episodes - e - 1)) / 60 if e > 0 else 0

            print(f"Episode: {e + 1}/{num_episodes}, Score: {self.game.score}, Max: {max_tile}, " +
                  f"ε: {self.epsilon:.3f}, Time: {episode_time:.1f}s, Moves: {moves_in_episode}, " +
                  f"Est. remaining: {remaining:.1f} min")

            # Save model periodically
            if (e + 1) % save_interval == 0 or e == num_episodes - 1:
                self.model.save(save_path)
                print(f"Model saved to {save_path}")

                # Print intermediate stats
                if e > 0:
                    recent_scores = scores[-save_interval:]
                    recent_max = max_tiles[-save_interval:]
                    print(f"Last {save_interval} games - Avg score: {np.mean(recent_scores):.1f}, " +
                          f"Avg max tile: {np.mean(recent_max):.1f}, Win rate: " +
                          f"{sum(t >= 2048 for t in recent_max) / save_interval * 100:.1f}%")

        # Final save
        self.model.save(save_path)
        print(f"Training completed in {(time.time() - start_time) / 60:.1f} minutes")
        return scores, max_tiles

    def run(self, max_moves=100000):
        """Run the trained agent on a game"""
        moves_count = 0
        max_tile = 0
        self.game.new_game()

        while moves_count < max_moves:
            state = self._preprocess_state(self.game.grid)
            best_move = self.get_action(state, training=False)

            try:
                self.game.move(best_move)
                moves_count += 1
                max_tile = max(max_tile, np.max(self.game.grid))
            except RuntimeError as inst:
                if str(inst) == "GO":
                    print(f"GAME OVER in {moves_count} moves")
                elif str(inst) == "WIN":
                    print(f"WIN in {moves_count} moves")
                break

        max_tile = np.max(self.game.grid)
        return self.game.score, moves_count, max_tile


def run_multiple_games(num_games=30, model_path="dqn_model.h5"):
    """Run multiple games with a trained DQN model and collect statistics"""
    scores = []
    moves_list = []
    max_tiles = []
    wins = 0
    start_time = time.time()

    for i in range(num_games):
        game = Game2048()
        game.new_game()

        solver = DQNSolver(game, model_path=model_path)
        score, moves, max_tile = solver.run(max_moves=100000)
        game_won = max_tile >= 2048  # Check if game was won

        if game_won:
            wins += 1

        scores.append(score)
        moves_list.append(moves)
        max_tiles.append(max_tile)

        win_status = "WON" if game_won else "LOST"
        elapsed = time.time() - start_time
        avg_game_time = elapsed / (i + 1)
        est_remaining = (num_games - i - 1) * avg_game_time

        print(f"Game {i + 1}: Score = {score}, Moves = {moves}, Max Tile = {max_tile}, Status = {win_status}, " +
              f"Est remaining: {est_remaining / 60:.1f} min")

    # Calculate statistics
    win_rate = (wins / num_games) * 100
    avg_score = np.mean(scores)
    avg_moves = np.mean(moves_list)
    max_score = np.max(scores)
    min_score = np.min(scores)
    max_moves = np.max(moves_list)
    min_moves = np.min(moves_list)
    total_time = time.time() - start_time

    # Count occurrences of each max tile value
    tile_counts = {}
    for tile in max_tiles:
        if tile in tile_counts:
            tile_counts[tile] += 1
        else:
            tile_counts[tile] = 1

    # Print statistics
    print("\n===== STATISTICS =====")
    print(f"Games played: {num_games} in {total_time / 60:.1f} minutes ({total_time / num_games:.1f} seconds/game)")
    print(f"Wins: {wins}/{num_games} ({win_rate:.1f}%)")
    print(f"Average score: {avg_score:.2f}")
    print(f"Average moves: {avg_moves:.2f}")
    print(f"Score range: {min_score} - {max_score}")
    print(f"Moves range: {min_moves} - {max_moves}")
    print("\nMax tile distribution:")
    for tile, count in sorted(tile_counts.items()):
        percentage = (count / num_games) * 100
        print(f"Tile {tile}: {count} games ({percentage:.1f}%)")

    return {
        'wins': wins,
        'win_rate': win_rate,
        'avg_score': avg_score,
        'avg_moves': avg_moves,
        'max_score': max_score,
        'min_score': min_score,
        'scores': scores,
        'moves': moves_list,
        'max_tiles': max_tiles,
        'tile_distribution': tile_counts
    }


if __name__ == "__main__":
    # Speed optimizations
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    # Detect if GPU is available and configure memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {len(gpus)} detected")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("No GPU detected, using CPU")

    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Train mode with faster settings
    train_model = True
    model_path = "dqn_model.h5"
    stats = run_multiple_games(30, model_path=model_path)