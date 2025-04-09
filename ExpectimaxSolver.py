import numpy as np
import time
import random
from Game2048 import Game2048

class ExpectimaxSolver:
    """
    A solver for the 2048 game using the expectimax algorithm.
    This is a decision-making algorithm that considers all possible outcomes,
    including the stochastic placement of new tiles.
    """
    def __init__(self, game, max_depth=3):
        """
        Initialize the solver with a game instance and search depth

        Args:
            game: An instance of Game2048
            max_depth: Maximum depth for the expectimax search tree
        """
        self.game = game
        self.max_depth = max_depth
        self.move_dict = {0: "up", 1: "right", 2: "down", 3: "left"}

    def _evaluate_grid(self, grid):
        """
        Heuristic evaluation function for a game state.

        Args:
            grid: 2D numpy array representing the game grid

        Returns:
            float: Score for the current grid state
        """
        # Weight matrices for different heuristic components
        # Monotonicity - encouraging tiles to be in ascending/descending order
        monotonicity_weight = 1.0

        # Smoothness - encouraging adjacent tiles to have similar values
        smoothness_weight = 0.1

        # Empty cells - more empty cells is better
        empty_weight = 2.7

        # Max value - higher max value is better
        max_value_weight = 1.0

        # Count empty cells
        empty_cells = np.count_nonzero(grid == 0)

        # Calculate max value
        max_value = np.max(grid)

        # Calculate smoothness (penalty for differences between adjacent tiles)
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if j < 3:  # Check horizontally adjacent
                    if grid[i, j] > 0 and grid[i, j+1] > 0:
                        smoothness -= abs(grid[i, j] - grid[i, j+1])
                if i < 3:  # Check vertically adjacent
                    if grid[i, j] > 0 and grid[i+1, j] > 0:
                        smoothness -= abs(grid[i, j] - grid[i+1, j])

        # Calculate monotonicity
        monotonicity = 0

        # Check monotonicity in all directions (both increasing and decreasing)
        for i in range(4):
            # Horizontal rows
            current_row = grid[i, :]
            monotonicity += self._calculate_direction_monotonicity(current_row)

            # Vertical columns
            current_col = grid[:, i]
            monotonicity += self._calculate_direction_monotonicity(current_col)

        # Combine all heuristics
        score = (monotonicity_weight * monotonicity +
                 smoothness_weight * smoothness +
                 empty_weight * empty_cells +
                 max_value_weight * max_value)

        return score

    def _calculate_direction_monotonicity(self, arr):
        """
        Calculate monotonicity for a single row or column.

        Args:
            arr: 1D array representing a row or column

        Returns:
            float: Monotonicity score (higher is better)
        """
        # Check left-to-right/top-to-bottom increasing
        increasing = 0
        for i in range(len(arr)-1):
            if arr[i] > 0 and arr[i+1] > 0:
                if arr[i] <= arr[i+1]:
                    increasing += 1
                else:
                    increasing -= 2

        # Check left-to-right/top-to-bottom decreasing
        decreasing = 0
        for i in range(len(arr)-1):
            if arr[i] > 0 and arr[i+1] > 0:
                if arr[i] >= arr[i+1]:
                    decreasing += 1
                else:
                    decreasing -= 2

        # Return the better of the two directions
        return max(increasing, decreasing)

    def expectimax(self, grid, depth, is_maximizing):
        """
        Expectimax algorithm implementation
        """
        # Base case: reached maximum depth or terminal state
        if depth == 0:
            return self._evaluate_grid(grid), None

        if is_maximizing:
            # Player's turn - try all four moves
            max_value = float('-inf')
            best_move = None

            for move in range(4):  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
                # Create a copy of the grid for simulation
                new_grid = grid.copy()

                # Create a temporary game instance for this simulation
                temp_game = Game2048()
                temp_game.grid = new_grid.copy()
                temp_game.score = 0  # Reset score for simulation

                try:
                    # Try to make the move on our temp game instance
                    temp_game.move(move)

                    # Check if the move actually changed the grid
                    if not np.array_equal(grid, temp_game.grid):
                        # Next level will be a chance node
                        current_value, _ = self.expectimax(temp_game.grid.copy(), depth - 1, False)

                        if current_value > max_value:
                            max_value = current_value
                            best_move = move
                except (ValueError, RuntimeError):
                    # Invalid move or game over
                    continue

            # If no valid moves, return a low score
            if best_move is None:
                return float('-inf'), None

            return max_value, best_move

        else:
            # Computer's turn - calculate expected value over all empty cells
            empty_cells = [(i, j) for i in range(4) for j in range(4) if grid[i, j] == 0]

            if not empty_cells:  # No empty cells
                return self._evaluate_grid(grid), None

            # Probability of getting a 2 (90%) or a 4 (10%)
            prob_2 = 0.9
            prob_4 = 0.1

            expected_value = 0

            # For each empty cell
            for i, j in empty_cells:
                # Try placing a 2
                grid_with_2 = grid.copy()
                grid_with_2[i, j] = 2
                value_with_2, _ = self.expectimax(grid_with_2, depth - 1, True)

                # Try placing a 4
                grid_with_4 = grid.copy()
                grid_with_4[i, j] = 4
                value_with_4, _ = self.expectimax(grid_with_4, depth - 1, True)

                # Calculate expected value for this cell
                cell_expected_value = prob_2 * value_with_2 + prob_4 * value_with_4

                # Add to total expected value (divided by number of empty cells)
                expected_value += cell_expected_value / len(empty_cells)

            return expected_value, None

    def get_best_move(self):
        """
        Get the best move for the current game state

        Returns:
            int: Best move (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        """
        _, best_move = self.expectimax(self.game.grid.copy(), self.max_depth, True)
        return best_move

    def run(self, max_moves=100000, verbose=True):
        """
        Run the solver on a game
        """
        moves_count = 0
        self.game.new_game()
        start_time = time.time()

        while moves_count < max_moves:
            best_move = self.get_best_move()

            # If no valid moves, game is over
            if best_move is None:
                if verbose:
                    print(f"No valid moves left. Game over in {moves_count} moves.")
                break

            # Check if the move is valid by simulating it first
            temp_game = Game2048()
            temp_game.grid = self.game.grid.copy()
            try:
                temp_game.move(best_move)
                # If we get here, the move is valid

                # Now make the actual move
                self.game.move(best_move)
                moves_count += 1

                if verbose and moves_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Move {moves_count}: Score = {self.game.score}, " +
                          f"Max Tile = {np.max(self.game.grid)}, " +
                          f"Time per move: {elapsed / moves_count:.3f}s")

            except (ValueError, RuntimeError) as inst:
                if isinstance(inst, RuntimeError):
                    if str(inst) == "GO":
                        if verbose:
                            print(f"GAME OVER in {moves_count} moves")
                    elif str(inst) == "WIN":
                        if verbose:
                            print(f"WIN in {moves_count} moves")
                elif isinstance(inst, ValueError):
                    if verbose:
                        print(f"Invalid move detected: {self.move_dict[best_move]}")
                break

        max_tile = np.max(self.game.grid)
        if verbose:
            print(f"Final Score: {self.game.score}, Moves: {moves_count}, Max Tile: {max_tile}")

        return self.game.score, moves_count, max_tile


def run_multiple_games(num_games=10, max_depth=3):
    """
    Run multiple games with the Expectimax solver and collect statistics

    Args:
        num_games: Number of games to play
        max_depth: Maximum depth for the expectimax search

    Returns:
        dict: Statistics about the games
    """
    scores = []
    moves_list = []
    max_tiles = []
    wins = 0
    start_time = time.time()

    for i in range(num_games):
        game = Game2048()
        game.new_game()

        solver = ExpectimaxSolver(game, max_depth=max_depth)
        score, moves, max_tile = solver.run(max_moves=100000, verbose=True)
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
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # You can adjust max_depth based on performance needs
    # Higher depth = better decisions but slower execution
    run_multiple_games(num_games=5, max_depth=3)