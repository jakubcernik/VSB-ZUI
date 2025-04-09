import numpy as np
import time
import random
from Game2048 import Game2048


class MonteCarloSolver:
    """
    A solver for the 2048 game using Monte Carlo method.
    Evaluates potential moves by running multiple random simulations and
    choosing the move with the best average outcome.
    """

    def __init__(self, game, num_simulations=100):
        """
        Initialize the solver with a game instance and simulation parameters

        Args:
            game: An instance of Game2048
            num_simulations: Number of random simulations per potential move
        """
        self.game = game
        self.num_simulations = num_simulations
        self.move_dict = {0: "up", 1: "right", 2: "down", 3: "left"}

    def _run_simulation(self, grid, score):
        """Run a random simulation from the given state until game over"""
        temp_game = Game2048()
        temp_game.grid = grid.copy()
        temp_game.score = score

        # Play random moves until the game ends
        while True:
            try:
                move = random.choice([0, 1, 2, 3])
                temp_game.move(move)
            except RuntimeError:
                break
            except ValueError:
                continue

        return temp_game.score, np.max(temp_game.grid)

    def get_best_move(self):
        """Determine the best move using Monte Carlo simulations"""
        best_move = None
        best_average_score = float('-inf')

        # Try each possible move
        for move in range(4):  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
            temp_game = Game2048()
            temp_game.grid = self.game.grid.copy()
            temp_game.score = self.game.score

            try:
                temp_game.move(move)
                # Skip if move doesn't change the grid
                if np.array_equal(self.game.grid, temp_game.grid):
                    continue
            except:
                continue

            # Run multiple simulations from this state
            simulation_scores = []

            for _ in range(self.num_simulations):
                final_score, _ = self._run_simulation(temp_game.grid.copy(), temp_game.score)
                simulation_scores.append(final_score)

            # Calculate average score for this move
            average_score = np.mean(simulation_scores)

            if average_score > best_average_score:
                best_average_score = average_score
                best_move = move

        return best_move

    def run(self, max_moves=100000, verbose=True):
        """Run the solver on a game"""
        moves_count = 0
        start_time = time.time()

        while moves_count < max_moves:
            best_move = self.get_best_move()

            # If no valid moves, game is over
            if best_move is None:
                if verbose:
                    print(f"No valid moves left. Game over in {moves_count} moves.")
                break

            try:
                self.game.move(best_move)
                moves_count += 1

                if verbose and moves_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Move {moves_count}: Score = {self.game.score}, " +
                          f"Max Tile = {np.max(self.game.grid)}, " +
                          f"Time per move: {elapsed / moves_count:.3f}s")

            except RuntimeError as inst:
                if str(inst) == "GO":
                    if verbose:
                        print(f"GAME OVER in {moves_count} moves")
                elif str(inst) == "WIN":
                    if verbose:
                        print(f"WIN in {moves_count} moves")
                break

        max_tile = np.max(self.game.grid)
        if verbose:
            print(f"Final Score: {self.game.score}, Moves: {moves_count}, Max Tile: {max_tile}")

        return self.game.score, moves_count, max_tile


def run_multiple_games(num_games=10, num_simulations=100):
    """Run multiple games and collect statistics"""
    scores = []
    moves_list = []
    max_tiles = []
    wins = 0
    start_time = time.time()

    for i in range(num_games):
        game = Game2048()
        game.new_game()

        solver = MonteCarloSolver(game, num_simulations=num_simulations)
        score, moves, max_tile = solver.run(max_moves=100000, verbose=True)
        game_won = max_tile >= 2048

        if game_won:
            wins += 1

        scores.append(score)
        moves_list.append(moves)
        max_tiles.append(max_tile)

        win_status = "WON" if game_won else "LOST"
        elapsed = time.time() - start_time
        avg_game_time = elapsed / (i + 1)
        est_remaining = (num_games - i - 1) * avg_game_time

        print(f"Game {i + 1}: Score = {score}, Moves = {moves}, Max Tile = {max_tile}, " +
              f"Status = {win_status}, Est remaining: {est_remaining / 60:.1f} min")

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

    # Adjust number of simulations based on performance needs
    # More simulations = better decisions but slower execution
    run_multiple_games(num_games=3, num_simulations=100)