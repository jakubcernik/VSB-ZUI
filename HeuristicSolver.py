import numpy as np
import time
import random
from Game2048 import Game2048

class HeuristicSolver:
    """
    A solver for the 2048 game using only heuristic evaluation.
    Evaluates each possible move directly and chooses the one with the highest score.
    """
    def __init__(self, game):
        """
        Initialize the solver with a game instance

        Args:
            game: An instance of Game2048
        """
        self.game = game
        self.move_dict = {0: "up", 1: "right", 2: "down", 3: "left"}

    def _evaluate_grid(self, grid):
        """
        Heuristic evaluation function for a game state.

        Args:
            grid: 2D numpy array representing the game grid

        Returns:
            float: Score for the current grid state
        """
        # Váhy pro různé heuristické komponenty
        monotonicity_weight = 1.0
        smoothness_weight = 0.1
        empty_weight = 2.7
        max_value_weight = 1.0

        # Počet prázdných buněk
        empty_cells = np.count_nonzero(grid == 0)

        # Maximální hodnota
        max_value = np.max(grid)

        # Výpočet plynulosti (penalizace za rozdíly mezi sousedními dlaždicemi)
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if j < 3:  # Kontrola horizontálně sousedících
                    if grid[i, j] > 0 and grid[i, j+1] > 0:
                        smoothness -= abs(grid[i, j] - grid[i, j+1])
                if i < 3:  # Kontrola vertikálně sousedících
                    if grid[i, j] > 0 and grid[i+1, j] > 0:
                        smoothness -= abs(grid[i, j] - grid[i+1, j])

        # Výpočet monotónnosti
        monotonicity = 0

        # Kontrola monotónnosti ve všech směrech
        for i in range(4):
            # Horizontální řádky
            current_row = grid[i, :]
            monotonicity += self._calculate_direction_monotonicity(current_row)

            # Vertikální sloupce
            current_col = grid[:, i]
            monotonicity += self._calculate_direction_monotonicity(current_col)

        # Kombinace všech heuristik
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
        # Kontrola rostoucí posloupnosti zleva doprava/shora dolů
        increasing = 0
        for i in range(len(arr)-1):
            if arr[i] > 0 and arr[i+1] > 0:
                if arr[i] <= arr[i+1]:
                    increasing += 1
                else:
                    increasing -= 2

        # Kontrola klesající posloupnosti zleva doprava/shora dolů
        decreasing = 0
        for i in range(len(arr)-1):
            if arr[i] > 0 and arr[i+1] > 0:
                if arr[i] >= arr[i+1]:
                    decreasing += 1
                else:
                    decreasing -= 2

        # Vrátíme lepší z obou směrů
        return max(increasing, decreasing)

    def get_best_move(self):
        """
        Get the best move for the current game state based on heuristic evaluation

        Returns:
            str: Best move ("up", "right", "down", "left")
        """
        best_move = None
        best_score = float('-inf')
        directions = ["up", "right", "down", "left"]

        # Zkusíme všechny možné tahy
        for move in directions:
            # Vytvoříme kopii mřížky pro simulaci
            test_grid = np.copy(self.game.grid)
            try:
                # Zkusíme provést tah
                if move == "up":
                    new_grid, _ = self.game.move_up(test_grid, self.game.score)
                elif move == "right":
                    new_grid, _ = self.game.move_right(test_grid, self.game.score)
                elif move == "down":
                    new_grid, _ = self.game.move_down(test_grid, self.game.score)
                elif move == "left":
                    new_grid, _ = self.game.move_left(test_grid, self.game.score)

                # Ověříme, zda se mřížka změnila
                if not np.array_equal(test_grid, new_grid):
                    # Vyhodnotíme stav po tahu
                    score = self._evaluate_grid(new_grid)

                    if score > best_score:
                        best_score = score
                        best_move = move
            except Exception as e:
                # Neplatný tah
                continue

        return best_move

    def run(self, max_moves=100000, verbose=True):
        """
        Run the solver on a game

        Args:
            max_moves: Maximum number of moves to make
            verbose: Whether to print progress information

        Returns:
            tuple: (final_score, moves_count, max_tile)
        """
        moves_count = 0
        start_time = time.time()
        max_tile = 0

        while moves_count < max_moves:
            best_move = self.get_best_move()

            # Pokud není žádný platný tah, hra končí
            if best_move is None:
                if verbose:
                    print(f"No valid moves left. Game over in {moves_count} moves.")
                break

            try:
                self.game.move(best_move)
                moves_count += 1
                max_tile = max(max_tile, np.max(self.game.grid))

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


def run_multiple_games(num_games=10):
    """
    Run multiple games with the Heuristic solver and collect statistics

    Args:
        num_games: Number of games to play

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

        solver = HeuristicSolver(game)
        score, moves, max_tile = solver.run(max_moves=100000, verbose=True)
        game_won = max_tile >= 2048  # Kontrola výhry

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

    # Výpočet statistik
    win_rate = (wins / num_games) * 100
    avg_score = np.mean(scores)
    avg_moves = np.mean(moves_list)
    max_score = np.max(scores)
    min_score = np.min(scores)
    max_moves = np.max(moves_list)
    min_moves = np.min(moves_list)
    total_time = time.time() - start_time

    # Počítání výskytů maximálních dlaždic
    tile_counts = {}
    for tile in max_tiles:
        if tile in tile_counts:
            tile_counts[tile] += 1
        else:
            tile_counts[tile] = 1

    # Výpis statistik
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
    # Nastavení seedu pro reprodukovatelnost
    np.random.seed(42)
    random.seed(42)

    run_multiple_games(num_games=1000)