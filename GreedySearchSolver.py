from Game2048 import Game2048
import numpy as np


class GreedySearchSolver:
    def __init__(self, game):
        self.game = game
        self.directions = ['left', 'right', 'up', 'down']
        self.move_functions = {
            'left': self.game.move_left,
            'right': self.game.move_right,
            'up': self.game.move_up,
            'down': self.game.move_down
        }

    def evaluate_state(self, grid, score):
        """
        Evaluate a grid state based on:
        1. Score gained
        2. Number of empty cells
        3. Monotonicity of tiles (snake-like arrangement)
        """
        # Count empty cells
        empty_cells = np.count_nonzero(grid == 0)

        # Calculate a monotonicity score (higher values on one side)
        monotonicity_score = 0
        weight_matrix = np.array([
            [15, 14, 13, 12],
            [8, 9, 10, 11],
            [7, 6, 5, 4],
            [0, 1, 2, 3]
        ])

        monotonicity_score = np.sum(grid * weight_matrix)

        # Combine factors with weights
        evaluation = score + empty_cells * 10 + monotonicity_score / 100
        return evaluation

    def get_best_move(self):
        """Find the move that gives the best evaluation"""
        best_move = None
        best_evaluation = float('-inf')

        for direction in self.directions:
            test_grid = np.copy(self.game.grid)
            try:
                new_grid, new_score = self.move_functions[direction](test_grid, self.game.score)

                # Check if the move changes the grid
                if not np.array_equal(test_grid, new_grid):
                    evaluation = self.evaluate_state(new_grid, new_score)
                    if evaluation > best_evaluation:
                        best_evaluation = evaluation
                        best_move = direction
            except:
                continue

        # If no valid move found, return a random move
        if best_move is None:
            return np.random.choice(self.directions)

        return best_move

    def run(self, max_moves=100000):
        """Run the solver until game over or max moves reached"""
        moves_count = 0
        max_tile = 0
        game_won = False

        while moves_count < max_moves:
            best_move = self.get_best_move()
            try:
                self.game.move(best_move)
                moves_count += 1
                max_tile = max(max_tile, np.max(self.game.grid))

            except RuntimeError as inst:
                if str(inst) == "GO":
                    print(f"GAME OVER in {moves_count} moves")
                elif str(inst) == "WIN":
                    print(f"WIN in {moves_count} moves")
                    game_won = True
                break

        # Get the maximum tile value achieved
        max_tile = np.max(self.game.grid)
        return self.game.score, moves_count, max_tile


def run_multiple_games(num_games=30):
    """Run multiple games and collect statistics"""
    scores = []
    moves_list = []
    max_tiles = []
    wins = 0

    for i in range(num_games):
        game = Game2048()
        game.new_game()

        solver = GreedySearchSolver(game)
        score, moves, max_tile = solver.run(max_moves=100000)
        game_won = max_tile >= 2048  # Check if game was won

        if game_won:
            wins += 1

        scores.append(score)
        moves_list.append(moves)
        max_tiles.append(max_tile)

        win_status = "WON" if game_won else "LOST"
        print(f"Game {i + 1}: Score = {score}, Moves = {moves}, Max Tile = {max_tile}, Status = {win_status}")

    # Calculate statistics
    win_rate = (wins / num_games) * 100
    avg_score = np.mean(scores)
    avg_moves = np.mean(moves_list)
    max_score = np.max(scores)
    min_score = np.min(scores)
    max_moves = np.max(moves_list)
    min_moves = np.min(moves_list)

    # Count occurrences of each max tile value
    tile_counts = {}
    for tile in max_tiles:
        if tile in tile_counts:
            tile_counts[tile] += 1
        else:
            tile_counts[tile] = 1

    # Print statistics
    print("\n===== STATISTICS =====")
    print(f"Games played: {num_games}")
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


# Run games and collect statistics
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    stats = run_multiple_games(300)