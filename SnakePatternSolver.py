from Game2048 import Game2048
import numpy as np


class SnakePatternSolver:
    def __init__(self, game):
        self.game = game
        self.preferred_moves = ['left', 'down']
        self.emergency_moves = ['up', 'right']

    def get_best_move(self):
        """Determine the best move according to the snake pattern strategy"""
        # Try preferred moves first (left and down)
        for move in self.preferred_moves:
            # Simulate the move
            test_grid = np.copy(self.game.grid)
            try:
                if move == 'left':
                    new_grid, _ = self.game.move_left(test_grid, self.game.score)
                else:  # move == 'down'
                    new_grid, _ = self.game.move_down(test_grid, self.game.score)

                # Check if the move changes the grid
                if not np.array_equal(test_grid, new_grid):
                    return move
            except:
                continue

        # If preferred moves don't work, try emergency moves
        for move in self.emergency_moves:
            test_grid = np.copy(self.game.grid)
            try:
                if move == 'up':
                    new_grid, _ = self.game.move_up(test_grid, self.game.score)
                else:  # move == 'right'
                    new_grid, _ = self.game.move_right(test_grid, self.game.score)

                # Check if the move changes the grid
                if not np.array_equal(test_grid, new_grid):
                    return move
            except:
                continue

        # If no valid move is found, return a random move as a last resort
        return np.random.choice(['left', 'down', 'up', 'right'])

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

        solver = SnakePatternSolver(game)
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


# Run 30 games and collect statistics
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    stats = run_multiple_games(30)