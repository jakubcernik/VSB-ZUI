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

        while moves_count < max_moves:
            best_move = self.get_best_move()
            try:
                self.game.move(best_move)
                moves_count += 1

                # Optional: Print progress every N moves
                if moves_count % 100 == 0:
                    print(f"Move {moves_count}, Score: {self.game.score}")

            except RuntimeError as inst:
                if str(inst) == "GO":
                    print(f"GAME OVER in {moves_count} moves")
                elif str(inst) == "WIN":
                    print(f"WIN in {moves_count} moves")
                break

        print(self.game.grid)
        print(f"Final score: {self.game.score}")
        return self.game.score, moves_count


# Run the solver
if __name__ == "__main__":
    game = Game2048()
    game.new_game()

    solver = SnakePatternSolver(game)
    final_score, moves = solver.run()