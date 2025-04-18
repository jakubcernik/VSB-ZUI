# imports
import numpy as np
import copy as cp


def check_win(grid):
    return 2048 in grid


class Game2048:
    def __init__(self):
        import numpy as np
        self.np = np
        self.score = 0
        self.grid = np.zeros((4, 4), dtype=int)

    # add other methods here (move, add_score, etc.)

    # adds given value to total score

    def add_score(self, val):
        self.score += val
        return self.score

    def move(self, direction):
        self.grid, self.score = self.play_2048(self.grid, direction, self.score)
        return self.grid, self.score

    # move the grid to the left and update score
    def move_left(self, grid, score):
        grid = cp.deepcopy(grid)
        for i in range(4):
            non_zero = [x for x in grid[i, :] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[i, :] = np.array(non_zero + zero)
            for j in range(3):
                if grid[i, j] == grid[i, j + 1]:
                    grid[i, j] *= 2
                    score = self.add_score(grid[i, j])
                    grid[i, j + 1] = 0
            non_zero = [x for x in grid[i, :] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[i, :] = np.array(non_zero + zero)
        return grid, score

    # move the grid to the right and update score
    def move_right(self, grid, score):
        grid = cp.deepcopy(grid)
        for i in range(4):
            non_zero = [x for x in grid[i, :] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[i, :] = np.array(zero + non_zero[::-1])
            for j in range(3, 0, -1):
                if grid[i, j] == grid[i, j - 1]:
                    grid[i, j] *= 2
                    score = self.add_score(grid[i, j])
                    grid[i, j - 1] = 0
            non_zero = [x for x in grid[i, :] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[i, :] = np.array(zero + non_zero[::-1])
        return grid, score

    # move the grid up and update score
    def move_up(self, grid, score):
        grid = cp.deepcopy(grid)
        for i in range(4):
            non_zero = [x for x in grid[:, i] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[:, i] = np.array(non_zero + zero)
            for j in range(3):
                if grid[j, i] == grid[j + 1, i]:
                    grid[j, i] *= 2
                    score = self.add_score(grid[j, i])
                    grid[j + 1, i] = 0
            non_zero = [x for x in grid[:, i] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[:, i] = np.array(non_zero + zero)
        return grid, score

    # move the grid down and update score
    def move_down(self, grid, score):
        grid = cp.deepcopy(grid)
        for i in range(4):
            non_zero = [x for x in grid[:, i] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[:, i] = np.array(zero + non_zero[::-1])
            for j in range(3, 0, -1):
                if grid[j, i] == grid[j - 1, i]:
                    grid[j, i] *= 2
                    score = self.add_score(grid[j, i])
                    grid[j - 1, i] = 0
            non_zero = [x for x in grid[:, i] if x != 0]
            zero = [0] * (4 - len(non_zero))
            grid[:, i] = np.array(zero + non_zero[::-1])
        return grid, score

    # generates new tile
    @staticmethod
    def add_new_number(grid):
        zero_indices = np.where(grid == 0)
        if len(zero_indices[0]) == 0:
            return False
        index = np.random.choice(len(zero_indices[0]))
        i, j = zero_indices[0][index], zero_indices[1][index]
        grid[i, j] = 2 if np.random.random() < 0.9 else 4
        return True

    # checks whether it is Game Over
    @staticmethod
    def check_game_over(grid):
        if not np.all(grid):
            return False

        for row in range(4):
            for col in range(4):
                if row != 3:
                    if grid[row, col] == grid[row + 1, col]:
                        return False
                if col != 3:
                    if grid[row, col] == grid[row, col + 1]:
                        return False

        return True

    # checks for potential win

    # move the grid in specified direction, check for win or lose
    # raises RuntimeError "GO" if the game is in GAME OVER state
    # raises RuntimeError "WIN" if the game is in WIN state
    def play_2048(self, grid, move, score):
        orig_grid = cp.deepcopy(grid)

        if self.check_game_over(grid):
            raise RuntimeError("GO")

        if move == 'left':
            grid, score = self.move_left(grid, score)
        elif move == 'right':
            grid, score = self.move_right(grid, score)
        elif move == 'up':
            grid, score = self.move_up(grid, score)
        elif move == 'down':
            grid, score = self.move_down(grid, score)
        else:
            raise ValueError("Invalid move")

        if check_win(grid):
            raise RuntimeError("WIN")

        # check whether the move was possible
        if not np.array_equal(grid, orig_grid):
            self.add_new_number(grid)
        return grid, score

    def new_game(self):
        self.score = 0
        self.grid = np.zeros((4, 4), dtype=int)
        self.add_new_number(self.grid)
        self.add_new_number(self.grid)
        return self.grid, self.score

    # print of the grid
    def print_grid(grid, score):
        print('Score: ', score)
        print("+----+----+----+----+")
        for i in range(4):
            line = "|"
            for j in range(4):
                if grid[i, j] == 0:
                    line += "    |"
                else:
                    line += "{:4d}|".format(grid[i, j])
            print(line)
            print("+----+----+----+----+")
