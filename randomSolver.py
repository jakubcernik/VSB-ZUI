from Game2048 import Game2048
import numpy as np

game = Game2048()
game.new_game()

#Random direction solver
grid = game.grid
score = game.score

for i in range(1000):
    direction = np.random.choice(('left','right','up','down'))
    try:
        game.move(direction)
        grid = game.grid
        score = game.score
    except RuntimeError as inst:
        if str(inst) == "GO":
            print("GAME OVER in ",(i+1)," moves")
        elif str(inst) == "WIN":
            print("WIN in ",(i+1)," moves")
        break
print(grid)
print("Final score:", score)