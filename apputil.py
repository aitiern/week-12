import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(current_board):
    """
    Execute one step of Conway's Game of Life on a binary NumPy array.
    Non-wrapping boundaries: cells outside the board are treated as dead.
    """
    board = current_board.astype(int)

    # Pad with a border of zeros so edges do NOT wrap
    padded = np.pad(board, pad_width=1, mode="constant", constant_values=0)

    # Count neighbors from the 8 surrounding shifts on the padded board
    neighbors = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] +                   padded[1:-1, 2:] +
        padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:]
    )

    alive = board == 1
    survive = alive & ((neighbors == 2) | (neighbors == 3))
    born = (~alive) & (neighbors == 3)

    return (survive | born).astype(int)



def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)


def play_random_game_recursive(n_steps=5):
    """
    Generate a random 10x10 board and recursively apply update_board
    n_steps times. Returns the final board.
    """
    board = np.random.randint(2, size=(10, 10))

    def helper(current_board, steps_left):
        if steps_left == 0:
            return current_board
        return helper(update_board(current_board), steps_left - 1)

    return helper(board, n_steps)





def knapsack(W, weights, values, full_table=False):
    # Get the number of items based on the values list
    n = len(values)
    # Initialize a (n+1) x (W+1) table filled with 0s for DP
    table = [[0 for x in range(W + 1)] for x in range(n + 1)]

    # Loop over the rows: i represents "using the first i items"
    for i in range(n + 1):
        # Loop over the columns: j represents current capacity from 0..W
        for j in range(W + 1):

            # Base case: if no items or zero capacity, max value is 0
            if i == 0 or j == 0:
                table[i][j] = 0

            # Otherwise, consider the i-th item (index i-1) if it fits
            elif weights[i-1] <= j:
                # Value of taking the i-th item
                a_1 = values[i-1]
                # Remaining capacity after taking the i-th item
                diff = j - weights[i-1]
                # Best value achievable with remaining capacity and previous items
                a_2 = table[i-1][diff]
                # Total value if we include the i-th item
                a = a_1 + a_2
                # Value if we do NOT include the i-th item (just use previous row)
                b = table[i-1][j]
                # Choose the better of including or excluding the item
                table[i][j] = max(a, b)

            # If the i-th item does not fit, we must exclude it
            else:
                # So the best we can do is the same as with the previous items
                table[i][j] = table[i-1][j]

    # If full_table is True, return the entire DP table
    if full_table:
        return table

    # Otherwise, return the optimal value for n items and capacity W
    return table[n][W]
