import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import lee_algorithm, conways_game_of_life


# Test Lee's algorithm
def test_lee_algorithm():
    # Generate a random matrix and test Lee's algorithm
    matrix = np.random.randint(2, size=(5, 5)).tolist()
    start = (0, 0)
    end = (4, 4)
    matrix[0][0] = 2
    matrix[4][4] = 2

    shortest_path = lee_algorithm(matrix, start, end)

    if shortest_path:
        print("Shortest path exists between %s and %s:" % (start, end))
        print(shortest_path)

        # create final matrix with the path
        final_matrix = np.zeros((5, 5))
        for i in range(len(shortest_path)):
            final_matrix[shortest_path[i][0]][shortest_path[i][1]] = 2

        # plot matrix and final matrix
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.subplot(121)
        ax = sns.heatmap(
            matrix,
            annot=True,
            cmap="inferno",
            linewidths=0.5,
            linecolor="black",
            cbar=False,
        )
        plt.title("Matrix")
        plt.subplot(122)
        ax = sns.heatmap(
            final_matrix,
            annot=True,
            cmap="inferno",
            linewidths=0.5,
            linecolor="black",
            cbar=False,
        )
        plt.title("Shortest Path")
        plt.tight_layout()

        # Save figure
        plt.savefig("./output/test_lee_algorithm.png", dpi=300)

        # plt.show()

        return 1

    else:
        print("There is no path between %s and %s" % (start, end))
        return 0


# Test Conway's Game of Life
def test_conways_game_of_life():
    # Generate a random matrix and apply Conway's Game of Life rules
    matrix = np.random.randint(2, size=(5, 5))
    matrix[0][0] = 2
    matrix[4][4] = 2

    # Set up plots
    nrows = 4
    ncols = 5
    index = 1
    plt.rcParams["figure.figsize"] = [21.00, 10.50]
    plt.rcParams["figure.autolayout"] = True
    plt.subplots(nrows, ncols)

    # plot the initial matrix
    plt.subplot(nrows, ncols, index)
    ax = sns.heatmap(
        matrix,
        annot=True,
        cmap="inferno",
        linewidths=0.5,
        linecolor="black",
        cbar=False,
    )
    plt.title("Matrix")

    temp_matrix = matrix.copy()
    num_of_conway_iterations = 0
    while True and num_of_conway_iterations < 19:
        num_of_conway_iterations += 1
        matrix_after_conways = conways_game_of_life(temp_matrix)

        # plot matrix after conways
        plt.subplot(nrows, ncols, index + num_of_conway_iterations)
        plt.title("Conways iteration %s" % (num_of_conway_iterations))
        ax = sns.heatmap(
            matrix_after_conways,
            annot=True,
            cmap="inferno",
            linewidths=0.5,
            linecolor="black",
            cbar=False,
        )

        temp_matrix = matrix_after_conways.copy()

    plt.tight_layout()

    # Save figure
    plt.savefig("./output/test_conways_game_of_life.png", dpi=300)

    # plt.show()


if __name__ == "__main__":
    # Check if output folder exists
    if not os.path.exists("./output"):
        os.makedirs("./output")

    # Test Lee's algorithm
    # If there is no path keep generating random matrices until there is a path
    while test_lee_algorithm() == 0:
        pass

    # Test Conway's Game of Life
    test_conways_game_of_life()
