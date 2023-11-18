import numpy as np
import sys
import time
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque
import os


"""
Utils.py contains the following functions:
    - lee_algorithm
    - get_neighbors
    - get_shortest_path
    - conways_game_of_life
    - generate_matrices
    - plot_matrices
    - generate_dataset
    - remove_duplicates
    - load_datasets
    - generate_matrices_without_path
    - generate_dataset_without_path
"""


# Implementing the Lee algorithm for finding the shortest path in a matrix
def lee_algorithm(matrix, start, end):
    # Using a deque for efficient pop and append operations
    queue = deque()
    # Set to keep track of visited nodes
    visited = set()
    # Dictionary to keep track of distances from the start
    distance = {start: 0}
    # Dictionary to store the previous node for each visited node
    previous = {}

    # Add the start node to the queue and mark it as visited
    queue.append(start)
    visited.add(start)

    # Continue until the queue is not empty
    while queue:
        # Pop the leftmost node from the queue
        node = queue.popleft()

        # Explore the neighboring nodes
        for neighbor in get_neighbors(matrix, node):
            # If the neighbor is not visited, update its distance and previous node, then add it to the queue
            if neighbor not in visited:
                visited.add(neighbor)
                distance[neighbor] = distance[node] + 1
                previous[neighbor] = node
                queue.append(neighbor)

            # If the neighbor is the end node, return the shortest path
            if neighbor == end:
                return get_shortest_path(previous, start, end)

    # If no path is found, return None
    return None


# Retrieve the neighboring nodes of a given node in the matrix
def get_neighbors(matrix, node):
    neighbors = []
    row, col = node

    # Check the top neighbor
    if row > 0 and matrix[row - 1][col] != 0:
        neighbors.append((row - 1, col))

    # Check the bottom neighbor
    if row < len(matrix) - 1 and matrix[row + 1][col] != 0:
        neighbors.append((row + 1, col))

    # Check the left neighbor
    if col > 0 and matrix[row][col - 1] != 0:
        neighbors.append((row, col - 1))

    # Check the right neighbor
    if col < len(matrix[0]) - 1 and matrix[row][col + 1] != 0:
        neighbors.append((row, col + 1))

    return neighbors


# Retrieve the shortest path based on the previously stored information
def get_shortest_path(prev, start, end):
    path = []
    node = end

    # Trace back the path from the end node to the start node
    while node != start:
        path.append(node)
        node = prev[node]

    path.append(start)
    path.reverse()

    return path


# Implement Conway's Game of Life rules for the given matrix
def conways_game_of_life(matrix):
    # Create a copy of the matrix for updating without altering the original
    N, M = matrix.shape
    updated_matrix = np.copy(matrix)

    # Iterate through each cell in the matrix
    for i in range(N):
        for j in range(M):
            # Compute the sum of the 8 neighbors
            total = (
                matrix[i, (j - 1) % M]
                + matrix[i, (j + 1) % M]
                + matrix[(i - 1) % N, j]
                + matrix[(i + 1) % N, j]
                + matrix[(i - 1) % N, (j - 1) % M]
                + matrix[(i - 1) % N, (j + 1) % M]
                + matrix[(i + 1) % N, (j - 1) % M]
                + matrix[(i + 1) % N, (j + 1) % M]
            )

            # Apply Conway's rules for cell survival or death
            if matrix[i, j] == 1:
                if (total < 2) or (total > 3):
                    updated_matrix[i, j] = 0
            else:  # matrix[i, j] == 0
                if total == 3:
                    updated_matrix[i, j] = 1

    return updated_matrix


# Generate Random Matrix and apply Conway's Game of Life until:
# a) end, start exists && b) Lee algorithm returns a path
# Return matrix, matrix_after_conways, matrix_with_path
def generate_matrices(N, M):
    start = (0, 0)
    end = (N - 1, M - 1)

    # Loop until the conditions are met
    while True:
        print("\nGenerating random matrix...")
        matrix = np.random.randint(2, size=(N, M))

        num_of_conway_iterations = 0
        temp_matrix = matrix.copy()

        # Apply Conway's Game of Life rules
        print("Applying Conway's Game of Life...")
        while True and num_of_conway_iterations < 100:
            # progress bar :)
            sys.stdout.write("\r")
            sys.stdout.write(
                "[%-100s] %d%%"
                % ("=" * num_of_conway_iterations, 1 * num_of_conway_iterations)
            )
            sys.stdout.flush()
            time.sleep(0.05)

            num_of_conway_iterations += 1
            matrix_after_conways = conways_game_of_life(temp_matrix)
            temp_matrix = matrix_after_conways.copy()

            # Check if end and start exist in matrix_after_conways
            if (
                matrix_after_conways[start[0]][start[1]] == 0
                or matrix_after_conways[end[0]][end[1]] == 0
            ):
                continue

            # Check if the shortest path exists
            shortest_path = lee_algorithm(
                matrix_after_conways, start, end
            )  # None or list of tuples (path)
            if shortest_path:  # if path exists
                print("\nShortest path exists between %s and %s:" % (start, end))
                print(shortest_path)

                # create final matrix with the path
                matrix_with_path = np.zeros((N, M))
                for i in range(len(shortest_path)):
                    matrix_with_path[shortest_path[i][0]][shortest_path[i][1]] = 2

                return (
                    matrix,
                    matrix_after_conways,
                    num_of_conway_iterations,
                    matrix_with_path,
                )

            # If the matrix is OFF, then there is no path
            if sum(sum(matrix_after_conways)) == 0:
                print(
                    "\nCells are all zeros after %s Conway's iterations."
                    % (num_of_conway_iterations)
                )
                print("Need to generate a new random matrix.")
                break


# plot generated matrices
def plot_matrices(
    matrix, matrix_after_conways, iteration, matrix_with_path, N, M, img_name, img_path
):
    start = (0, 0)
    end = (N - 1, M - 1)

    # Mark start and end points on each matrix
    matrix[start[0]][start[1]] = 2
    matrix[end[0]][end[1]] = 2
    matrix_after_conways[start[0]][start[1]] = 2
    matrix_after_conways[end[0]][end[1]] = 2
    matrix_with_path[start[0]][start[1]] = 2
    matrix_with_path[end[0]][end[1]] = 2

    # plot matrix
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.subplot(131)
    ax = sns.heatmap(
        matrix,
        annot=True,
        cmap="inferno",
        linewidths=0.5,
        linecolor="black",
        cbar=False,
    )
    plt.title("Matrix")
    plt.subplot(132)
    ax = sns.heatmap(
        matrix_after_conways,
        annot=True,
        cmap="inferno",
        linewidths=0.5,
        linecolor="black",
        cbar=False,
    )
    plt.title("Conways with %s iterations" % (iteration))
    plt.subplot(133)
    ax = sns.heatmap(
        matrix_with_path,
        annot=True,
        cmap="inferno",
        linewidths=0.5,
        linecolor="black",
        cbar=False,
    )
    plt.title("Shortest Path")
    plt.tight_layout()

    # save image locally
    # cwd = os.getcwd()
    plt.savefig("./output/" + img_path + "/" + img_name)
    plt.close()


# Generate dataset of matrices
def generate_dataset(num_of_matrices, N, M, test_name):
    X_input = []  # initialize empty list to store the input matrices
    X_after_conways = []  # initialize empty list to store the matrices after conways
    y_target = (
        []
    )  # initialize empty list to store the output matrices (with the path on them)

    # check if directories exist, if not create them
    # cwd = os.getcwd()
    if not os.path.exists("./output/" + test_name):
        os.mkdir("./output/" + test_name)
        os.mkdir("./output/" + test_name + "/matrices")

    for i in range(num_of_matrices):
        print("\nGenerating matrix %s..." % (i))
        image_name = (
            "matrix_" + str(i) + "_dimensions_" + str(N) + "X" + str(M) + ".png"
        )
        matrix, matrix_after_conways, iteration, matrix_with_path = generate_matrices(
            N, M
        )
        plot_matrices(
            matrix,
            matrix_after_conways,
            iteration,
            matrix_with_path,
            N,
            M,
            image_name,
            test_name,
        )

        # append matrices to lists
        X_input.append(matrix)
        X_after_conways.append(matrix_after_conways)
        y_target.append(matrix_with_path)

    # convert final lists to numpy arrays
    X_input = np.array(X_input)
    X_after_conways = np.array(X_after_conways)
    y_target = np.array(y_target)

    # save numpy arrays to files
    np.save("./output/" + test_name + "/matrices/X_input.npy", X_input)
    np.save("./output/" + test_name + "/matrices/X_after_conways.npy", X_after_conways)
    np.save("./output/" + test_name + "/matrices/y_target.npy", y_target)

    print("\n\nMatrices saved to files successfully!")


# Function to remove duplicates from dataset
def remove_duplicates(matrix, matrix_after_conways, matrix_with_path):
    # Create copies of matrices
    matrix_clean = np.copy(matrix)
    matrix_after_conways_clean = np.copy(matrix_after_conways)
    matrix_with_path_clean = np.copy(matrix_with_path)

    while True:
        # List to store the indexes of duplicates
        list_of_duplicates_indexes = []

        # Iterate through all matrices and remove duplicates
        for i in range(matrix_clean.shape[0]):
            for j in range(i + 1, matrix_clean.shape[0]):
                # Check if the index is out of the range of the matrix
                if j >= matrix_clean.shape[0]:
                    break

                # Check if duplicates are found
                if np.all(
                    matrix_after_conways_clean[i] == matrix_after_conways_clean[j]
                ):
                    # Add the indexes of duplicates to the list
                    list_of_duplicates_indexes.append(j)

                    # Remove duplicates
                    matrix_clean = np.delete(matrix_clean, j, axis=0)
                    matrix_after_conways_clean = np.delete(
                        matrix_after_conways_clean, j, axis=0
                    )
                    matrix_with_path_clean = np.delete(
                        matrix_with_path_clean, j, axis=0
                    )

        # If no duplicates are found, exit the loop
        if len(list_of_duplicates_indexes) == 0:
            break

    # Return the matrices and the list of duplicate indexes
    return matrix_clean, matrix_after_conways_clean, matrix_with_path_clean


# Load datasets from the output directory
def load_datasets(name):
    # Load the first dataset
    dataset_name = name + "_1"
    matrix = np.load("./output/" + dataset_name + "/matrices/X_input.npy")
    matrix_after_conways = np.load(
        "./output/" + dataset_name + "/matrices/X_after_conways.npy"
    )
    matrix_with_path = np.load("./output/" + dataset_name + "/matrices/y_target.npy")

    # Load all datasets from the output directory that start with "dataset_"
    with os.scandir("./output") as entries:
        for entry in entries:
            if (
                entry.is_dir()
                and entry.name.startswith("dataset_")
                and entry.name != dataset_name
            ):
                matrix = np.concatenate(
                    (
                        matrix,
                        np.load("./output/" + entry.name + "/matrices/X_input.npy"),
                    ),
                    axis=0,
                )
                matrix_after_conways = np.concatenate(
                    (
                        matrix_after_conways,
                        np.load(
                            "./output/" + entry.name + "/matrices/X_after_conways.npy"
                        ),
                    ),
                    axis=0,
                )
                matrix_with_path = np.concatenate(
                    (
                        matrix_with_path,
                        np.load("./output/" + entry.name + "/matrices/y_target.npy"),
                    ),
                    axis=0,
                )

    # # Print the shapes of the matrices
    # print("matrix shape: ", matrix.shape)
    # print("matrix_after_conways shape: ", matrix_after_conways.shape)
    # print("matrix_with_path shape: ", matrix_with_path.shape)

    # Remove duplicates from the matrices
    (
        matrix_clean,
        matrix_after_conways_clean,
        matrix_with_path_clean,
    ) = remove_duplicates(matrix, matrix_after_conways, matrix_with_path)

    # # Print the shapes of the matrices after removing duplicates
    # print("matrix_clean shape: ", matrix_clean.shape)
    # print("matrix_after_conways_clean shape: ", matrix_after_conways_clean.shape)
    # print("matrix_with_path_clean shape: ", matrix_with_path_clean.shape)

    # # Save the matrices
    # np.save("./output/X_input_clean.npy", matrix_clean)
    # np.save("./output/X_after_conways_clean.npy", matrix_after_conways_clean)
    # np.save("./output/y_target_clean.npy", matrix_with_path_clean)

    # print("\n\nMatrices saved to files successfully!")

    return matrix_clean, matrix_after_conways_clean, matrix_with_path_clean


# Generate a Random Matrix and apply Conway's Game of Life
# until: a) end, start exists && b) Lee algorithm DOESN'T return a path
# Return matrix_without, matrix_after_conways_without, matrix_without_path
def generate_matrices_without_path(N, M):
    start = (0, 0)
    end = (N - 1, M - 1)

    while True:
        print("\nGenerating a random matrix...")
        matrix = np.random.randint(2, size=(N, M))

        num_of_conway_iterations = 0
        temp_matrix = matrix.copy()

        print("Applying Conway's Game of Life...")
        while True and num_of_conway_iterations < 100:
            # Progress bar :)
            sys.stdout.write("\r")
            sys.stdout.write(
                "[%-100s] %d%%"
                % ("=" * num_of_conway_iterations, 1 * num_of_conway_iterations)
            )
            sys.stdout.flush()
            time.sleep(0.05)

            num_of_conway_iterations += 1
            matrix_after_conways = conways_game_of_life(temp_matrix)

            temp_matrix = matrix_after_conways.copy()

            # Check if end and start exist in matrix_after_conways
            if (
                matrix_after_conways[start[0]][start[1]] == 0
                or matrix_after_conways[end[0]][end[1]] == 0
            ):
                # print("Start or end does not exist in matrix. Need to generate a new matrix.")
                continue

            shortest_path = lee_algorithm(
                matrix_after_conways, start, end
            )  # None or list of tuples (path)

            # If path DOESN'T exist
            if not shortest_path:
                print("\nShortest path doesn't exist between %s and %s:" % (start, end))

                # Create the final matrix without a path as a copy of matrix_after_conways
                matrix_without_path = matrix_after_conways.copy()

                return (
                    matrix,
                    matrix_after_conways,
                    num_of_conway_iterations,
                    matrix_without_path,
                )

            # If the matrix is OFF, then there is no path
            if sum(sum(matrix_after_conways)) == 0:
                print(
                    "\nCells are all zeros after %s conways iteration."
                    % (num_of_conway_iterations)
                )
                print("Need to generate a new random matrix.")
                break


# Generate a dataset of matrices without a path
def generate_dataset_without_path(num_of_matrices, N, M, test_name):
    X_input = []  # initialize an empty list to store the input matrices
    X_after_conways = []  # initialize an empty list to store the matrices after conways
    y_target = (
        []
    )  # initialize an empty list to store the output matrices (with the path on them)

    for i in range(num_of_matrices):
        print("\nGenerating matrix without a path %s..." % (i))
        image_name = (
            "matrix_without_path_"
            + str(i)
            + "_dimensions_"
            + str(N)
            + "X"
            + str(M)
            + ".png"
        )
        (
            matrix,
            matrix_after_conways,
            iteration,
            matrix_with_path,
        ) = generate_matrices_without_path(N, M)
        plot_matrices(
            matrix,
            matrix_after_conways,
            iteration,
            matrix_with_path,
            N,
            M,
            image_name,
            test_name,
        )

        # append matrices to lists
        X_input.append(matrix)
        X_after_conways.append(matrix_after_conways)
        y_target.append(matrix_with_path)

    # convert final lists to numpy arrays
    X_input = np.array(X_input)
    X_after_conways = np.array(X_after_conways)
    y_target = np.array(y_target)

    # save numpy arrays to files
    np.save("./output/" + test_name + "/matrices/X_input_without.npy", X_input)
    np.save(
        "./output/" + test_name + "/matrices/X_after_conways_without.npy",
        X_after_conways,
    )
    np.save("./output/" + test_name + "/matrices/y_target_without.npy", y_target)

    print("\n\nMatrices saved to files successfully!")

    return X_input, X_after_conways, y_target
