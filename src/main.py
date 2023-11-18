import random
from utils import load_datasets


# Global Model Parameters
PATH = "CNN_model"
ACTIVATION_FUNCTION = "sigmoid"
LOSS_FUNCTION = "mean_squared_error"
OPTIMIZER = "adam"
METRICS = ["accuracy"]
EPOCHS = 100


# Preprocess dataset (shuffle and split into training and test sets)
def preprocess(matrix, matrix_after_conways, matrix_with_path):
    # Shuffle the matrices
    def custom_random():
        # Define your custom random function logic here
        return random.random() * 0.5

    random_order = [i for i in range(len(matrix))]
    random.shuffle(random_order, custom_random)

    # Shuffle the matrices according to the randomly created list
    matrix[:] = matrix[random_order]
    matrix_after_conways[:] = matrix_after_conways[random_order]
    matrix_with_path[:] = matrix_with_path[random_order]

    # Split the dataset into a training set and test set (80%:20%)
    split_ratio = 0.8

    # Training set
    matrix_train = matrix[: int(len(matrix) * split_ratio)]
    matrix_after_conways_train = matrix_after_conways[
        : int(len(matrix_after_conways) * split_ratio)
    ]
    matrix_with_path_train = matrix_with_path[
        : int(len(matrix_with_path) * split_ratio)
    ]

    # Test set
    matrix_test = matrix[int(len(matrix) * split_ratio) :]
    matrix_after_conways_test = matrix_after_conways[
        int(len(matrix_after_conways) * split_ratio) :
    ]
    matrix_with_path_test = matrix_with_path[int(len(matrix_with_path) * split_ratio) :]

    return (
        matrix_train,
        matrix_after_conways_train,
        matrix_with_path_train,
        matrix_test,
        matrix_after_conways_test,
        matrix_with_path_test,
    )


if __name__ == "__main__":
    # load dataset
    matrix, matrix_after_conways, matrix_with_path = load_datasets("dataset")

    # print matrices shapes
    print("\nDataset before preprocessing:")
    print(" -Matrix shape: ", matrix.shape)
    print(" -Matrix after conways shape: ", matrix_after_conways.shape)
    print(" -Matrix with path shape: ", matrix_with_path.shape)

    # preprocess dataset
    (
        matrix_train,
        matrix_after_conways_train,
        matrix_with_path_train,
        matrix_test,
        matrix_after_conways_test,
        matrix_with_path_test,
    ) = preprocess(matrix, matrix_after_conways, matrix_with_path)

    # print matrices shapes
    print("\n\nDataset after preprocessing:")
    print(" -Matrix train shape: ", matrix_train.shape)
    print(" -Matrix after conways train shape: ", matrix_after_conways_train.shape)
    print(" -Matrix with path train shape: ", matrix_with_path_train.shape)
    print(" -Matrix test shape: ", matrix_test.shape)
    print(" -Matrix after conways test shape: ", matrix_after_conways_test.shape)
    print(" -Matrix with path test shape: ", matrix_with_path_test.shape)

    # TODO: Create model
