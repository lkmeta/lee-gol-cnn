import os
import random
from utils import load_datasets

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Reshape
from keras.optimizers import Adam
import tensorflow as tf
import seaborn as sns
from contextlib import redirect_stdout


# Global Model Parameters
OUTPUT_PATH = "output/"
CURRENT_RUN = "run_"
MODEL_NAME = "CNN_model"
ACTIVATION_FUNCTION = "sigmoid"
LOSS_FUNCTION = "mean_squared_error"
OPTIMIZER = "adam"
METRICS = ["accuracy"]
EPOCHS = 20
PATIENCE = 10

NUM_OF_MATRICES = 100
MATRIX_ROWS = 5
MATRIX_COLS = 5


# Function to find the last run of the model
def find_last_run():
    # Find the last run of the model
    last_run = 0
    for entry in os.scandir(OUTPUT_PATH):
        if entry.is_dir() and entry.name.startswith(CURRENT_RUN):
            last_run = max(last_run, int(entry.name.split("_")[1]))

    return last_run


# Preprocess dataset (shuffle and split into training and test sets (80%:20%))
def preprocess(matrix, matrix_after_conways, matrix_with_path):
    # Shuffle the matrices
    def custom_random():
        # Define your custom random function logic here
        return random.random() * 0.5

    random_order = [i for i in range(len(matrix))]
    # random.shuffle(random_order, custom_random)

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


class CNN:
    def __init__(
        self,
        matrix_after_conways_train,
        matrix_with_path_train,
        matrix_after_conways_test,
        matrix_with_path_test,
        model=None,
    ):
        self.matrix_after_conways_train = matrix_after_conways_train
        self.matrix_with_path_train = matrix_with_path_train
        self.matrix_after_conways_test = matrix_after_conways_test
        self.matrix_with_path_test = matrix_with_path_test
        self.model = model

    def build(self):
        # Initialize the model
        self.model = Sequential(name=MODEL_NAME)

        # CNN layer for 2D input
        self.model.add(
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(MATRIX_ROWS, MATRIX_COLS, 1)
            )
        )
        # Flatten layer for the tensor to 1D vector
        self.model.add(Flatten())

        # Dense layer
        self.model.add(Dense(25, activation=ACTIVATION_FUNCTION))

        # Need to reshape the tensor to 2D matrix
        self.model.add(Reshape((MATRIX_ROWS, MATRIX_COLS, 1)))

        # Compile the model
        self.model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)

        # Callback to prevent overfitting
        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=PATIENCE)

        # Print the model summary
        self.model.summary()

        # Fit the model
        self.model.fit(
            self.matrix_after_conways_train,
            self.matrix_with_path_train,
            epochs=EPOCHS,
            validation_data=(
                self.matrix_after_conways_test,
                self.matrix_with_path_test,
            ),
            callbacks=[callback],
        )

        return self.model

    def save(self):
        output_path = OUTPUT_PATH + CURRENT_RUN + str(find_last_run() + 1) + "/"

        # Create the output folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the summary of the model
        with open(output_path + MODEL_NAME + "_summary.txt", "w") as f:
            with redirect_stdout(f):
                self.model.summary()

        # Save the model architecture
        model_json = self.model.to_json()
        with open(output_path + MODEL_NAME + ".json", "w") as json_file:
            json_file.write(model_json)

        # Save the model parameters
        with open(output_path + MODEL_NAME + "_parameters.txt", "w") as f:
            f.write("Activation function: " + ACTIVATION_FUNCTION + "\n")
            f.write("Loss function: " + LOSS_FUNCTION + "\n")
            f.write("Optimizer: " + OPTIMIZER + "\n")
            f.write("Metrics: " + str(METRICS) + "\n")
            f.write("Epochs: " + str(EPOCHS) + "\n")
            f.write("Patience: " + str(PATIENCE) + "\n")

        # Save the model
        self.model.save("models/" + MODEL_NAME + ".h5")

    def plot(self):
        output_path = OUTPUT_PATH + CURRENT_RUN + str(find_last_run()) + "/"
        # Plot the model
        tf.keras.utils.plot_model(
            self.model,
            to_file=output_path + MODEL_NAME + ".png",
            show_shapes=True,
            show_layer_names=True,
        )

        # Plot the training and validation accuracy and loss at each epoch
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.model.history.history["accuracy"], label="Training Accuracy")
        plt.plot(
            self.model.history.history["val_accuracy"], label="Validation Accuracy"
        )
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.model.history.history["loss"], label="Training Loss")
        plt.plot(self.model.history.history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(output_path + MODEL_NAME + "_plot.png")
        plt.close()

    # Plot extra plots that show the matrices before and after Conways,
    # the prediction and the difference between the prediction and the matrix with path
    def plot_extra(self):
        output_path = OUTPUT_PATH + CURRENT_RUN + str(find_last_run()) + "/"
        for i in range(5):
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            plt.subplot(151)
            ax = sns.heatmap(
                self.matrix_after_conways_test[i, :, :],
                annot=True,
                cmap="inferno",
                linewidths=0.5,
                linecolor="black",
                cbar=False,
            )
            plt.title("Matrix after Conways")
            plt.subplot(152)
            ax = sns.heatmap(
                self.matrix_with_path_test[i, :, :],
                annot=True,
                cmap="inferno",
                linewidths=0.5,
                linecolor="black",
                cbar=False,
            )
            plt.title("Matrix with path")

            plt.subplot(153)
            ax = sns.heatmap(
                np.around(
                    np.abs(
                        self.model.predict(
                            self.matrix_after_conways_test[i, :, :].reshape(1, 5, 5, 1)
                        ).reshape(5, 5)
                    ),
                    decimals=2,
                ),
                annot=True,
                cmap="inferno",
                linewidths=0.5,
                linecolor="black",
                cbar=False,
            )
            plt.title("Prediction")

            plt.subplot(154)
            ax = sns.heatmap(
                np.around(
                    np.abs(
                        self.matrix_with_path_test[i, :, :]
                        - self.model.predict(
                            self.matrix_after_conways_test[i, :, :].reshape(1, 5, 5, 1)
                        ).reshape(5, 5)
                        * 2
                    ),
                    decimals=2,
                ),
                annot=True,
                cmap="inferno",
                linewidths=0.5,
                linecolor="black",
                cbar=False,
            )
            plt.title("Difference")

            plt.subplot(155)
            ax = sns.heatmap(
                np.around(
                    np.abs(
                        self.matrix_with_path_test[i, :, :]
                        - self.model.predict(
                            self.matrix_after_conways_test[i, :, :].reshape(1, 5, 5, 1)
                        ).reshape(5, 5)
                        * 2
                    ),
                    decimals=2,
                )
                > 1.0,
                annot=True,
                cmap="inferno",
                linewidths=0.5,
                linecolor="black",
                cbar=False,
            )
            plt.title("Difference > 1.0")

            fig = plt.gcf()
            fig.set_size_inches((22, 11), forward=False)
            plt.savefig(
                output_path + MODEL_NAME + "_plot_extra_" + str(i) + ".png", dpi=500
            )
            plt.close()


if __name__ == "__main__":
    # load datasets
    # matrix, matrix_after_conways, matrix_with_path = load_datasets("dataset", 1)

    # load cleaned datasets
    matrix, matrix_after_conways, matrix_with_path = (
        np.load("./output/X_input_clean.npy"),
        np.load("./output/X_after_conways_clean.npy"),
        np.load("./output/y_target_clean.npy"),
    )

    # print matrices shapes
    print("\n\nDataset before preprocessing:")
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

    # Create the CNN model
    cnn = CNN(
        matrix_after_conways_train,
        matrix_with_path_train,
        matrix_after_conways_test,
        matrix_with_path_test,
    )

    # Build the CNN model
    cnn.build()

    # Save the CNN model
    cnn.save()

    # Plot the CNN model
    cnn.plot()

    # Plot extra useful plots
    cnn.plot_extra()
