from utils import generate_dataset, generate_dataset_without_path

"""
create_10_datasets function creates 10 datasets that will be used for training and testing

Attention!
This process may takes a while to complete... (hours)
If you want to create a single dataset, use generate_dataset function from utils.py

"""


## Create 10 datasets for training and testing
## Dimensions of matrices: 5x5
## Number of matrices: 100
## Name of dataset: dataset_1, dataset_2, ..., dataset_10
def create_10_datasets():
    for i in range(10):
        test_name = "dataset_" + str(i + 1)
        num_of_matrices = 100
        matrix_rows = 5
        matrix_cols = 5
        generate_dataset(num_of_matrices, matrix_rows, matrix_cols, test_name)

    print("10 datasets created successfully!")

    return


if __name__ == "__main__":
    # Short test example
    # Generate dataset with 1 matrix of dimensions 5x5
    # generate_dataset(1, 5, 5, "test")

    # Create 10 datasets for training and testing
    create_10_datasets()

    # Generate dataset with 1 matrix WITHOUT PATH of dimensions 5x5
    # generate_dataset_without_path(1, 5, 5, "test")

    # Generate dataset with 100 matrices WITHOUT PATH of dimensions 5x5
    # generate_dataset_without_path(100, 5, 5, "dataset")
