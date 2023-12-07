# lee-gol-cnn
## Conway's Game of Life and Shortest Path Finding with CNN

Welcome to Conway's Game of Life and Shortest Path Finding project! This project combines Conway's Game of Life cellular automaton with a Convolutional Neural Network (CNN) to predict the shortest path in a matrix after a certain number of iterations.

## Table of Contents
- [About](#about)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Output Directory](#output)
- [Run in Google Colab](#usage-in-google-colab)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

<a name="about"></a>
## About
This project consists of two main components:

1. **Conway's Game of Life**: The cellular automaton rules are applied to a randomly generated binary matrix. The game evolves over a specified number of iterations.

2. **Convolutional Neural Network (CNN)**: A CNN is used to predict the shortest path in the matrix after Conway's Game of Life iterations.

<a name="prerequisites"></a>
## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- NumPy
- Matplotlib
- Keras
- TensorFlow
- Seaborn

<a name="installation"></a>
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lkmeta/lee-gol-cnn.git

2. Navigate to the project directory:
   ```bash
   cd lee-gol-cnn

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

<a name="usage"></a>
## Usage
1. Run the ```main.py``` script:
   ```bash
   python main.py

The script will load random matrices (that I generated), apply Conway's Game of Life, train the CNN, and save the model and visualizations in the output directory.

<a name="output"></a>
## Output Directory
The `output` directory contains the following:

**Datasets for Testing**
- `test` folder: Contains datasets generated for testing purposes.

**Data Files**  
*Note*: use the following datasets if you don't want to spend time generating them.
- `X_after_conways_clean.npy`: Data after Conway's Game of Life.
- `X_input_clean.npy`: Cleaned input data.
- `y_target_clean.npy`: Cleaned target data.

**Test Images**
- `test_conways_game_of_life.png`: Testing Conway's Game of Life.
- `test_lee_algorithm.png`: Testing the Lee algorithm.

<a name="usage-in-google-colab"></a>
## Run in Google Colab
To run this project in a Google Colab environment, you can use the [Google Colab notebook](https://github.com/lkmeta/lee-gol-cnn/blob/main/src/cloud_env.ipynb) which has all the needed steps.

<a name="results"></a>
## Results
After training, the CNN demonstrates the capability to find the right path in the generated matrices as we can see in the following picture.

<div style="text-align:center">
	<img src="https://github.com/lkmeta/lee-gol-cnn/blob/main/output/CNN_model_v2_plot_extra_2.png" alt="Training Results" "/>
</div>

<a name="contributing"></a>
## Contributing
Feel free to contribute by opening issues, suggesting improvements, or submitting pull requests. Your feedback is highly appreciated!

<a name="license"></a>
## License
This project is licensed under [MIT](https://github.com/lkmeta/lee-gol-cnn/blob/main/LICENSE).
