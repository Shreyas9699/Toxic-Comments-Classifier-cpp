# Toxic Comments Classifier

## Overview

The Toxic Comments Classifier is a machine learning project that aims to detect toxic comments on online platforms. Built using C++, this project utilizes word embeddings and a simple multi-layer perceptron (MLP) to classify comments as toxic or non-toxic. It achieves over 90% accuracy, contributing to healthier online discussions by effectively filtering harmful content.

This repo is an extension of [Basic Architecture](https://github.com/Shreyas9699/Neural-Network-CPP), check it out to get overview of the MLP and Perceptron implementation

## Features
- *header/DataProcessor.*{*h,cpp*}
    > For loading GloVe word embeddings, training data, prediciting test data post model is trained.
- *header/MLPerceptrons.*{*h,cpp*}
    > Implements Perceptron, multilayer perceptrons, activation function, uses Binary Cross-Entropy Loss instead of MSE, different activation functions (sigmoid, relu, thanh, step, leaky_relu), apply dropout only during training and only to hidden layers, weight initialization, forward/backward pass details.
- *utility/DataPreProcessor.py*
    > For preprocess the raw data from `data/train.7z` and `test.7z`.
- *utility/TextToBinary.cpp*
    > Converts the word embedding file into binar for faster loading (~3m to 5s)
- *main.cpp*
    > Based on the activation function specified and model layers, the model is trained and then prompts the user to enter text and predicts whether it is Toxic or Non-toxic. It takes in various parameters for customization of the network.

## Getting Started
### Prerequisites
- C++ compiler (e.g., g++, clang++ or Visual Studio(C++ 17 or above) )
- [GloVe](https://nlp.stanford.edu/projects/glove/) 
    - glove.6B.100d.txt (used in this project)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Shreyas9699/Toxic-Comments-Classifier-cpp.git
   cd Toxic-Comments-Classifier-cpp
   ```

2. [Click here](https://nlp.stanford.edu/data/glove.6B.zip) to download the latest GloVe 6B file or visit [GloVe](https://nlp.stanford.edu/projects/glove/), unzip them and copy `glove.6B.100d.txt` to `data/glove6B/`

3. run the `utility/TextToBinary.cpp`, make sure to the input and outfile names are correct in the file, to convert into a binary file for faster loading.

4. Compile the Project: To compile the project, use the following command:
    ```bash
    g++ -g -std=c++17 main.cpp header/MLPerceptrons.cpp header/DataProcessor.cpp -o main
    ```
    Or 
    Build if using Visual Studio

5. Run the Classifier: Execute the compiled program:
    ```bash
    ./main
    # same ass
    ./main --train data/train_data.csv --test data/test_data.csv --val_ratio 0.2 --glove data/glove6B/glove.6B.100d.bin --epochs 3 --lr 0.01 --batch_size 64 --log main.log --hidden_layers 64,32,16 --activation TANH --bias 1.0 --dropout 0.2
    ```
6. You should be able to see model details, time take for each epoch on console and weights after each epoch in `main.log`. 

### Flag Reference
Usage:  ```./main [--train PATH] [--test PATH] [--glove PATH] [--epochs N] [--lr FLOAT] [--batch_size N] [--log FILE] [--hidden_layers N,N,...] [--activation TANH|RELU|SIGMOID] [--bias FLOAT] [--dropout FLOAT]```

| Flag               | Type           | Default                           | Description                                                                                           |
|--------------------|----------------|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| `--train PATH`     | string (file)  | `data/train_data.csv`             | Path to the CSV file containing your training examples. Each row should be a preprocessed comment and its label. |
| `--test PATH`      | string (file)  | `data/test_data.csv`              | Path to the CSV file used for final evaluation and prediction.                                         |
| `--val_ratio R`    | float (0–1)    | `0.2`                             | Fraction of the full dataset reserved for validation. A value of `0.2` means 20% validation / 80% training. |
| `--glove PATH`     | string (file)  | `data/glove6B/glove.6B.100d.bin`  | Path to your GloVe embeddings. Can be a pre-converted binary (`.bin`) or a text file (`.txt`)—if missing, the program will attempt on-the-fly conversion. |
| `--epochs N`       | integer        | `3`                               | Number of full passes over the training dataset.                                                       |
| `--lr FLOAT`       | float          | `0.01`                            | Learning rate for gradient descent in backpropagation.                                                 |
| `--batch_size N`   | integer        | `64`                              | Number of samples processed before each weight update.                                                 |
| `--log FILE`       | string (file)  | `main.log`                        | Path to the log file where epoch metrics, weights dumps, and overall progress are recorded.            |
| `--hidden_layers L`| comma-sep list | `64,32,16`                        | Sizes of each hidden layer in the MLP. For example, `--hidden_layers 128,64` builds two hidden layers of 128 and 64 units. |
| `--activation A`   | enum           | `TANH`                            | Activation function for all hidden layers. Options: `TANH`, `RELU`, `SIGMOID`, `LEAKY_RELU`, `STEP`.    |
| `--bias FLOAT`     | float          | `1.0`                                 | Initial bias value added to each neuron before activation.                                             |
| `--dropout FLOAT`  | float (0–1)    | `0.2`                             | Dropout rate applied to hidden layers during training (fraction of neurons zeroed out each pass).      |
| `--help`           | flag           | —                                 | Prints the full usage information (list of flags and defaults) and exits.   


### File Structure
```
.
├── data
│   ├── glove6B
│   │   ├── glove.6B.100d.txt
│   │   ├── glove.6B.100d.bin
│   │   ├── glove.6B.50d.txt
│   │   ├── glove.6B.200d.txt
│   │   └── glove.6B.300d.txt
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── test.7z
│   └── train.7z
├── header
│   ├── DataProcessor.cpp
│   ├── DataProcessor.h
│   ├── MLPerceptrons.cpp
│   └── MLPerceptrons.h
├── utility
│   ├── DataPreProcessor.py
│   └── TextToBinary.cpp
├── main.cpp
└── README.md
└── gitignore.txt
```

### Contributing
Contributions are welcome! If you'd like to improve this project, please fork the repository and submit a pull request.

### Acknowledgements
[GloVe](https://nlp.stanford.edu/projects/glove/) for the pre-trained word embeddings.


### Contact
For questions or feedback, feel free to reach out to me at [shreyas.official13@gmail.com](mailto:shreyas.official13@gmail.com).


### Notes:
- The `train_data.csv` and `test_data.csv` are processed csv files. I have used the `DataPreProcessor.py` python script to process the raw data.
- You can find the raw data `data/train.7z` and `test.7z`
    - to unzip the data follow below commands
        ```bash
        cd Toxic-Comments-Classifier-cpp
        sudo apt-get update
        sudo apt-get install p7zip-ful
        7z x train.7z -odata/
        7z x test.7z -odata/
        ```

- To run the DataPreProcessor so recreate the `train_data.csv` and `test_data.csv` (feel free to modify the `DataPreProcessor.py` based on your requirement)
    ```bash
    python DataPreProcessor.py
    ```

- If you are not able to download the `glove.6B.100d.txt` or want to get it manually, [Click here](https://nlp.stanford.edu/data/glove.6B.zip) to download the latest GloVe 6B file or visit [GloVe](https://nlp.stanford.edu/projects/glove/)