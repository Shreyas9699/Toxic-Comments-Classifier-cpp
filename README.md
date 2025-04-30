# Toxic Comments Classifier

## Overview

The Toxic Comments Classifier is a machine learning project that aims to detect toxic comments on online platforms. Built using C++, this project utilizes word embeddings and a simple multi-layer perceptron (MLP) to classify comments as toxic or non-toxic. It achieves over 90% accuracy, contributing to healthier online discussions by effectively filtering harmful content.


## Features

- **Toxicity Detection**: Classifies comments based on their toxicity level.
- **Word Embeddings**: Utilizes GloVe embeddings for feature extraction.
- **MLP Architecture**: Implements a simple multi-layer perceptron for classification.
- **Data Handling**: Efficiently processes and manages training and testing data.

## Getting Started

### Prerequisites

- C++ compiler (e.g., g++, clang++)
- Git LFS (if handling large files like GloVe embeddings)
- [Glove](https://nlp.stanford.edu/projects/glove/) 
    - glove.6B.100d.txt (used in this project)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Shreyas9699/Toxic-Comments-Classifier-cpp.git
   cd Toxic-Comments-Classifier-cpp
   ```

2. [Click here](https://nlp.stanford.edu/data/glove.6B.zip) to download the latest Glove 6B file or visit [GloVe](https://nlp.stanford.edu/projects/glove/), unzip them and copy `glove.6B.100d.txt` to `data/glove6B/`

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
    ```
6. You should be able to see model details, time take for each epoch on console and weights after each epoch in `main.log`. 


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

- If you are not able to download the `glove.6B.100d.txt` or want to get it manually, [Click here](https://nlp.stanford.edu/data/glove.6B.zip) to download the latest Glove 6B file or visit [GloVe](https://nlp.stanford.edu/projects/glove/)



