#include "csvReader.h"
#include "tokenizer.h"
#include "MLPerceptrons.h"
#include <iostream>
#include <vector>
using namespace std;

void printRandomSubset(const std::vector<std::vector<int>>& xTrainVal, int subsetSize);

int main () {
    srand(time(NULL));
    rand();

    // loading training data
    vector<string> xTrainData = readData ("data/x_train.csv"); // Assuming a function to read CSV data
    vector<double> yTrainData = convertToDouble(readData ("data/y_train.csv"));
    vector<string> xTestData = readData ("data/x_test.csv");

    // Parameters
    int maxFeatures = 8000; // 20000 reduced to 10000
    int maxTextLength = 350; // 400 reduced to 350

    // Tokenizer instance
    Tokenizer xTokenizer(maxFeatures);

    // Fit on texts and get tokenized sequences
    vector<vector<int>> xTokenized = xTokenizer.fitOnTexts(vector<string> (xTrainData.begin(), xTrainData.begin() + 30000));

    // Pad sequences to maxTextLength
    vector<vector<int>> xTrainVal = xTokenizer.padSequences(xTokenized, maxTextLength);
    cout << "Tokenization got completed!" << endl;
    cout << "Training data dimension : " << xTrainVal.size() << "x" <<  xTrainVal[10].size() << endl;

    // Print random records from xTrainVal
    // printRandomSubset(xTrainVal, 5);
    
    int embedding_dim = 100;
    int epochs = 1000;
    double MSE;

    vector<vector<float>> embedded_data = createEmbeddingMatrix (xTrainVal, maxFeatures, embedding_dim, maxTextLength);

    // MultilayerPerceptron model({100, 64, 1});

    // for (int i = 0; i < epochs; i++) {
    //     MSE = 0.0;
    //     for (int i = 0; i < )
    //     MSE += model.backPropagation({1, 1, 1, 1, 1, 1, 0}, {0.05});
    // MSE /= 10.0;            // number of different ouputs
    // cout << "7 to 1 Network MSE: " << MSE << endl;
    
    return 0;
}

void printRandomSubset(const std::vector<std::vector<int>>& xTrainVal, int subsetSize) {
    // Create a vector of indices
    std::vector<int> indices(xTrainVal.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices randomly
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (int i = 0; i < subsetSize; ++i) {
        for (int j = 0; j < xTrainVal[indices[i]].size(); ++j) {
            cout << xTrainVal[i][j] << " ";
        }
         cout << endl;
    }
}