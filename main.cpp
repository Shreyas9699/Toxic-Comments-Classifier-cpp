#include "csvReader.h"
#include "tokenizer.h"
#include "MLPerceptrons.h"
#include <iostream>
#include <vector>
using namespace std;

int main () {
    srand(time(NULL));
    rand();

    // loading training data
    vector<string> xTrainData = readData ("data/x_train.csv"); // Assuming a function to read CSV data
    vector<double> yTrainData = convertToDouble(readData ("data/y_train.csv"));
    vector<string> xTestData = readData ("data/x_test.csv");

    // Parameters
    int maxFeatures = 20000;
    int maxTextLength = 400;

    // Tokenizer instance
    Tokenizer xTokenizer(maxFeatures);

    // Fit on texts and get tokenized sequences
    vector<vector<int>> xTokenized = xTokenizer.fitOnTexts(xTrainData);

    // Pad sequences to maxTextLength
    vector<vector<int>> xTrainVal = xTokenizer.padSequences(xTokenized, maxTextLength);
    cout << xTrainVal.size() << " " <<  xTrainVal[10].size() << endl;

    cout << "Tokenization got completed!" << endl;

    int cnt = 25;
    for (auto row: xTrainVal) {
        if (cnt--) {
            for (auto i: row){
                cout << i << " ";
            }
            cout << endl;
        } else {
            break;
        }
    }

    int embedding_dim = 100;
    int epochs = 1000;
    double MSE;

    // cout << "Calling createEmbeddingMatrix" << endl;
    // vector<vector<vector<float>>> embedded_data = createEmbeddingMatrix (xTrainVal, embedding_dim, maxFeatures);

    // MultilayerPerceptron model({100, 64, 1});

    // for (int i = 0; i < epochs; i++) {
    //     MSE = 0.0;
    //     for (int i = 0; i < )
    //     MSE += model.backPropagation({1, 1, 1, 1, 1, 1, 0}, {0.05});
    // MSE /= 10.0;            // number of different ouputs
    // cout << "7 to 1 Network MSE: " << MSE << endl;
    
    return 0;
}