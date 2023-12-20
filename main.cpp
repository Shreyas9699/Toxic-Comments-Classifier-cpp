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
    // printSample(vector<string> (xTrainData.begin(), xTrainData.begin() + 10));

    // printSample(vector<string> (yTrainData.begin(), yTrainData.begin() + 10));

    // printSample(vector<string> (xTestData.begin(), xTestData.begin() + 10));

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

    int embedding_dim = 100;
    int epochs;
    double MSE;
    // Segment Display Recognition:
    // Recognize number from a seven-segment display
    cout << "-------------------------------- Segment Display Recognition System --------------------------------" << endl;
    cout << "How many epochs?: ";
    cin >> epochs;
    cin.ignore();

    vector<vector<double>> embedding_matrix = createEmbeddingMatrix(xTrainVal);

    MultilayerPerceptron model({100, 64, 1}); 
    
    return 0;
}