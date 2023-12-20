#include "csvReader.h"
#include "tokenizer.h"
#include <iostream>
#include <vector>
using namespace std;

int main () {
    // loading training data
    vector<string> xTrainData = readData ("data/x_train.csv"); // Assuming a function to read CSV data
    vector<string> yTrainData = readData ("data/y_train.csv");
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

    // Display the results
    cout << "Tokenized Sequences:" << endl;
    for (const auto& sequence : xTokenized) {
        for (int token : sequence) {
            cout << token << " ";
        }
        cout << endl;
    }

    cout << "\n\nPadded Sequences:" << endl;
    for (const auto& sequence : xTrainVal) {
        for (int token : sequence) {
            cout << token << " ";
        }
        cout << endl;
    }
    
    return 0;
}