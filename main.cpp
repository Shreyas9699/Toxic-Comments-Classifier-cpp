#include "csvReader.h"
#include <iostream>
#include <vector>
using namespace std;

int main () {
    // loading training data
    vector<string> xTrainData = readData ("data/x_train.csv"); // Assuming a function to read CSV data
    vector<string> yTrainData = readData ("data/y_train.csv");
    vector<string> xTestData = readData ("data/x_test.csv");
    printSample(vector<string> (xTrainData.begin(), xTrainData.begin() + 10));

    printSample(vector<string> (yTrainData.begin(), yTrainData.begin() + 10));

    printSample(vector<string> (xTestData.begin(), xTestData.begin() + 10));
    return 0;
}