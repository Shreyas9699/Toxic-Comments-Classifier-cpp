#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include <random>
#include "header/csvReader.h"
#include "header/MLPerceptrons.h"
#include "header/dataPreProcessor.h"

int main () {
    std::vector<std::string> rawXTrainData = readData ("data/100toxic_nonToxic_X.csv");
    std::vector<double> rawYTrainData = convertToDouble(readData ("data/100toxic_nonToxic_Y.csv"));

    for (int i = 0; i < 300; i += 50) {
        std::cout << rawXTrainData[i] << std::endl;
        std::cout << rawYTrainData[i] << std::endl;
    }

	MultilayerPerceptron mlpModel({100, 66, 33, 1});
	
    std::vector<std::string> xTrainData = rawXTrainData;
    std::vector<double> yTrainData = rawYTrainData;
    std::cout << "\n -------------- Training Dataset size : " << xTrainData.size() << " --------------" << std::endl;

    // Tokenize
    std::vector<std::vector<std::string>> tokenizedData = tokenizeData(xTrainData);

        // Create vocabulary
    std::set<std::string> vocabulary = createVocabulary(tokenizedData);
    std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;
    createWordIdx(vocabulary);
    std::vector<std::vector<int>> numData = tokenizeAndNumberizeData(tokenizedData);

    // Pad
    padData(numData, 400, wordToIndex["<PAD>"]);  // Adjust maxSequenceLength and paddingToken as needed
        
    // Create embedding matrix
    std::vector<std::vector<double>> embeddingMatrix = createEmbeddingMatrix(wordToIndex.size(), 50);  // Adjust embeddingDimension
    std::cout << "Embedding matrix dimensions: " << 
                embeddingMatrix.size() << " x " << embeddingMatrix[0].size() << std::endl;

    int epochs = 800;
    double MSE;
    for (int i = 0; i < epochs; i++) { 
        MSE = 0.0;
        for (int j = 0; j < embeddingMatrix.size(); j++) {
            MSE += mlpModel.backPropagation(embeddingMatrix[j], {yTrainData[j]});
        }
        if ( i % 100 == 0) {
            std::cout << i << "th run MSE is : " << MSE << std::endl;
        }
    }
    MSE /= 2.0;            // number of different ouputs
    std::cout << "MSE :" << MSE << std::endl;
    // mlpModel.printWeights();

    // Classifier tester
	std::string inputString;

	while (true) {
		std::cout << "Enter a string (or press Enter twice to quit): ";
		getline(std::cin, inputString);

		if (inputString.empty()) {
			// Check for two consecutive empty inputs
			std::cout << "Enter another string to confirm quitting (or press Enter again to quit): ";
			getline(std::cin, inputString);

			if (inputString.empty()) {
               break; // Exit the loop
			} else {
				// Process the input string as needed
				// (Replace this with your actual processing code)
				std::cout << "Processing string: " << inputString << std::endl;
			}
		}
		
		std::vector<std::string> testData = {inputString};
		
		// Preprocess test string
        std::vector<std::vector<std::string>> testTokens = tokenizeData(testData);
        std::vector<int> testIndices = tokenizeAndNumberizeData(testTokens)[0];
        std::vector<std::vector<int>>padd = {testIndices};
        padData(padd, 400, wordToIndex["<PAD>"]);

        // Generate embeddings
        std::vector<double> testInput;
        for (int index : testIndices) {
            if (index == wordToIndex["<PAD>"]) {
            testInput.insert(testInput.end(), embeddingMatrix[0].begin(), embeddingMatrix[0].end());
            } else {
            testInput.insert(testInput.end(), embeddingMatrix[index].begin(), embeddingMatrix[index].end());
            }
        }

        // Run model
        std::vector<double> output = mlpModel.run(testInput);
        double toxicityScore = output[0];

        // Interpret output
        if (toxicityScore >= 0.5) {
            std::cout << "Test string is likely toxic." << std::endl;
        } else {
            std::cout << "Test string is likely non-toxic." << std::endl;
        }
	}

	return 0;
}