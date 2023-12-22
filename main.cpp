#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include <random>
#include <boost/tokenizer.hpp> // sudo apt install libboost-all-dev
#include "csvReader.h"
#include "MLPerceptrons.h"

std::vector<std::vector<std::string>> tokenizeData(const std::vector<std::string>& textData) {
    std::vector<std::vector<std::string>> tokenizedData;
    for (const std::string& text : textData) {
        boost::tokenizer<boost::escaped_list_separator<char>> tok(text);  // Use robust tokenizer
        tokenizedData.push_back(std::vector<std::string>(tok.begin(), tok.end()));
    }
    return tokenizedData;
}

std::set<std::string> createVocabulary(const std::vector<std::vector<std::string>>& tokenizedData) {
    std::set<std::string> vocabulary;
    for (const auto& tokens : tokenizedData) {
        vocabulary.insert(tokens.begin(), tokens.end());
    }
    return vocabulary;
}

void padData(std::vector<std::vector<std::string>>& tokenizedData, int maxSequenceLength, const std::string& paddingToken) {
    for (auto& tokens : tokenizedData) {
        tokens.resize(maxSequenceLength, paddingToken);  // Pad with specified token
    }
}

std::vector<std::vector<double>> createEmbeddingMatrix(const std::set<std::string>& vocabulary, int embeddingDimension) {
    const int vocabSize = vocabulary.size();
    std::unordered_map<std::string, int> wordToIndex;
    int index = 0;
    for (const std::string& word : vocabulary) {
        wordToIndex[word] = index;
        index++;
    }

    std::vector<std::vector<double>> embeddingMatrix(vocabSize, std::vector<double>(embeddingDimension));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);  // Consistent initialization
    for (int i = 0; i < vocabSize; i++) {
        for (int j = 0; j < embeddingDimension; j++) {
            embeddingMatrix[i][j] = dis(gen);
        }
    }
    return embeddingMatrix;
}


int main () {
    std::vector<std::string> rawXTrainData = readData ("data/x_train.csv");
    std::vector<std::string> xTrainData = std::vector<std::string> (rawXTrainData.begin(), rawXTrainData.begin() + 300);

    std::vector<double> rawYTrainData = convertToDouble(readData ("data/y_train.csv"));
    std::vector<double> yTrainData = std::vector<double> (rawYTrainData.begin(), rawYTrainData.begin() + 300);

	std::cout << "Training Dataset size: " << xTrainData.size() << std::endl;
	// Tokenize
	std::vector<std::vector<std::string>> tokenizedData = tokenizeData(xTrainData);

    // for (int i = 0; i < 5; i++) { // Print first 5 tokenized sequences
    //     std::cout << "\n\n --------------- Tokenized sequence: --------------- \n\n";
    //     for (const std::string& token : tokenizedData[i]) {
    //         std::cout << token << " ";
    //     }
    //     std::cout << std::endl;
    // }
	
	// Create vocabulary
	std::set<std::string> vocabulary = createVocabulary(tokenizedData);
    std::cout << "\n --------------- Vocabulary size: " << vocabulary.size() << " ---------------" << std::endl;
    // std::cout << "\n --------------- Sample vocabulary words: --------------- \n";
    // int count = 0;
    // for (const std::string& word : vocabulary) {
    //     std::cout << word << " ";
    //     count++;
    //     if (count == 10) { // Print up to 10 words
    //         break;
    //     }
    // }
    // std::cout << std::endl;
	
	// Pad
	padData(tokenizedData, 300, "<PAD>");  // Adjust maxSequenceLength and paddingToken as needed
    // for (int i = 0; i < 5; i++) { // Print first 5 padded sequences
    //     std::cout << "\n --------------- Padded sequence: --------------- \n";
    //     for (const std::string& token : tokenizedData[i]) {
    //         std::cout << token << " ";
    //     }
    //     std::cout << std::endl;
    // }

	
	// Create embedding matrix
    std::vector<std::vector<double>> embeddingMatrix = createEmbeddingMatrix(vocabulary, 100);  // Adjust embeddingDimension
    std::cout << "\n\n --------------- Embedding matrix dimensions: " << 
                embeddingMatrix.size() << " x " << embeddingMatrix[0].size() << " ---------------\n" << std::endl;
    // std::cout << "\n --------------- Sample rows of embedding matrix: --------------- \n" << std::endl;
    // for (int i = 0; i < 5; i++) { // Print first 5 rows
    //     for (double value : embeddingMatrix[i]) {
    //         std::cout << value << " ";
    //     }
    //     std::cout << std::endl;
    // }

    int epochs = 3000;
    double MSE;

    MultilayerPerceptron mlpModel({100, 66, 1});  // 
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
    std::cout << "Final MSE: " << MSE << std::endl;

    // std::vector<std::string> rawXTestData = readData ("data/x_test.csv");
    // std::vector<std::string> xTest = std::vector<std::string> (rawXTestData.begin(), rawXTestData.begin() + 10);
    // std::vector<double> rawYTestData = convertToDouble(readData ("data/y_test.csv"));
    // std::vector<double> yTest = std::vector<double> (rawYTestData.begin(), rawYTestData.begin() + 10);

    // std::vector<std::vector<std::string>> tokenizedTestData = tokenizeData(rawXTestData);
	// padData(tokenizedTestData, 300, "<PAD>");
    // std::vector<std::vector<double>> embeddingTestMatrix = createEmbeddingMatrix(vocabulary, 100); 

    // for (int j = 0; j < embeddingTestMatrix.size(); j++) {
    //     mlpModel.run(embeddingMatrix[j], {yTrainData[j]});
    // }

	return 0;
}

// g++ -g main.cpp csvReader.cpp MLPerceptrons.cpp -o main
// gdb main