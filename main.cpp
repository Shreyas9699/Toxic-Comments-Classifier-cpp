#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include <random>
#include <boost/tokenizer.hpp> // sudo apt install libboost-all-dev
#include "csvReader.h"
#include "MLPerceptrons.h"

const int maxVocabularySize = 250;
std::unordered_map<std::string, int> wordToIndex = {{"<PAD>", 0}};
int index = 1;

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
        if (maxVocabularySize < vocabulary.size()) {
            break;
        }
        vocabulary.insert(tokens.begin(), tokens.end());
    }
    return vocabulary;
}

void padData(std::vector<std::vector<int>>& numData, int maxSequenceLength, int paddingTokenIndex) {
    for (auto& tokens : numData) {
        tokens.resize(maxSequenceLength, paddingTokenIndex); // Pad with specified token
    }
}

void createWordIdx(const std::set<std::string>& vocabulary) {
    for (const std::string& word : vocabulary) {
        wordToIndex[word] = index;
        index++;
    }
}

std::vector<std::vector<int>> tokenizeAndNumberizeData(const std::vector<std::vector<std::string>>& tokenizedData) {
    std::vector<std::vector<int>> numData;
    for (const std::vector<std::string>& tokens : tokenizedData) {
        std::vector<int> numTokens;
        for (const std::string& token : tokens) {
            int wordIndex = wordToIndex[token];  // Access word index from dictionary
            numTokens.push_back(wordIndex);
        }
        numData.push_back(numTokens);
    }
    return numData;
}

std::vector<std::vector<double>> createEmbeddingMatrix(int vocabSize, int embeddingDimension) {
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
    std::vector<double> rawYTrainData = convertToDouble(readData ("data/y_train.csv"));

	MultilayerPerceptron mlpModel({100, 66, 1});
	
    for (int n = 0; n < 1000; n += 100) { // only first 1000 values from dataset
        std::vector<std::string> xTrainData = std::vector<std::string> (rawXTrainData.begin(), rawXTrainData.begin() + n + 100);
        std::vector<double> yTrainData = std::vector<double> (rawYTrainData.begin(), rawYTrainData.begin() + n + 300);
        std::cout << "\n -------------- Training Dataset starting from : " << n << " to " << n + 100
                    << " --------------" << std::endl;

        // Tokenize
        std::vector<std::vector<std::string>> tokenizedData = tokenizeData(xTrainData);

        // Create vocabulary
        std::set<std::string> vocabulary = createVocabulary(tokenizedData);
        std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;
        createWordIdx(vocabulary);
        std::vector<std::vector<int>> numData = tokenizeAndNumberizeData(tokenizedData);

        // Pad
        padData(numData, 100, wordToIndex["<PAD>"]);  // Adjust maxSequenceLength and paddingToken as needed
        
        // Create embedding matrix
        std::vector<std::vector<double>> embeddingMatrix = createEmbeddingMatrix(wordToIndex.size(), 50);  // Adjust embeddingDimension
        std::cout << "Embedding matrix dimensions: " << 
                embeddingMatrix.size() << " x " << embeddingMatrix[0].size() << std::endl;

        int epochs = 700;
        double MSE;
        for (int i = 0; i < epochs; i++) { 
            MSE = 0.0;
            for (int j = 0; j < embeddingMatrix.size(); j++) {
                MSE += mlpModel.backPropagation(embeddingMatrix[j], {yTrainData[j]});
            }
        }
        MSE /= 2.0;            // number of different ouputs
        std::cout << "MSE after processing data  from " << n << " to " <<  n + 100 << " is :" << MSE << std::endl;
        xTrainData.clear();
        yTrainData.clear();
        tokenizedData.clear();
        embeddingMatrix.clear();
    }
	
	return 0;
}

// g++ -g main.cpp csvReader.cpp MLPerceptrons.cpp -o main
// gdb main