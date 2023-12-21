#include "tokenizer.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <unordered_map>

// Tokenizer class to tokenize the input text
Tokenizer::Tokenizer(int maxFeatures) {
    this->maxFeatures = maxFeatures;
}

// Fit on texts and return tokenized sequences
std::vector<std::vector<int>> Tokenizer::fitOnTexts(const std::vector<std::string>& data) {
    createWordIndex(data);
    return textsToSequences(data);
}

// Tokenize all texts and return sequences of token indices
std::vector<std::vector<int>> Tokenizer::textsToSequences (const std::vector<std::string>& data) const {
    std::cout << "Tokenizing the data into sequence" << std::endl;
    std::vector<std::vector<int>> sequences;
    for (const auto& text : data) {
        sequences.push_back(textToSequence(text));
    }
    std::cout << "Tokenizing Completed! Tokenized data size is :" << sequences.size() << " " << sequences[0].size() << std::endl;
    return sequences;
}

// Tokenize each text and return sequence of token indices
std::vector<int> Tokenizer::textToSequence (const std::string& text) const {
    std::vector<int> sequence;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        if (wordIdx.find(word) != wordIdx.end()) {
            sequence.push_back(wordIdx.at(word));
        }
    }
    return sequence;
}

// Pad sequences to the specified maxlen
std::vector<std::vector<int>> Tokenizer::padSequences (const std::vector<std::vector<int>>& sequences, int maxlen) {
    std::cout << "Padding the data" << std::endl;
    std::vector<std::vector<int>> paddedSequences;
    for (const auto& sequence : sequences) {
        paddedSequences.push_back(padSequence(sequence, maxlen));
    }
    std::cout << "Completed Padding!" << std::endl;
    return paddedSequences;
}

// Pad sequence to the specified maxlen
std::vector<int> Tokenizer::padSequence (const std::vector<int>& sequence, int maxlen) const {
    std::vector<int> paddedSequence;
    if (sequence.size() >= maxlen) {
        paddedSequence = std::vector<int>(sequence.begin(), sequence.begin() + maxlen);
    } else {
        paddedSequence = sequence;
        paddedSequence.resize(maxlen, 0);  // Pad with zeros
    }
    return paddedSequence;
}

// Create word index from texts
void Tokenizer::createWordIndex (const std::vector<std::string>& texts) {
    std::cout << "Creating wordIdx" << std::endl;
    std::vector<std::string> allWords;
    for (const auto& text : texts) {
        std::istringstream iss(text);
        std::copy(std::istream_iterator<std::string>(iss),
               std::istream_iterator<std::string>(),
               std::back_inserter(allWords));
    }

    // Remove duplicate words
    std::sort(allWords.begin(), allWords.end());
    allWords.erase(std::unique(allWords.begin(), allWords.end()), allWords.end());

    // Select top maxFeatures words
    allWords.resize(std::min(maxFeatures, static_cast<int>(allWords.size())));

    // Create word index
    for (int i = 0; i < allWords.size(); ++i) {
        wordIdx[allWords[i]] = i + 1;  // Index 0 is reserved
    }
    std::cout << "Created wordIdx, size is : " << wordIdx.size() << std::endl;
}

std::vector<std::vector<float>> createEmbeddingMatrix(const std::vector<std::vector<int>>& xTrainVal, int maxFeatures, int embeddingDim, int maxTextLength) {
    // Create a zero-filled embedding matrix using NumPy
    std::cout << "Started creating EmbeddingMatrix of dimension : " << maxFeatures << "x" << embeddingDim << std::endl;
    std::vector<std::vector<float>> embeddingMatrix(maxFeatures, std::vector<float>(embeddingDim, 0.0));
    std::vector<std::vector<float>> embeddedData;
    for (const auto& sequence : xTrainVal) {
        std::vector<std::vector<float>> embeddedSequence;
        for (int wordIndex : sequence) {
            embeddedSequence.push_back(embeddingMatrix[wordIndex]);
        }
        std::vector<float> flattenedSequence(maxTextLength * embeddingDim);
        std::transform(embeddedSequence.begin(), embeddedSequence.end(), flattenedSequence.begin(),
               [](const std::vector<float>& vec) { return vec[0]; }); // Return the first element
        embeddedData.push_back(flattenedSequence);
    }
    std::cout << "Completed creating EmbeddingMatrix !" << std::endl;
    return embeddingMatrix;

}