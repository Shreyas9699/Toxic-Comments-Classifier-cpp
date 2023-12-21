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
std::vector<std::vector<int>> Tokenizer::fitOnTexts(const std::vector<std::string>& texts) {
    createWordIndex(texts);
    return textsToSequences(texts);
}

// Pad sequences to the specified maxlen
std::vector<std::vector<int>> Tokenizer::padSequences (const std::vector<std::vector<int>>& sequences, int maxlen) {
    std::vector<std::vector<int>> paddedSequences;
    for (const auto& sequence : sequences) {
        paddedSequences.push_back(padSequence(sequence, maxlen));
    }
    return paddedSequences;
}


// Tokenize text and return sequence of token indices
std::vector<int> Tokenizer::textToSequence (const std::string& text) const {
    std::vector<int> sequence;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        if (wordIndex.find(word) != wordIndex.end()) {
            sequence.push_back(wordIndex.at(word));
        }
    }
    return sequence;
}

// Tokenize all texts and return sequences of token indices
std::vector<std::vector<int>> Tokenizer::textsToSequences (const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> sequences;
    for (const auto& text : texts) {
        sequences.push_back(textToSequence(text));
    }
    return sequences;
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
        wordIndex[allWords[i]] = i + 1;  // Index 0 is reserved
    }
}

std::vector<std::vector<std::vector<float>>> createEmbeddingMatrix(const std::vector<std::vector<int>>& xTrainVal, int embeddingDim, int maxFeatures) {
    // Create a zero-filled embedding matrix using NumPy
    std::cout << "Started creating EmbeddingMatrix" << std::endl;
    std::vector<std::vector<float>> embedding_matrix(maxFeatures, std::vector<float>(embeddingDim, 0.0f));
    std::vector<std::vector<std::vector<double>>> embedded_sequences;

    std::cout << "initiliased EmbeddingMatrix" << std::endl;
    
    for (const std::vector<int>& sequence : xTrainVal) {
        std::vector<std::vector<double>> embedded_sequence;
        for (int word_index : sequence) {
            std::vector<double> embedding_vector(embeddingDim);
            for (int i = 0; i < embeddingDim; i++) {
                embedding_vector[i] = embedding_matrix[word_index][i];
            }
            embedded_sequence.push_back(embedding_vector);
        }
        embedded_sequences.push_back(embedded_sequence);
    }

    std::cout << "Created embedded_sequences" << std::endl;

    std::vector<std::vector<std::vector<float>>> embedded_data(embedded_sequences.size(), std::vector<std::vector<float>>(maxFeatures, std::vector<float>(embeddingDim, 0.0f)));
    std::cout << "Created embedded_data" << std::endl;
    
    // Copy data from embedded_sequences to embedded_data
    for (size_t i = 0; i < embedded_sequences.size(); i++) {
        for (size_t j = 0; j < maxFeatures; j++) {
            for (size_t k = 0; k < embeddingDim; k++) {
                embedded_data[i][j][k] = embedded_sequences[i][j][k];
            }
        }
    }
    std::cout << "Copying data from embedded_sequences to embedded_data complete!" << std::endl;

    std::cout << "Embedded data dim: " << embedded_data.size() << "x" << embedded_data[0].size() << "x" << embedded_data[0][0].size() << std::endl;

    return embedded_data;

}