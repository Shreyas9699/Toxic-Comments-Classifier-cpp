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