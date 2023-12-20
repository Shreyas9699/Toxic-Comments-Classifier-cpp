#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <unordered_map> 

class Tokenizer {
public:
    Tokenizer(int maxFeatures);
    std::vector<std::vector<int>> fitOnTexts(const std::vector<std::string>& texts);
    std::vector<std::vector<int>> padSequences(const std::vector<std::vector<int>>& sequences, int maxlen);

private:
    int maxFeatures;
    std::unordered_map<std::string, int> wordIndex;

    std::vector<int> textToSequence(const std::string& text) const;
    std::vector<std::vector<int>> textsToSequences(const std::vector<std::string>& texts) const;
    std::vector<int> padSequence(const std::vector<int>& sequence, int maxlen) const;
    void createWordIndex(const std::vector<std::string>& texts);
};