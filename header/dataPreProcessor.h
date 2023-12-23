#pragma once
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include <random>
#include <boost/tokenizer.hpp>

std::vector<std::vector<std::string>> tokenizeData(const std::vector<std::string>& textData);
std::set<std::string> createVocabulary(const std::vector<std::vector<std::string>>& tokenizedData);
std::vector<std::vector<int>> tokenizeAndNumberizeData(const std::vector<std::vector<std::string>>& tokenizedData);
std::vector<std::vector<double>> createEmbeddingMatrix(int vocabSize, int embeddingDimension);
void padData(std::vector<std::vector<int>>& numData, int maxSequenceLength, int paddingTokenIndex);
void createWordIdx(const std::set<std::string>& vocabulary);

// Declare wordToIndex here
extern std::unordered_map<std::string, int> wordToIndex;