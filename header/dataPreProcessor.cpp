#include "dataPreProcessor.h"

const int maxVocabularySize = 20000;
std::unordered_map<std::string, int> wordToIndex;
static int index = 1;

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