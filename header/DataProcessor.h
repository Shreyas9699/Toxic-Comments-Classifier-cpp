#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include "MLPerceptrons.h"

std::unordered_map<std::string, std::vector<float>> loadGloVeEmbeddings(const std::string& filepath);

std::unordered_map<std::string, std::vector<float>> loadGloVeEmbeddingsBinary(const std::string& filename);

std::vector<float> preprocessComment(const std::string& comment_text, const std::unordered_map<std::string, std::vector<float>>& embeddings);

std::vector<std::pair<std::vector<float>, int>> loadTrainingData(const std::string& filepath, std::unordered_map<std::string, std::vector<float>>& embeddings);

void predictTestData(const std::string& testFilePath, const std::unordered_map<std::string, std::vector<float>>& embeddings, MultilayerPerceptron& mlp, std::ofstream& logFile, size_t print_limit = 5) ;