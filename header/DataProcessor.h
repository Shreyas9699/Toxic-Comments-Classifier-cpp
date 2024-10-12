#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include "MLPerceptrons.h"

std::unordered_map<std::string, std::vector<double>> loadGloVeEmbeddings(const std::string& filepath);
std::vector<double> preprocessComment(const std::string& comment_text, const std::unordered_map<std::string, std::vector<double>>& embeddings);
std::vector<std::pair<std::vector<double>, int>> loadTrainingData(const std::string& filepath, std::unordered_map<std::string, std::vector<double>>& embeddings);
void predictTestData(const std::string& testFilePath, const std::unordered_map<std::string, std::vector<double>>& embeddings, MultilayerPerceptron& mlp, std::ofstream& logFile, size_t print_limit = 5) ;