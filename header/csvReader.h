#pragma once
#include <vector>
#include <string>

std::pair<std::vector<std::string>, std::vector<double>> readCSV(const std::string& filename);
void printSample (std::vector<std::string> sample);