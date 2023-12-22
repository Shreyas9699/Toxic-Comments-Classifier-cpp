#pragma once
#include <vector>
#include <string>

std::vector<std::string> readData(const std::string& filename);
std::vector<double> convertToDouble(const std::vector<std::string>& stringVector);
void printSample (std::vector<std::string> sample);