#pragma once
#include <vector>
#include <string>

std::vector<std::vector<std::string>> csvFileReader (const char* filename);

void printSample (std::vector<std::vector<std::string>> sample);
