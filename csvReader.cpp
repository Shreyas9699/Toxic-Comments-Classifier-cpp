#include "csvReader.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

std::vector<std::vector<std::string>> csvFileReader (const char* filename) {
    //std::string filename = "train.csv";
    std::ifstream file(filename);
    std::vector<std::vector<std::string>> data;

    std::string line;
    while (std::getline (file, line)) {
        std::vector<std::string> row;
        std::istringstream iss(line);
        std::string field;
        while (std::getline (iss, field, ',')) {
            row.push_back(field);
        }
        data.push_back(row);
    }

    file.close();

    // Iterate through each row and element, replacing empty strings with your desired fill value (" " in this case):
    for (auto& row : data) {
        for (auto& element : row) {
            if (element.empty()) {
                element = " ";
            }
        }
    }

    std::random_device ranD;
    std::mt19937 gen(ranD());
    std::shuffle(data.begin(), data.end(), gen);

    return data;
}

void printSample (std::vector<std::vector<std::string>> sample) {
    std::cout << "First 10 rows from the of the data set are : " << std::endl;

    for (const auto& row : sample) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}