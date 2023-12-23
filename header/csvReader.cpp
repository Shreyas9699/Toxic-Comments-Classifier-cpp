#include "csvReader.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>


std::vector<std::string> readData(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> data;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data; // Return empty data on error
    }

    std::string line;
    while (std::getline(file, line)) {
        // boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
        data.push_back(line);
    }

    return data;
}

std::vector<double> convertToDouble(const std::vector<std::string>& stringVector) {
    std::vector<double> doubleVector;

    for (const auto& str : stringVector) {
        try {
            // Attempt to convert the string to double and push it into the new vector
            double convertedValue = std::stod(str);
            doubleVector.push_back(convertedValue);
        } catch (const std::invalid_argument& e) {
            // Handle conversion errors if needed
            std::cerr << "Error converting string to double: " << e.what() << std::endl;
        }
    }

    return doubleVector;
}


void printSample (std::vector<std::string> sample) {
    std::cout << "First 10 rows from the of the data set are : " << std::endl;

    int rowCnt = 0;
    for (const auto& row : sample) {
        std::cout << "Row " << rowCnt + 1 << ": \t" << row << std::endl;
        rowCnt++;
    }
}