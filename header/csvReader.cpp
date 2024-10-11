#include "csvReader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

std::pair<std::vector<std::string>, std::vector<double>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    std::vector<std::string> commentText;
    std::vector<double> toxic;
    std::unordered_map<std::string, int> unexpectedValues;

    std::string line, cell;
    std::getline(file, line); // Skip header row

    // Create a log file
    std::ofstream logFile("error_log.log");

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::getline(lineStream, cell, ','); // Extract comment_text column
        commentText.push_back(cell);

        std::getline(lineStream, cell, ','); // Extract toxic column
        try {
            double toxicValue = std::stod(cell);
            if (toxicValue != 0.0 && toxicValue != 1.0) {
                unexpectedValues[cell]++;
                logFile << "Warning: Unexpected value in toxic column: " << cell << std::endl;
            }
            toxic.push_back(toxicValue);
        } catch (const std::invalid_argument& e) {
            unexpectedValues[cell]++;
            logFile << "Error: Invalid conversion to double: " << cell << std::endl;
            toxic.push_back(0.0); // or some other default value
        } catch (const std::out_of_range& e) {
            unexpectedValues[cell]++;
            logFile << "Error: Out of range conversion to double: " << cell << std::endl;
            toxic.push_back(0.0); // or some other default value
        }
    }

    logFile << "Unexpected values in toxic column:" << std::endl;
    for (const auto& pair : unexpectedValues) {
        logFile << pair.first << ": " << pair.second << std::endl;
    }

    file.close();
    logFile.close();

    return std::make_pair(commentText, toxic);
}

void printSample (std::vector<std::string> sample) {
    std::cout << "First 10 rows from the of the data set are : " << std::endl;

    int rowCnt = 0;
    for (const auto& row : sample) {
        std::cout << "Row " << rowCnt + 1 << ": \t" << row << std::endl;
        rowCnt++;
    }
}