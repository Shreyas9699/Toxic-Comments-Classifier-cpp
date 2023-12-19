#include "header/CSV_READER.h"
#include <iostream>
using namespace std;

int main () {
    // loading the data
    vector<vector<string>> data = csvFileReader ("train.csv");
    // std::vector<std::vector<std::string>> sample;
    printSample (vector<vector<string>> (data.begin(), data.begin() + 10));
    return 0;
}


// #include <iostream>
// #include <sstream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <random>
// #include <algorithm>

// int main () {
//     // loading the data
//     std::ifstream file("data/train.csv");
//     std::vector<std::vector<std::string>> data;

//     std::string line;
//     while (std::getline (file, line)) {
//         std::vector<std::string> row;
//         std::istringstream iss(line);
//         std::string field;
//         while (std::getline (iss, field, ',')) {
//             row.push_back(field);
//         }
//         data.push_back(row);
//     }

//     file.close();

//     // Iterate through each row and element, replacing empty strings with your desired fill value (" " in this case):
//     for (auto& row : data) {
//         for (auto& element : row) {
//             if (element.empty()) {
//                 element = " ";
//             }
//         }
//     }

//     std::random_device ranD;
//     std::mt19937 gen(ranD());
//     std::shuffle(data.begin(), data.end(), gen);

//     std::vector<std::vector<std::string>> sample (data.begin(), data.begin() + 10);

//     for (const auto& row : sample) {
//         for (const auto& element : row) {
//             std::cout << element << " ";
//         }
//         std::cout << std::endl;
//     }
//     return 0;
// }