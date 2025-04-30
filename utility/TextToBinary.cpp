#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <sstream>

int main()
{
    std::cout << "Converting Glove into binary format." << std::endl;
	std::string inputFile = "data/glove6B/glove.6B.100d.txt";
	std::string outputFile = "data/glove6B/glove.6B.100d.bin";
	int dimensions = 100;

    std::ifstream in(inputFile);
    std::ofstream out(outputFile, std::ios::binary);

    if (!in.is_open()) 
    {
        std::cerr << "Cannot open input file: " << inputFile << std::endl;
        return -1;
    }

    if (!out.is_open()) 
    {
        std::cerr << "Cannot open output file: " << outputFile << std::endl;
        return -1;
    }

    std::string line, word;
    std::vector<float> values(dimensions);

    // Write total vocabulary size as header (initially 0)
    size_t vocabSize = 0;
    out.write(reinterpret_cast<char*>(&vocabSize), sizeof(size_t));

    // Write dimensions as header
    out.write(reinterpret_cast<char*>(&dimensions), sizeof(int));

    // Process each line
    while (std::getline(in, line)) 
    {
        std::istringstream iss(line);

        // Read word
        iss >> word;

        // Write word length
        int length = word.length();
        out.write(reinterpret_cast<char*>(&length), sizeof(int));

        // Write word
        out.write(word.c_str(), length);

        // Read and write embedding values
        for (int i = 0; i < dimensions; i++) 
        {
            iss >> values[i];
        }
        out.write(reinterpret_cast<char*>(values.data()), dimensions * sizeof(float));

        vocabSize++;
    }

    // Go back and update the vocabulary size at the beginning of the file
    out.seekp(0);
    out.write(reinterpret_cast<char*>(&vocabSize), sizeof(size_t));

    std::cout << "Converted " << vocabSize << " words to binary format." << std::endl;

	return 0;
}