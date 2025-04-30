#define NOMINMAX  // Add this line before including any Windows headers  
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <unordered_set>
#include <mutex>
#include "DataProcessor.h"
#include "Timer.h"


// Function to load GloVe embeddings
std::unordered_map<std::string, std::vector<float>> loadGloVeEmbeddings(const std::string& filepath) 
{
    Timer t("Loading Glove Embeddings using loadGloVeEmbeddings");
    std::cout << "Loading GloVe Embeddings [loadGloVeEmbeddings fn]" << std::endl;

    size_t embedding_dim = 100;
    size_t reserve_vocab = 400000;
    std::unordered_map<std::string, std::vector<float>> embeddings;
    embeddings.reserve(400000);
    std::ifstream file(filepath);

    if (!file) 
    {
        std::cerr << "Cannot open " << filepath << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(file, line)) 
    {
        std::istringstream ss(line);
        std::string word;
        ss >> word;

        std::vector<float> vec;
        float value;
        while (ss >> value) 
        {
            vec.push_back(value);
        }
        embeddings[word] = vec;
    }

    // Debugging: Check number of loaded embeddings
    std::cout << "Number of embeddings: " << embeddings.size() << std::endl; // Print the number of loaded embeddings

    return embeddings;
}

// Load embeddings from binary format (much faster) but need file to be converted into binary file
std::unordered_map<std::string, std::vector<float>> loadGloVeEmbeddingsBinary(const std::string& filename) 
{
    Timer t("Loading Glove Embeddings using loadGloVeEmbeddingsBinary");
    std::cout << "Loading GloVe Embeddings using binary file" << std::endl;

    std::unordered_map<std::string, std::vector<float>> embeddings;
    std::ifstream in(filename, std::ios::binary);

    if (!in.is_open()) 
    {
        std::cerr << "Cannot open binary file: " << filename << std::endl;
        return embeddings;
    }

    // Read vocabulary size from header
    size_t vocabSize;
    in.read(reinterpret_cast<char*>(&vocabSize), sizeof(size_t));

    // Read dimensions from header
    int dimensions;
    in.read(reinterpret_cast<char*>(&dimensions), sizeof(int));

    // Reserve space for better performance
    embeddings.reserve(vocabSize);

    // Read each word and its embedding
    for (size_t i = 0; i < vocabSize; i++) 
    {
        // Read word length
        int length;
        in.read(reinterpret_cast<char*>(&length), sizeof(int));

        // Read word
        std::string word(length, ' ');
        in.read(&word[0], length);

        // Read embedding
        std::vector<float> embedding(dimensions);
        in.read(reinterpret_cast<char*>(embedding.data()), dimensions * sizeof(float));

        embeddings[word] = embedding;
    }
    std::cout << "Number of embeddings: " << embeddings.size() << std::endl; // Print the number of loaded embeddings
    return embeddings;
}

// Function to preprocess comments and convert them to feature vectors using GloVe
std::vector<float> preprocessComment(const std::string& comment_text, const std::unordered_map<std::string, std::vector<float>>& embeddings)
{
    static std::mutex logMutex;
    static std::unordered_set<std::string> missingWordsGlobal;
    static bool logFileInitialized = false;

    std::vector<float> features(100, 0.0); // Assuming 100 dimensions for GloVe vectors

    std::string processed_comment = comment_text;
    // Convert to lowercase
    std::transform(processed_comment.begin(), processed_comment.end(), processed_comment.begin(), ::tolower);
    // Remove punctuation
    processed_comment.erase(std::remove_if(processed_comment.begin(), processed_comment.end(),
        [](unsigned char c) { return std::ispunct(c); }),
        processed_comment.end());

    std::istringstream processed_ss(processed_comment);
    std::string word;
    size_t count = 0;

    // Local set to store missing words from this comment
    std::unordered_set<std::string> missingWordsLocal;

    while (processed_ss >> word)
    {
        // Skip empty words
        if (word.empty()) continue;

        auto it = embeddings.find(word);
        if (it != embeddings.end())
        {
            const auto& vec = it->second;
            for (size_t i = 0; i < std::min(features.size(), vec.size()); ++i)
            {
                features[i] += vec[i];
            }
            count++;
        }
        else
        {
            // Add to local missing words set
            missingWordsLocal.insert(word);
        }
    }

    // Average the feature vector
    if (count > 0)
    {
        float invCount = 1.0f / count;
        for (size_t i = 0; i < features.size(); ++i)
        {
            features[i] *= invCount;
        }
    }

    // Add missing words to global set with mutex protection
    if (!missingWordsLocal.empty()) 
    {
        std::lock_guard<std::mutex> lock(logMutex);
        missingWordsGlobal.insert(missingWordsLocal.begin(), missingWordsLocal.end());

        // Initialize log file if this is the first time
        // Using a delayed initialization pattern to avoid opening the file 
        // if no missing words are encountered
        if (!logFileInitialized) 
        {
            std::ofstream logFile("wordsMissingFromGlove.log", std::ios::out | std::ios::trunc);
            if (logFile) 
            {
                logFileInitialized = true;
                logFile << "Words missing from GloVe embeddings will be logged here." << std::endl;
                logFile.close();
            }
        }

        // Periodically flush the missing words to the log file
        // This is a compromise between writing too frequently and losing data if the program crashes
        if (missingWordsGlobal.size() % 1000 == 0) 
        {
            std::ofstream logFile("wordsMissingFromGlove.log", std::ios::out | std::ios::app);
            if (logFile) {
                for (const auto& word : missingWordsLocal) 
                {
                    logFile << "Word not found in embeddings: " << word << std::endl;
                }
                logFile.close();
            }
        }
    }

    return features;
}

// spliting training data into training and validation
std::pair<std::vector<std::pair<std::vector<float>, int>>, 
          std::vector<std::pair<std::vector<float>, int>>> 
splitTrainValidation(const std::vector<std::pair<std::vector<float>, int>>& data, float validation_ratio) 
{
    Timer t("Split training data into training and validation");
    // First shuffle the data to ensure random split
    std::vector<std::pair<std::vector<float>, int>> shuffled_data = data;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);
    
    // Calculate split point
    size_t validation_size = static_cast<size_t>(shuffled_data.size() * validation_ratio);
    size_t training_size = shuffled_data.size() - validation_size;
    
    // Create training and validation sets
    std::vector<std::pair<std::vector<float>, int>> training_data(
        shuffled_data.begin(), 
        shuffled_data.begin() + training_size
    );
    
    std::vector<std::pair<std::vector<float>, int>> validation_data(
        shuffled_data.begin() + training_size, 
        shuffled_data.end()
    );
    
    return {training_data, validation_data};
}

// Function to load training data with optimizations
std::vector<std::pair<std::vector<float>, int>> loadTrainingData(const std::string& filepath, std::unordered_map<std::string, std::vector<float>>& embeddings)
{
    Timer t("Loading training dataset");
    std::cout << "Loading training dataset" << std::endl;

    // Read the entire file into memory first
    std::vector<std::string> lines;
    {
        std::ifstream file(filepath);
        std::string line;
        std::getline(file, line); // Skip header

        while (std::getline(file, line)) 
        {
            lines.push_back(std::move(line));
        }
    }

    std::cout << "Read " << lines.size() << " lines from file" << std::endl;

    std::vector<std::string> toxic_comments;
    std::vector<std::string> non_toxic_comments;

    // Pre-allocate the training data vector
    std::vector<std::pair<std::vector<float>, int>> training_data;
    training_data.reserve(lines.size());

    // Process data in batches for better cache locality
    const size_t BATCH_SIZE = 1000;

    for (size_t i = 0; i < lines.size(); i += BATCH_SIZE) 
    {
        size_t end = std::min(i + BATCH_SIZE, lines.size());

        // Process this batch
        for (size_t j = i; j < end; ++j) 
        {
            std::istringstream ss(lines[j]);
            std::string comment_text;
            int toxic_flag;

            std::getline(ss, comment_text, ','); // Assuming comma-separated values
            ss >> toxic_flag;

            // Process the comment and add to training data
            training_data.emplace_back(
                preprocessComment(comment_text, embeddings),
                toxic_flag
            );
        }
    }

    std::cout << "Total training dataset size is: " << training_data.size() << std::endl;

    return training_data;
}

// Function to load test data, predict toxicity, and print limited samples
void predictTestData(const std::string& testFilePath, const std::unordered_map<std::string, std::vector<float>>& embeddings, MultilayerPerceptron& mlp, std::ofstream& logFile, size_t print_limit) 
{
    Timer t("Model predection on test dataset");
	std::cout << "Model predection on test dataset" << std::endl;
	logFile << "Model predection on test dataset" << std::endl;

    std::ifstream testFile(testFilePath);
    if (!testFile) 
    {
        std::cerr << "Error opening test file." << std::endl;
        return;
    }

    std::string line;
    std::getline(testFile, line); // Read and discard the header line

    std::vector<std::pair<std::string, int>> all_predictions; // Store all predictions as (comment, label)
    std::vector<std::string> non_toxic_samples;
    std::vector<std::string> toxic_samples;

    size_t half_limit = print_limit / 2;

    // Process each comment in the test data
    while (std::getline(testFile, line)) 
    {
        // Preprocess the comment
        std::vector<float> features = preprocessComment(line, embeddings);

        // Predict using the model
        std::vector<float> predicted_output = mlp.run(features);
        int predicted_label = predicted_output[0] > 0.5 ? 1 : 0; // Assuming output is between 0 and 1

        // Store the comment and its prediction
        all_predictions.push_back({line, predicted_label});

        // Log the prediction
        logFile << "Test comment: " << line << std::endl;
        logFile << "Prediction: " << (predicted_label == 1 ? "Toxic" : "Non-toxic") << std::endl;
    }

    testFile.close();
}
