#include "DataProcessor.h"

// Function to load GloVe embeddings
std::unordered_map<std::string, std::vector<double>> loadGloVeEmbeddings(const std::string& filepath) 
{
    std::unordered_map<std::string, std::vector<double>> embeddings;
    std::ifstream file(filepath);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string word;
        ss >> word;

        std::vector<double> vec;
        double value;
        while (ss >> value) {
            vec.push_back(value);
        }
        embeddings[word] = vec;
    }

    // Debugging: Check number of loaded embeddings
    std::cout << "Loaded GloVe Embeddings" << std::endl;
    std::cout << "Number of embeddings: " << embeddings.size() << std::endl; // Print the number of loaded embeddings

    return embeddings;
}

// Function to preprocess comments and convert them to feature vectors using GloVe
std::vector<double> preprocessComment(const std::string& comment_text, const std::unordered_map<std::string, std::vector<double>>& embeddings) 
{
    // Open log file for output
    // std::ofstream logFile("wordsMissingFromGlove.log"); // Specify the log file name here
    // if (!logFile) {
    //     std::cerr << "Error opening log file." << std::endl;
    //     exit; // Exit if file can't be opened
    // }
    std::vector<double> features(100, 0.0); // Assuming 100 dimensions for GloVe vectors
    std::istringstream ss(comment_text);
    std::string word;
    size_t count = 0;

    std::string processed_comment = comment_text;
    // Convert to lowercase
    std::transform(processed_comment.begin(), processed_comment.end(), processed_comment.begin(), ::tolower);
    // Remove punctuation
    processed_comment.erase(std::remove_if(processed_comment.begin(), processed_comment.end(),
                                            [](unsigned char c) { return std::ispunct(c); }),
                                            processed_comment.end());

    std::istringstream processed_ss(processed_comment); // Use processed comment for further processing

    while (processed_ss >> word) 
    {
        // Trim whitespace just in case
        word.erase(std::remove_if(word.begin(), word.end(), ::isspace), word.end());

        if (embeddings.find(word) != embeddings.end()) 
        {
            const auto& vec = embeddings.at(word);
            for (size_t i = 0; i < features.size(); ++i) 
            {
                features[i] += vec[i];
            }
            count++;
        } 
        else 
        {
            // enable the logger above to see the words missing from the glove
            // logFile<< "Word not found in embeddings: " << word << std::endl; // Log words not found
        }
    }

    // Average the feature vector
    if (count > 0) 
    {
        for (size_t i = 0; i < features.size(); ++i) 
        {
            features[i] /= count;
        }
    }

    // logFile.close(); 

    return features;
}

// Function to load training data
std::vector<std::pair<std::vector<double>, int>> loadTrainingData(const std::string& filepath, std::unordered_map<std::string, std::vector<double>>& embeddings) 
{
    int sampleSize = 2;
    std::vector<std::pair<std::vector<double>, int>> training_data;
    std::ifstream file(filepath);
    std::string line;

    std::getline(file, line); // Read and discard the header line

    // Temporary containers for toxic and non-toxic comments
    std::vector<std::string> toxic_comments;
    std::vector<std::string> non_toxic_comments;

    while (std::getline(file, line)) 
    {
        std::istringstream ss(line);
        std::string comment_text;
        int toxic_flag;

        std::getline(ss, comment_text, ','); // Assuming comma-separated values
        ss >> toxic_flag;

        training_data.emplace_back(preprocessComment(comment_text, embeddings), toxic_flag);

        // Store toxic and non-toxic comments
        if (toxic_flag == 1 && toxic_comments.size() < sampleSize) 
        {
            toxic_comments.push_back(comment_text);
        } 
        else if (toxic_flag == 0 && non_toxic_comments.size() < sampleSize) 
        {
            non_toxic_comments.push_back(comment_text);
        }
    }

    // Print some examples from the training data
    std::cout << "Non-Toxic Comments:" << std::endl;
    for (const auto& comment : non_toxic_comments) 
    {
        std::cout << "- " << comment << std::endl;
    }
    std::cout << "Toxic Comments:" << std::endl;
    for (const auto& comment : toxic_comments) 
    {
        std::cout << "- " << comment << std::endl;
    }

    std::cout << "Total training dataset size is: " << training_data.size() << std::endl;

    return training_data;
}

// Function to load test data, predict toxicity, and print limited samples
void predictTestData(const std::string& testFilePath, const std::unordered_map<std::string, std::vector<double>>& embeddings, MultilayerPerceptron& mlp, std::ofstream& logFile, size_t print_limit) 
{
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
        std::vector<double> features = preprocessComment(line, embeddings);

        // Predict using the model
        std::vector<double> predicted_output = mlp.run(features);
        int predicted_label = predicted_output[0] > 0.5 ? 1 : 0; // Assuming output is between 0 and 1

        // Store the comment and its prediction
        all_predictions.push_back({line, predicted_label});

        // Log the prediction
        // logFile << "Test comment: " << line << std::endl;
        // logFile << "Prediction: " << (predicted_label == 1 ? "Toxic" : "Non-toxic") << std::endl;
    }

    testFile.close();

    // Select samples for printing
    for (const auto& prediction : all_predictions) 
    {
        if (prediction.second == 1 && toxic_samples.size() < half_limit) 
        {
            toxic_samples.push_back(prediction.first);
        } 
        else if (prediction.second == 0 && non_toxic_samples.size() < half_limit) 
        {
            non_toxic_samples.push_back(prediction.first);
        }

        // Stop selecting if we've reached the print limit for both categories
        if (toxic_samples.size() == half_limit && non_toxic_samples.size() == half_limit) 
        {
            break;
        }
    }

    // Print the selected examples
    std::cout << "\nSample Predictions from Test Data:" << std::endl;

    std::cout << "Non-Toxic Samples:" << std::endl;
    for (const auto& comment : non_toxic_samples) 
    {
        std::cout << "> " << comment << "\n-> Prediction: Non-toxic" << std::endl;
    }

    std::cout << "\nToxic Samples:" << std::endl;
    for (const auto& comment : toxic_samples) 
    {
        std::cout << "> " << comment << "\n-> Prediction: Toxic" << std::endl;
    }
}
