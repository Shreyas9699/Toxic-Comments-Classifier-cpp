#include <fstream>
#include "header/DataProcessor.h"

int main() {
    // Open log file for output
    std::ofstream logFile("main.log"); // Specify the log file name here
    if (!logFile) {
        std::cerr << "Error opening log file." << std::endl;
        return 1; // Exit if file can't be opened
    }

    // Load GloVe embeddings
    auto embeddings = loadGloVeEmbeddings("data/glove.6B.100d.txt"); // Update path as needed

    // Load and preprocess data
    std::vector<std::pair<std::vector<double>, int>> training_data = loadTrainingData("data/train_data.csv", embeddings);

    // Define MLP structure
    size_t input_size = 100;       // Size of GloVe vector
    size_t hidden_layer_size = 10; // Number of neurons in the hidden layer
    size_t num_epochs = 3;         // Number of training epochs
    size_t batch_size = 32;        // Define your desired batch size
    std::vector<size_t> layers = {input_size, hidden_layer_size, 1}; // Input, Hidden, Output

    MultilayerPerceptron mlp(layers);

    // Train the model
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::vector<std::vector<double>> batch_features;
        std::vector<double> batch_labels;

        for (size_t i = 0; i < training_data.size(); ++i) {
            batch_features.push_back(training_data[i].first);
            batch_labels.push_back(static_cast<double>(training_data[i].second));

            // If the batch is full or it's the last iteration
            if (batch_features.size() == batch_size || i == training_data.size() - 1) {
                // Perform backpropagation for the current batch
                for (size_t j = 0; j < batch_features.size(); ++j) {
                    mlp.backPropagation({batch_features[j]}, {batch_labels[j]}); // Call backpropagation for each sample
                }

                // Clear the batch
                batch_features.clear();
                batch_labels.clear();
            }
        }

        // Print weights after each epoch to the log file
        logFile << "Weights after epoch " << epoch + 1 << ":\n";
        mlp.printWeights(logFile); // Assuming you modify printWeights to accept an ofstream
        logFile << "Epoch " << epoch + 1 << " completed." << std::endl; // Optional: Print progress
    }

    // Calculate training accuracy
    int correct_predictions = 0;
    for (const auto& data_point : training_data) {
        std::vector<double> predicted_output = mlp.run(data_point.first);
        int predicted_label = predicted_output[0] > 0.5 ? 1 : 0; // Assuming binary classification
        if (predicted_label == data_point.second) {
            correct_predictions++;
        }
    }

    double training_accuracy = static_cast<double>(correct_predictions) / training_data.size();
    std::cout << "Training Accuracy: " << training_accuracy * 100.0 << "%" << std::endl;
    logFile << "Training Accuracy: " << training_accuracy * 100.0 << "%" << std::endl;

    // Predict on test data
    predictTestData("data/test_data.csv", embeddings, mlp, logFile, 10);

    // Continuous input for testing the model
    std::string input_comment;
    std::cout << "\nEnter comments to check for toxicity (press Enter twice to exit):" << std::endl;
    
    while (true) {
        std::getline(std::cin, input_comment); // Read a line of input

        logFile << "Input comment: " << input_comment << std::endl;
        
        // Check for exit condition (double Enter)
        if (input_comment.empty()) {
            break; // Exit the loop if Enter is pressed twice
        }

        // Preprocess the comment
        std::vector<double> features = preprocessComment(input_comment, embeddings);

        // Predict using the model
        std::vector<double> predicted_output = mlp.run(features);
        int predicted_label = predicted_output[0] > 0.5 ? 1 : 0; // Assuming output is between 0 and 1

        // Output the result
        if (predicted_label == 1) {
            logFile << "The comment is TOXIC." << std::endl;
            std::cout << "The comment is TOXIC." << std::endl;
        } else {
            logFile << "The comment is NOT TOXIC." << std::endl;
            std::cout << "The comment is not TOXIC." << std::endl;
        }
    }

    logFile << "Exiting the program." << std::endl;
    logFile.close(); // Close the log file
    std::cout << "Exiting the program." << std::endl;
    return 0;
}