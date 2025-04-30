#include <fstream>
#include <chrono>
#include "header/DataProcessor.h"
#include "header/Timer.h"

int main() 
{
    std::ofstream logFile("main.log");
    if (!logFile) 
    {
        std::cerr << "Error opening log file." << std::endl;
        return 1;
    }

    // Load GloVe embeddings
    auto embeddings = loadGloVeEmbeddingsBinary("data/glove6B/glove.6B.100d.bin");
    // auto embeddings = loadGloVeEmbeddingsMMap("data/glove6B/glove.6B.100d.txt"); // reduced from ~3m to 53s

    // Load and preprocess data
    std::vector<std::pair<std::vector<float>, int>> training_data = loadTrainingData("data/train_data.csv", embeddings);

    // Define MLP structure
    size_t input_size = 100;       // Size of GloVe vector
    size_t num_epochs = 3;         // Number of training epochs
    size_t batch_size = 32;        // Define your desired batch size
	ActivationType activation_type = ActivationType::TANH;
	std::vector<size_t> layers = { input_size, 64, 32, 16, 1 }; // Input, Hidden, Output {input_size, hidden_layer_size, 1} 

    MultilayerPerceptron mlp(layers, activation_type, 1.0f, 0.01f, 0.2f);
    mlp.printStructure();
	mlp.printStructure(logFile);

    std::cout << "Started Training the model" << std::endl;
    logFile << "Started Training the model" << std::endl;

    mlp.setTrainingMode(true);
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) 
    {
		std::string epoch_str = "Epoch " + std::to_string(epoch + 1);
        Timer t(epoch_str);

        std::vector<std::vector<float>> batch_features;
        std::vector<float> batch_labels;

        for (size_t i = 0; i < training_data.size(); ++i) 
        {
            batch_features.push_back(training_data[i].first);
            batch_labels.push_back(static_cast<float>(training_data[i].second));

            // If the batch is full or it's the last iteration
            if (batch_features.size() == batch_size || i == training_data.size() - 1) 
            {
                // Perform backpropagation for the current batch
                for (size_t j = 0; j < batch_features.size(); ++j) 
                {
                    mlp.backPropagation({batch_features[j]}, std::vector<float>{ batch_labels[j] }); // Call backpropagation for each sample
                }
                batch_features.clear();
                batch_labels.clear();
            }
        }

        logFile << "Weights after epoch " << epoch + 1 << ":\n";
        mlp.printWeights(logFile);
        logFile << "Epoch " << epoch + 1 << " completed." << std::endl; // Optional: Print progress
    }
    mlp.setTrainingMode(false);
    std::cout << "Training Completed!" << std::endl;
    logFile << "Training Completed!" << std::endl;

    // Calculate training accuracy
    std::cout << "Calculate training accuracy" << std::endl;
    logFile << "Calculate training accuracy" << std::endl;
    int correct_predictions = 0;
    for (const auto& data_point : training_data) 
    {
        std::vector<float> predicted_output = mlp.run(data_point.first);
        int predicted_label = predicted_output[0] > 0.5 ? 1 : 0;
        if (predicted_label == data_point.second) 
        {
            correct_predictions++;
        }
    }

    float training_accuracy = static_cast<float>(correct_predictions) / training_data.size();
    std::cout << "Training Accuracy: " << training_accuracy * 100.0 << "%" << std::endl;
    logFile << "Training Accuracy: " << training_accuracy * 100.0 << "%" << std::endl;

    // Predict on test data
    std::cout << "Predicting test data" << std::endl;
    logFile << "Predicting test data" << std::endl;
    predictTestData("data/test_data.csv", embeddings, mlp, logFile, 10);

    // Continuous input for testing the model
    std::string input_comment;
    std::cout << "\nEnter comments to check for toxicity (press Enter twice to exit):" << std::endl;
    
    while (true) 
    {
        std::getline(std::cin, input_comment); // Read a line of input

        logFile << "Input comment: " << input_comment << std::endl;
        
        // Check for exit condition (double Enter)
        if (input_comment.empty()) 
        {
            break;
        }

        // Preprocess the comment
        std::vector<float> features = preprocessComment(input_comment, embeddings);

        // Predict using the model
        std::vector<float> predicted_output = mlp.run(features);
        int predicted_label = predicted_output[0] > 0.5 ? 1 : 0;

        // Output the result
        if (predicted_label == 1) 
        {
            logFile << "The comment is TOXIC." << std::endl;
            std::cout << "The comment is TOXIC." << std::endl;
        }
        else 
        {
            logFile << "The comment is NOT TOXIC." << std::endl;
            std::cout << "The comment is NOT TOXIC." << std::endl;
        }
    }

    logFile << "Exiting the program." << std::endl;
    logFile.close();
    std::cout << "Exiting the program." << std::endl;
    return 0;
}