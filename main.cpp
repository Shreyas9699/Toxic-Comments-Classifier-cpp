#include <fstream>
#include "header/DataProcessor.h"
#include "header/Timer.h"

std::unordered_map<std::string, std::string> parseFlags(int argc, char* argv[])
{
    std::unordered_map<std::string, std::string> M;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0)
        {
            if (arg == "--help")
            {
                M["help"] = "";
            }
            else if (i + 1 < argc && argv[i + 1][0] != '-') // Check if next arg isn't a flag
            {
                M[arg.substr(2)] = argv[i + 1];
                i++; // Skip the value in the next iteration
            }
            else
            {
                std::cerr << "Warning: Flag " << arg << " has no value." << std::endl;
            }
        }
    }
    return M;
}

int main(int argc, char* argv[])
{
    auto flags = parseFlags(argc, argv);

    if (flags.count("help"))
    {
        std::cout
            << "Usage: ./main [--train PATH] [--test PATH] [--glove PATH] "
            "[--epochs N] [--lr FLOAT] [--batch_size N] [--log FILE] "
            "[--hidden_layers N,N,...] [--activation TANH|RELU|SIGMOID] "
            "[--bias FLOAT] [--dropout FLOAT]\n"
            << "Defaults:\n"
            << "  --train         data/train_data.csv\n"
            << "  --test          data/test_data.csv\n"
            << "  --val_ratio     0.2\n"
            << "  --glove         data/glove6B/glove.6B.100d.bin\n"
            << "  --epochs        3\n"
            << "  --lr            0.01\n"
            << "  --batch_size    64\n"
            << "  --log           main.log\n"
            << "  --hidden_layers 64,32,16\n"
            << "  --activation    TANH\n"
            << "  --bias          1.0\n"
            << "  --dropout       0.2\n";
        return 0;
    }

    // Parse basic parameters
    std::string train_path  = flags.count("train") ? flags["train"] : "data/train_data.csv";
    std::string test_path   = flags.count("test") ? flags["test"] : "data/test_data.csv";
    std::string glove_path  = flags.count("glove") ? flags["glove"] : "data/glove6B/glove.6B.100d.bin";
    int epochs              = flags.count("epochs") ? std::stoi(flags["epochs"]) : 3;
    double lr               = flags.count("lr") ? std::stod(flags["lr"]) : 0.01;
    int batch_size          = flags.count("batch_size") ? std::stoi(flags["batch_size"]) : 64;
    std::string log_file    = flags.count("log") ? flags["log"] : "main.log";
    float validation_ratio  = flags.count("val_ratio") ? static_cast<float>(std::stof(flags["val_ratio"])) : 0.2f;
    
    // Parse model architecture parameters
    std::string hidden_layers_str = flags.count("hidden_layers") ? flags["hidden_layers"] : "64,32,16";
    std::string activation_str    = flags.count("activation") ? flags["activation"] : "TANH";
    float bias                    = flags.count("bias") ? std::stof(flags["bias"]) : 1.0f;
    float dropout                 = flags.count("dropout") ? std::stof(flags["dropout"]) : 0.2f;

    // Set up logging
    std::ofstream logFile(log_file);
    if (!logFile)
    {
        std::cerr << "Error opening log file." << std::endl;
        return 1;
    }

    // Parse hidden layer dimensions from string
    std::vector<size_t> hidden_layers;
    std::stringstream ss(hidden_layers_str);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        try
        {
            hidden_layers.push_back(std::stoi(item));
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error parsing hidden layer dimensions: " << e.what() << std::endl;
            return 1;
        }
    }

    // Parse activation function
    Activation::Type activation_type;
    if (activation_str == "RELU")
    {
        activation_type = Activation::Type::RELU;
    }
    else if (activation_str == "SIGMOID")
    {
        activation_type = Activation::Type::SIGMOID;
    }
    else if (activation_str == "LEAKY_RELU")
    {
        activation_type = Activation::Type::LEAKY_RELU;
    }
    else if (activation_str == "STEP")
    {
        activation_type = Activation::Type::STEP;
    }
    else if (activation_str == "TANH")
    {
        activation_type = Activation::Type::TANH;
    }
    else
    {
		std::cout << "Invalid activation function specified. Defaulting to TANH." << std::endl;
        activation_type = Activation::Type::TANH; // Default
    }

    // Display all parameters
    std::cout << "=== Running with the following parameters ===" << std::endl;
    std::cout << "Training data:    " << train_path << std::endl;
    std::cout << "Validation ratio: " << validation_ratio << std::endl;
    std::cout << "Test data:        " << test_path << std::endl;
    std::cout << "GloVe embedding:  " << glove_path << std::endl;
    std::cout << "Epochs:           " << epochs << std::endl;
    std::cout << "Learning rate:    " << lr << std::endl;
    std::cout << "Batch size:       " << batch_size << std::endl;
    std::cout << "Log file:         " << log_file << std::endl;
    std::cout << "Hidden layers:    " << hidden_layers_str << std::endl;
    std::cout << "Activation:       " << activation_str << std::endl;
    std::cout << "bias:             " << bias << std::endl;
    std::cout << "Dropout:          " << dropout << std::endl;
    std::cout << "=============================================" << std::endl;

    // Log the same information
    logFile << "=== Running with the following parameters ===" << std::endl;
    logFile << "Training data:    " << train_path << std::endl;
    logFile << "Test data:        " << test_path << std::endl;
    logFile << "Validation ratio: " << validation_ratio << std::endl;
    logFile << "GloVe embedding:  " << glove_path << std::endl;
    logFile << "Epochs:           " << epochs << std::endl;
    logFile << "Learning rate:    " << lr << std::endl;
    logFile << "Batch size:       " << batch_size << std::endl;
    logFile << "Log file:         " << log_file << std::endl;
    logFile << "Hidden layers:    " << hidden_layers_str << std::endl;
    logFile << "Activation:       " << activation_str << std::endl;
    logFile << "bias:             " << bias << std::endl;
    logFile << "Dropout:          " << dropout << std::endl;
    logFile << "=============================================" << std::endl;

    // Load GloVe embeddings
    auto embeddings = loadGloVeEmbeddingsBinary(glove_path);

    // Load and preprocess data
    auto all_data = loadTrainingData(train_path, embeddings);
    auto split_result = splitTrainValidation(all_data, validation_ratio);

    auto training_data = std::move(split_result.first);
    auto validation_data = std::move(split_result.second);

    all_data.clear();
    all_data.shrink_to_fit();

    std::cout << "Training data size: " << training_data.size() << std::endl;
    std::cout << "Validation data size: " << validation_data.size() << std::endl;

    logFile << "Training data size: " << training_data.size() << std::endl;
    logFile << "Validation data size: " << validation_data.size() << std::endl;


    // Define MLP structure
    size_t input_size = 100;       // Size of GloVe vector

    // Create complete layer structure (input + hidden + output)
    std::vector<size_t> layers;
    layers.push_back(input_size);  // Input layer
    for (size_t hidden_size : hidden_layers)
    {
        layers.push_back(hidden_size);
    }
    layers.push_back(1);           // Output layer

    // Log model architecture
    logFile << "=== Model architecture ===" << std::endl;
    logFile << "MLP layer sizes:   ";
    for (size_t size : layers) {
        logFile << size << " ";
    }
    logFile << std::endl;
    logFile << "Activation:        " << activation_str << std::endl;
    logFile << "bias:              " << bias << std::endl;
    logFile << "Learning rate:     " << lr << std::endl;
    logFile << "Dropout rate:      " << dropout << std::endl;
    logFile << "===========================" << std::endl;

    // Initialize MLP with parameters from command line
    MultilayerPerceptron mlp(layers, activation_type, bias, static_cast<float>(lr), dropout);
    mlp.printStructure();
    mlp.printStructure(logFile);

    std::cout << "Started Training the model" << std::endl;
    logFile << "Started Training the model" << std::endl;

    // Training loop with validation
    std::vector<float> training_losses;
    std::vector<float> validation_losses;
    std::vector<float> training_accuracies;
    std::vector<float> validation_accuracies;

    mlp.setTrainingMode(true);
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        std::string epoch_str = "Epoch " + std::to_string(epoch + 1);
        Timer t(epoch_str);

        // Training phase
        float epoch_loss = 0.0f;
        int correct_train = 0;
        std::vector<std::vector<float>> batch_features;
        std::vector<float> batch_labels;

        // Shuffle training data at the beginning of each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(training_data.begin(), training_data.end(), g);

        for (size_t i = 0; i < training_data.size(); ++i)
        {
            batch_features.push_back(training_data[i].first);
            batch_labels.push_back(static_cast<float>(training_data[i].second));

            // Run prediction for accuracy calculation
            std::vector<float> predicted_output = mlp.run(training_data[i].first);
            int predicted_label = predicted_output[0] > 0.5 ? 1 : 0;
            if (predicted_label == training_data[i].second)
            {
                correct_train++;
            }

            // If the batch is full or it's the last iteration
            if (batch_features.size() == batch_size || i == training_data.size() - 1)
            {
                // Perform backpropagation for the current batch
                for (size_t j = 0; j < batch_features.size(); ++j)
                {
                    float loss = mlp.backPropagation({ batch_features[j] }, { batch_labels[j] });
                    epoch_loss += loss;
                }
                batch_features.clear();
                batch_labels.clear();
            }
        }

        epoch_loss /= training_data.size();
        float train_accuracy = static_cast<float>(correct_train) / training_data.size();
        training_losses.push_back(epoch_loss);
        training_accuracies.push_back(train_accuracy);

        // Validation phase (no backpropagation)
        mlp.setTrainingMode(false);
        float val_loss = 0.0f;
        int correct_val = 0;

        for (const auto& data_point : validation_data)
        {
            std::vector<float> predicted_output = mlp.run(data_point.first);
            int predicted_label = predicted_output[0] > 0.5 ? 1 : 0;

            // calculate BCE loss
            float label = static_cast<float>(data_point.second);
            float pred = predicted_output[0];
            float sample_loss = -(label * log(pred + 1e-8f) + (1.0f - label) * log(1.0f - pred + 1e-8f));
            val_loss += sample_loss;

            if (predicted_label == data_point.second)
            {
                correct_val++;
            }
        }

        val_loss /= validation_data.size();
        float val_accuracy = static_cast<float>(correct_val) / validation_data.size();
        validation_losses.push_back(val_loss);
        validation_accuracies.push_back(val_accuracy);

        // Switch back to training mode for next epoch
        mlp.setTrainingMode(true);

        // Log metrics
        logFile << "Epoch " << epoch + 1 << " metrics:" << std::endl;
        logFile << "  Training Loss: " << epoch_loss << ", Training Accuracy: " << train_accuracy * 100.0f << "%" << std::endl;
        logFile << "  Validation Loss: " << val_loss << ", Validation Accuracy: " << val_accuracy * 100.0f << "%" << std::endl;

        // Print to console
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
            << " - Loss: " << epoch_loss
            << " - Accuracy: " << train_accuracy * 100.0f << "%"
            << " - Val Loss: " << val_loss
            << " - Val Accuracy: " << val_accuracy * 100.0f << "%"
            << std::endl;


        logFile << "Weights after epoch " << epoch + 1 << ":\n";
        mlp.printWeights(logFile);
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
    predictTestData(test_path, embeddings, mlp, logFile, 10);

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