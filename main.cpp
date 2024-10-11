#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "header/csvReader.h"
#include "header/MLPerceptrons.h"
#include "header/dataPreProcessor.h"
#include "header/Tokenizer.h"

// Thread-safe queue for batch processing
class ThreadSafeQueue 
{
private:
    std::queue<std::vector<std::string>> queue;
    std::mutex mutex;
    std::condition_variable cond;
    bool finished = false;

public:
    void push(const std::vector<std::string>& value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(value);
        cond.notify_all();
    }

    bool pop(std::vector<std::string>& value) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] { return !queue.empty() || finished; });
        if (queue.empty() && finished) return false;
        value = queue.front();
        queue.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(mutex);
        finished = true;
        cond.notify_all();
    }
};

// Function to process a batch of data
void processBatch(ThreadSafeQueue& dataQueue, MultilayerPerceptron& mlpModel,
                  std::vector<std::vector<double>>& embeddingMatrix,
                  std::vector<double>& yTrainData, double& totalMSE, 
                  int& processedSamples, std::mutex& mse_mutex) 
                  {
    std::vector<std::string> batch;
    while (dataQueue.pop(batch)) 
    {
        std::vector<std::vector<std::string>> tokenizedBatch = tokenizeData(batch);
        std::vector<std::vector<int>> numericBatch = tokenizeAndNumberizeData(tokenizedBatch);
        padData(numericBatch, 400, wordToIndex["<PAD>"]);

        double batchMSE = 0.0;
        for (size_t i = 0; i < numericBatch.size(); ++i) 
        {
            std::vector<double> input;
            for (int index : numericBatch[i]) 
            {
                input.insert(input.end(), embeddingMatrix[index].begin(), embeddingMatrix[index].end());
            }
            batchMSE += mlpModel.backPropagation(input, {yTrainData[processedSamples + i]});
        }

        {
            std::lock_guard<std::mutex> lock(mse_mutex);
            totalMSE += batchMSE;
            processedSamples += numericBatch.size();
        }
    }
}

int main () {
    std::pair<std::vector<std::string>, std::vector<double>> data = readCSV("data/train_data.csv");
    size_t dataSize = data.first.size();
    std::cout << "Train data set size: " << dataSize << std::endl;
    std::cout << "Completed reading and loading the train csv data \n";
    size_t toxicCommentsCount = std::count(data.second.begin(), data.second.end(), 1.0);
    std::cout << "Toxic comments: " << toxicCommentsCount << "\n";
    std::cout << "Non-toxic comments: " << dataSize - toxicCommentsCount << "\n";

    std::cout << "Balancing the data set by taking only 20K non toxic comments as they are a lot more when compared against the toxic\n";

    std::vector<std::string> commentText;
    std::vector<double> toxicLable;

    commentText.reserve(dataSize);
    toxicLable.reserve(dataSize);

    size_t nonToxicCount = 0;
    for (size_t i = 0; i < dataSize; i++)
    {
        const std::string& comment = data.first[i];
        const double& label = data.second[i];
        if (label == 1.0 || (label == 0.0 && nonToxicCount < 20000))
        {
            commentText.emplace_back(comment);
            toxicLable.emplace_back(label);
            if (label == 0.0) nonToxicCount++;
        }
    }

    data.first.clear();
    data.second.clear();
    std::pair<std::vector<std::string>, std::vector<double>> trainData = std::make_pair(std::move(commentText), std::move(toxicLable));
    std::cout << trainData.first.size() << std::endl;
	
    //MultilayerPerceptron mlpModel({100, 66, 33, 1});

    // // Tokenize
    // std::vector<std::vector<std::string>> tokenizedData = tokenizeData(xTrainData);

    //     // Create vocabulary
    // std::set<std::string> vocabulary = createVocabulary(tokenizedData);
    // std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;
    // createWordIdx(vocabulary);
    // std::vector<std::vector<int>> numData = tokenizeAndNumberizeData(tokenizedData);

    // // Pad
    // padData(numData, 400, wordToIndex["<PAD>"]);  // Adjust maxSequenceLength and paddingToken as needed
        
    // // Create embedding matrix
    // std::vector<std::vector<double>> embeddingMatrix = createEmbeddingMatrix(wordToIndex.size(), 50);  // Adjust embeddingDimension
    // std::cout << "Embedding matrix dimensions: " << 
    //             embeddingMatrix.size() << " x " << embeddingMatrix[0].size() << std::endl;

    // int epochs = 800;
    // double MSE;
    // int batchSize = 100;
    // ThreadSafeQueue dataQueue;
    // std::mutex mse_mutex;
    // int processedSamples = 0;

    // for (int i = 0; i < epochs; i++) {
    //     MSE = 0.0;
    //     processedSamples = 0;

    //     // Create batches
    //     for (int j = 0; j < xTrainData.size(); j += batchSize) {
    //         std::vector<std::string> batch(xTrainData.begin() + j, xTrainData.begin() + j + batchSize);
    //         dataQueue.push(batch);
    //     }

    //     dataQueue.setFinished(); // Call setFinished after pushing all data

    //     // Process batches in parallel
    //     std::vector<std::thread> threads;
    //     for (int j = 0; j < 4; j++) {
    //         threads.push_back(std::thread(processBatch, std::ref(dataQueue), std::ref(mlpModel), std::ref(embeddingMatrix), std::ref(yTrainData), std::ref(MSE), std::ref(processedSamples), std::ref(mse_mutex)));
    //     }

    //     // Wait for all threads to finish
    //     for (auto& thread : threads) {
    //         thread.join();
    //     }

    //     // Print the results
    //     if ( i % 100 == 0) 
    //     {
    //         std::cout << i << "th run MSE is : " << MSE << std::endl;
    //     }
    // }
    // MSE /= 2.0;            // number of different ouputs
    // std::cout << "MSE :" << MSE << std::endl;
    // // mlpModel.printWeights();

    // // Classifier tester
	// std::string inputString;

	// while (true) {
	// 	std::cout << "Enter a string (or press Enter twice to quit): ";
	// 	getline(std::cin, inputString);

	// 	if (inputString.empty()) {
	// 		// Check for two consecutive empty inputs
	// 		std::cout << "Enter another string to confirm quitting (or press Enter again to quit): ";
	// 		getline(std::cin, inputString);

	// 		if (inputString.empty()) {
    //            break; // Exit the loop
	// 		} else {
	// 			// Process the input string as needed
	// 			// (Replace this with your actual processing code)
	// 			std::cout << "Processing string: " << inputString << std::endl;
	// 		}
	// 	}
		
	// 	std::vector<std::string> testData = {inputString};
		
	// 	// Preprocess test string
    //     std::vector<std::vector<std::string>> testTokens = tokenizeData(testData);
    //     std::vector<int> testIndices = tokenizeAndNumberizeData(testTokens)[0];
    //     std::vector<std::vector<int>>padd = {testIndices};
    //     padData(padd, 400, wordToIndex["<PAD>"]);

    //     // Generate embeddings
    //     std::vector<double> testInput;
    //     for (int index : testIndices) {
    //         if (index == wordToIndex["<PAD>"]) {
    //         testInput.insert(testInput.end(), embeddingMatrix[0].begin(), embeddingMatrix[0].end());
    //         } else {
    //         testInput.insert(testInput.end(), embeddingMatrix[index].begin(), embeddingMatrix[index].end());
    //         }
    //     }

    //     // Run model
    //     std::vector<double> output = mlpModel.run(testInput);
    //     double toxicityScore = output[0];

    //     // Interpret output
    //     if (toxicityScore >= 0.5) {
    //         std::cout << "Test string is likely toxic." << std::endl;
    //     } else {
    //         std::cout << "Test string is likely non-toxic." << std::endl;
    //     }
	// }

	return 0;
}