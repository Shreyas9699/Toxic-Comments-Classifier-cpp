#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <time.h>
#include <iostream>

#include "Activation.h"

class Perceptron
{
public:
    std::vector<float> weights;
    float bias;
    Activation activation;

    Perceptron(size_t inputs, Activation::Type type = Activation::Type::SIGMOID, float bias = 1.0);

    float run(std::vector<float> x);
    void set_weights(std::vector<float> w_init);
};

class MultilayerPerceptron
{
private:
    std::vector<size_t> layers;
    float bias;
    float eta;                                     // learning rate
    Activation::Type actType;                        // Store activation type for the network
    float dropout_rate;
    bool is_training;
    std::vector<std::vector<Perceptron>> network;  // this is the hidden network layer
    std::vector<std::vector<float>> values;        // to store the output values of the network layer
    std::vector<std::vector<float>> d;             // to store the error

    void set_weights(std::vector< std::vector< std::vector<float>>> w_init);
    void clipGradient(float& gradient, float threshold);

public:
    MultilayerPerceptron(std::vector<size_t> layers, Activation::Type type = Activation::Type::SIGMOID, float bias = 1.0, float eta = 0.5, float dropout = 0.0f);
    void printStructure(std::ostream& out = std::cout) const;
    void printWeights();
    void printWeights(std::ofstream& logFile);
    std::vector<float> run(std::vector<float> x);
    float backPropagation(std::vector<float> x, std::vector<float> y);

    void setTrainingMode(bool training_mode);
    void setDropoutRate(float rate);
};