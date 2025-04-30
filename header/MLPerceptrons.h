#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <cmath>
#include <time.h>
#include <iostream>

enum class ActivationType
{
    SIGMOID,
    TANH,
    RELU,
	LEAKY_RELU,
    STEP
};

using ActFn = float(*)(float);
using DerivFn = float(*)(float);

struct Activation
{
    ActFn fn;
    DerivFn dfn;
};

Activation getActivation(ActivationType t);

class Perceptron
{
public:
    std::vector<float> weights;
    float bias;
    ActivationType aType;
    ActFn activate;
    DerivFn derivative;

    Perceptron(size_t inputs, ActivationType type = ActivationType::SIGMOID, float bias = 1.0);

    float run(std::vector<float> x);
    void set_weights(std::vector<float> w_init);
};

class MultilayerPerceptron
{
private:
    std::vector<size_t> layers;
    float bias;
    float eta;                                     // learning rate
    ActivationType actType;                        // Store activation type for the network
    float dropout_rate;
    bool is_training;
    std::vector<std::vector<Perceptron>> network;  // this is the hidden network layer
    std::vector<std::vector<float>> values;        // to store the output values of the network layer
    std::vector<std::vector<float>> d;             // to store the error

    void set_weights(std::vector< std::vector< std::vector<float>>> w_init);
    void clipGradient(float& gradient, float threshold);
    static std::string activationName(ActivationType t);

public:
    MultilayerPerceptron(std::vector<size_t> layers, ActivationType type = ActivationType::SIGMOID, float bias = 1.0, float eta = 0.5, float dropout = 0.0f);
    void printStructure(std::ostream& out = std::cout) const;
    void printWeights();
    void printWeights(std::ofstream& logFile);
    std::vector<float> run(std::vector<float> x);
    float backPropagation(std::vector<float> x, std::vector<float> y);
    
    void setTrainingMode(bool training_mode);
    void setDropoutRate(float rate);
};