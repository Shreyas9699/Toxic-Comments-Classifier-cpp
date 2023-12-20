#pragma once
#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <cmath>
#include <time.h>

class Perceptron {
    public:
        std::vector<double> weights;
        double bias;

        Perceptron (size_t inputs, double bias = 1.0);

        double run (std::vector<double> x);

        void set_weights(std::vector<double> w_init);

        double sigmoid (double x);
};

class MultilayerPerceptron {
    public:
        MultilayerPerceptron (std::vector<size_t> layers, double bias = 1.0, double eta = 0.5);
        void set_weights(std::vector< std::vector< std::vector<double>>> w_init);
        void printWeights();
        std::vector<double> run(std::vector<double> x);
        double backPropagation(std::vector<double> x, std::vector<double> y);


        std::vector<size_t> layers;
        double bias;
        double eta;                                     // learning rate
        std::vector<std::vector<Perceptron>> network;   // this is the hidden network layer
        std::vector<std::vector<double>> values;        // to store the output values of the network layer
        std::vector<std::vector<double>> d;             // to store the error
};

