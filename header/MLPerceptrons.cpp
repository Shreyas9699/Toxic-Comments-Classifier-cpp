#include "MLPerceptrons.h"

double frand () 
{
    return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

//constructor for Perceptron class
Perceptron::Perceptron (size_t inputs, double bias) 
{
    this->bias = bias;
    weights.resize(inputs + 1); // +1 since we have bias as well
    generate(weights.begin(), weights.end(), frand);
}

double Perceptron::run (std::vector<double> x) 
{
    x.push_back(bias); // push bias into the vector
    return sigmoid(inner_product(x.begin(), x.end(), weights.begin(), (double)0.0));
}

void Perceptron::set_weights (std::vector<double> w_init) 
{
    weights = w_init;
}

double Perceptron::sigmoid (double x) 
{
    return 1.0 / ( 1.0 + std::exp(-x));
}

MultilayerPerceptron::MultilayerPerceptron (std::vector<size_t> layers, double bias, double eta) 
{
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;

    // create neurons layer by layers
    for (size_t i = 0; i < layers.size(); i++) 
    {
        // to store output values for each neuron in layer i, initilised to 0
        values.push_back(std::vector<double> (layers[i], 0.0));
        d.push_back(std::vector<double> (layers[i], 0.0));
        network.push_back(std::vector<Perceptron> () ); // initially empty
        if (i > 0) { // network[0] is the input layers, so it has no neurons
            for (size_t j = 0; j < layers[i]; j++) 
            {
                network[i].push_back(Perceptron(layers[i - 1], bias)); // create same number of neurons as the previous layer
            }
        }
    }
}

void MultilayerPerceptron::set_weights (std::vector< std::vector< std::vector<double> > > w_init) 
{
    for (size_t i = 0; i < w_init.size(); i++) 
    { // to itr thorugh layers in the network
        for (size_t j = 0; j < w_init[i].size(); j++) 
        { // to itr thorugh neurons in the layer
            network[i + 1][j].set_weights(w_init[i][j]); // i + 1 -> to skip input layer
        }
    }
}

void MultilayerPerceptron::printWeights(std::ofstream& logFile)
{
    for (size_t i = 1; i < network.size(); i++) 
    {
        for (size_t j = 0; j < layers[i]; j++) 
        {
            logFile << "Layer " << i + 1 << " weights:\n";
            for (auto &itr: network[i][j].weights)
            {
                logFile << itr << "   ";
            }
            logFile << std::endl;
        }
    }
    logFile << std::endl;
}

std::vector<double> MultilayerPerceptron::run (std::vector<double> x) 
{
    values[0] = x; // input
    for (size_t i = 1; i < network.size(); i++) 
    { // excluding the input layer
        for (size_t j = 0; j < layers[i]; j++) 
        {
            values[i][j] = network[i][j].run(values[i - 1]);
        }
    }
    return values.back();
}

void clipGradient(double& gradient, double threshold) 
{
    if (gradient > threshold) gradient = threshold;
    if (gradient < -threshold) gradient = -threshold;
}

double MultilayerPerceptron::backPropagation(std::vector<double> x, std::vector<double> y) 
{
    // STEP 1: Feed a sample to the network
    std::vector<double> output = run(x);
    
    // STEP 2: Calculate Binary Cross-Entropy Loss instead of MSE
    double BCE = 0.0;
    std::vector<double> error;
    for (size_t i = 0; i < y.size(); i++) {
        double predicted = output[i];
        double actual = y[i];
        error.push_back(actual - predicted);
        // Add a small value (e.g., 1e-8) to avoid log(0)
        BCE += -(actual * log(predicted + 1e-8) + (1.0 - actual) * log(1.0 - predicted + 1e-8));
    }
    BCE /= y.size(); // Normalize by number of outputs

    // STEP 3: Calculate output error terms (same as before)
    for (size_t i = 0; i < output.size(); i++) {
        d.back()[i] = output[i] * (1 - output[i]) * error[i];
    }
    
    // STEP 4: Calculate the error term of each unit on each layer
    for (size_t i = network.size() - 2; i > 0; i--) 
    {
        for (size_t j = 0; j < network[i].size(); j++) 
        {
            double fwdErr = 0.0;
            for (size_t k = 0; k < layers[i + 1]; k++) 
            {
                fwdErr += network[i + 1][k].weights[j] * d[i + 1][k];
            }
            d[i][j] = values[i][j] * (1 - values[i][j]) * fwdErr;
        }
    }

    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (size_t i = 1; i < network.size(); i++) 
    {               // trough the layers
        for (size_t j = 0; j < layers[i]; j++) 
        {                // through the neurons
            for (size_t k = 0; k < layers[i - 1] + 1; k++) 
            {    // through the inputs
                double delta;
                if (k == layers[i - 1]) 
                {                       // is k i last weight, we use the bias to calculate the delta
                    delta = eta * d[i][j] * bias;
                } 
                else 
                {                                        // else use the value from previous layer, which is input
                    delta = eta * d[i][j] * values[i - 1][k];
                }
                network[i][j].weights[k] += delta;              // update the weights
            }
        }
    }

    return BCE;
}