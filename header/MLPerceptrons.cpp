#include "MLPerceptrons.h"
#include <iomanip> 

float frand()
{
    return static_cast<float>((2.0 * rand() / RAND_MAX) - 1.0);
}

// Activation and derivative functions
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float tanh_act(float x) { return std::tanh(x); }
float dtanh_act(float x) { return 1 - std::tanh(x) * std::tanh(x); }

float relu(float x) { return x > 0 ? x : 0.0f; }
float drelu(float x) { return x > 0 ? 1.0f : 0.0f; }

float leaky_relu(float x) { return x > 0 ? x : 0.01f * x; }
float dleaky_relu(float x) { return x > 0 ? 1.0f : 0.01f; }

float step_fn(float x) { return x >= 0 ? 1.0f : 0.0f; }
float dstep_fn(float x) { return 0.0; /* not differentiable at 0, rarely used in backprop */ }

Activation getActivation(ActivationType t)
{
    switch (t)
    {
    case ActivationType::TANH:       return { tanh_act,   dtanh_act };
    case ActivationType::RELU:       return { relu,       drelu };
    case ActivationType::STEP:       return { step_fn,    dstep_fn };
	case ActivationType::LEAKY_RELU: return { leaky_relu, dleaky_relu };
    default:                         return { sigmoid,    dsigmoid };
    }
}

//constructor for Perceptron class
Perceptron::Perceptron(size_t inputs, ActivationType type, float bias)
    : bias(bias), aType(type)
{
    weights.resize(inputs + 1); // +1 since we have bias as well
    generate(weights.begin(), weights.end(), frand);
    auto act = getActivation(type);
    activate = act.fn;
    derivative = act.dfn;
}

float Perceptron::run(std::vector<float> x)
{
    x.push_back(bias); // push bias into the vector
    return activate(inner_product(x.begin(), x.end(), weights.begin(), (float)0.0));
}

void Perceptron::set_weights(std::vector<float> w_init)
{
    weights = w_init;
}

MultilayerPerceptron::MultilayerPerceptron(std::vector<size_t> layers, ActivationType type, float bias, float eta, float dropout)
	: layers(layers), bias(bias), eta(eta), actType(type), dropout_rate(dropout), is_training(false)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        values.push_back(std::vector<float>(layers[i], 0.0));
        d.push_back(std::vector<float>(layers[i], 0.0));
        network.push_back(std::vector<Perceptron>());
        if (i > 0)
        {
            for (size_t j = 0; j < layers[i]; j++)
            {
                network[i].push_back(Perceptron(layers[i - 1], actType, bias)); // create same number of neurons as the previous layer
            }
        }
    }
}

void MultilayerPerceptron::set_weights(std::vector< std::vector< std::vector<float> > > w_init)
{
    for (size_t i = 0; i < w_init.size(); i++)
    {
        for (size_t j = 0; j < w_init[i].size(); j++)
        {
            network[i + 1][j].set_weights(w_init[i][j]); 
        }
    }
}

std::string MultilayerPerceptron::activationName(ActivationType t)
{
    switch (t) 
    {
        case ActivationType::TANH:    
            return "tanh";
        case ActivationType::RELU:    
            return "relu";
        case ActivationType::STEP:    
            return "step";
        case ActivationType::LEAKY_RELU:
            return "leaky relu";
        default:                      
            return "sigmoid";
    }
}

void MultilayerPerceptron::printStructure(std::ostream& out) const 
{
    out << "\nNetwork Structure\n"
        << "-----------------\n"
        // column headers
        << std::left
        << std::setw(8) << "Layer"
        << std::setw(12) << "Type"
        << std::setw(8) << "Units"
        << std::setw(12) << "Activation"
        << std::setw(12) << "Params"
        << "\n";

    size_t total_params = 0;
    for (size_t i = 0; i < layers.size(); ++i) 
    {
        std::string layerType;
        if (i == 0)                         layerType = "Input";
        else if (i == layers.size() - 1)   layerType = "Output";
        else                                layerType = "Hidden";

        size_t params = 0;
        if (i > 0) 
        {
            // each neuron has (previous layer units + 1) weights (incl. bias)
            params = layers[i] * (layers[i - 1] + 1);
            total_params += params;
        }

        out << std::left
            << std::setw(8) << i
            << std::setw(12) << layerType
            << std::setw(8) << layers[i]
            << std::setw(12) << (i == 0 ? "-" : activationName(actType))
                << std::setw(12) << params
                << "\n";
    }

    out << "-----------------\n"
        << "Total params: " << total_params << "\n\n";
}


void MultilayerPerceptron::printWeights()
{
    std::cout << std::endl;
    for (size_t i = 1; i < network.size(); i++) 
    {
        for (size_t j = 0; j < layers[i]; j++) 
        {
            std::cout << "Layer " << i << " Neuron " << j << ": ";
            for (auto& itr : network[i][j].weights) 
            {
                std::cout << itr << "   ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
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

std::vector<float> MultilayerPerceptron::run(std::vector<float> x)
{
    values[0] = x;
    // Apply to hidden layers only
    for (size_t i = 1; i < network.size() - 1; i++)
    {
        for (size_t j = 0; j < layers[i]; j++)
        {
            values[i][j] = network[i][j].run(values[i - 1]);
        }

        // Apply dropout only during training and only to hidden layers
        if (is_training && dropout_rate > 0)
        {
            for (size_t j = 0; j < layers[i]; j++)
            {
                if (static_cast<float>(rand() / static_cast<double>(RAND_MAX)) < dropout_rate)
                {
                    // Drop this neuron
                    values[i][j] = 0.0f;
                }
                else
                {
                    // Scale to maintain expected sum
                    values[i][j] /= (1.0f - dropout_rate);
                }
            }
        }
    }

    // Output layer (no dropout)
    size_t lastIdx = network.size() - 1;
    for (size_t j = 0; j < layers[lastIdx]; j++)
    {
        values[lastIdx][j] = network[lastIdx][j].run(values[lastIdx - 1]);
    }

    return values.back();
}


void MultilayerPerceptron::clipGradient(float& gradient, float threshold)
{
    if (gradient > threshold) gradient = threshold;
    if (gradient < -threshold) gradient = -threshold;
}

float MultilayerPerceptron::backPropagation(std::vector<float> x, std::vector<float> y)
{
    // STEP 1: Feed a sample to the network
    std::vector<float> output = run(x);

    // STEP 2: Calculate Binary Cross-Entropy Loss instead of MSE
    float BCE = 0.0;
    std::vector<float> error;
    for (size_t i = 0; i < y.size(); i++)
    {
        float predicted = output[i];
        float actual = y[i];
        error.push_back(actual - predicted);
        // Add a small value (e.g., 1e-8) to avoid log(0)
        BCE += -(actual * log(predicted + 1e-8f) + (1.0f - actual) * log(1.0f - predicted + 1e-8f));
    }
    BCE /= y.size(); // Normalize by number of outputs

    // STEP 3: Calculate output error terms (same as before)
    for (size_t i = 0; i < output.size(); i++)
    {
        d.back()[i] = network.back()[i].derivative(values.back()[i]) * error[i];
    }

    // STEP 4: Calculate the error term of each unit on each layer (backpropagate)
    for (size_t i = network.size() - 2; i > 0; i--)
    {
        for (size_t j = 0; j < network[i].size(); j++)
        {
            float fwdErr = 0.0;
            for (size_t k = 0; k < layers[i + 1]; k++)
            {
                fwdErr += network[i + 1][k].weights[j] * d[i + 1][k];
            }
            d[i][j] = network[i][j].derivative(values[i][j]) * fwdErr;
        }
    }

    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (size_t i = 1; i < network.size(); i++)
    {
        for (size_t j = 0; j < layers[i]; j++)
        {
            for (size_t k = 0; k < layers[i - 1] + 1; k++)
            {
                float delta;
                if (k == layers[i - 1])
                {
                    delta = eta * d[i][j] * bias;
                }
                else
                {
                    delta = eta * d[i][j] * values[i - 1][k];
                }
                network[i][j].weights[k] += delta;
            }
        }
    }
    return BCE;
}

void MultilayerPerceptron::setTrainingMode(bool training_mode)
{
    is_training = training_mode;
}

void MultilayerPerceptron::setDropoutRate(float rate)
{
    dropout_rate = std::max(0.0f, std::min(1.0f, rate)); // Clamp between 0 and 1
}