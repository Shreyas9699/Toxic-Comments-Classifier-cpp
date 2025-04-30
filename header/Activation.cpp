#include "Activation.h"

// Static activation functions implementation
float Activation::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float Activation::dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float Activation::tanh_act(float x) { return std::tanh(x); }
float Activation::dtanh_act(float x) { return 1 - std::tanh(x) * std::tanh(x); }

float Activation::relu(float x) { return x > 0 ? x : 0.0f; }
float Activation::drelu(float x) { return x > 0 ? 1.0f : 0.0f; }

float Activation::leaky_relu(float x) { return x > 0 ? x : 0.01f * x; }
float Activation::dleaky_relu(float x) { return x > 0 ? 1.0f : 0.01f; }

float Activation::step_fn(float x) { return x >= 0 ? 1.0f : 0.0f; }
float Activation::dstep_fn(float x) { return 0.0f; /* not differentiable at 0, rarely used in backprop */ }

Activation::Activation(Type type) : type(type) 
{
    updateFunctions();
}


void Activation::updateFunctions() 
{
    switch (type) 
    {
    case Type::TANH:
        activationFn = tanh_act;
        derivativeFn = dtanh_act;
        break;
    case Type::RELU:
        activationFn = relu;
        derivativeFn = drelu;
        break;
    case Type::LEAKY_RELU:
        activationFn = leaky_relu;
        derivativeFn = dleaky_relu;
        break;
    case Type::STEP:
        activationFn = step_fn;
        derivativeFn = dstep_fn;
        break;
    case Type::SIGMOID:
    default:
        activationFn = sigmoid;
        derivativeFn = dsigmoid;
        break;
    }
}


float Activation::apply(float x) const 
{
    return activationFn(x);
}


float Activation::derivative(float x) const
{
    return derivativeFn(x);
}


void Activation::setType(Type type) 
{
    this->type = type;
    updateFunctions();
}

Activation::Type Activation::getType() const 
{
    return type;
}

std::string Activation::toString() const 
{
    switch (type) 
    {
    case Type::SIGMOID:
        return "sigmoid";
    case Type::TANH:
        return "tanh";
    case Type::RELU:
        return "relu";
    case Type::LEAKY_RELU:
        return "leaky relu";
    case Type::STEP:
        return "step";
    default:
        return "Invalid";
    }
}