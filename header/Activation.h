#pragma once
#include <algorithm>
#include <cmath>
#include <string>

class Activation 
{
public:
    enum class Type 
    {
        SIGMOID,
        TANH,
        RELU,
        LEAKY_RELU,
        STEP
    };

    Activation(Type type = Type::SIGMOID);
    float apply(float x) const;
    float derivative(float x) const;
    void setType(Type type);
    Type getType() const;
    std::string toString() const;

private:
    Type type;
    using ActFn = float(*)(float);
    using DerivFn = float(*)(float);

    ActFn activationFn;
    DerivFn derivativeFn;

    void updateFunctions();

    static float sigmoid(float x);
    static float dsigmoid(float x);
    static float tanh_act(float x);
    static float dtanh_act(float x);
    static float relu(float x);
    static float drelu(float x);
    static float leaky_relu(float x);
    static float dleaky_relu(float x);
    static float step_fn(float x);
    static float dstep_fn(float x);
};