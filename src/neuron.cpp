#include "neuron.h"

#include <cmath>

using namespace snn;

ConstInput Neuron::sConstInput = {};

const snn::float_t snn::fastSig(snn::float_t value)
{
    return value / (1 + std::fabs(value));
}

Neuron::Neuron(size_t inputs)
{
    _weightedNodes.resize(inputs + 1, { nullptr });
    _weightedNodes.back().node = &sConstInput;
}

const snn::float_t Neuron::value() const
{
    return _value;
}

const NodeType Neuron::type() const
{
    return NodeType::NEURON;
}

void Neuron::calculate()
{
    _value = 0;

    for (const auto &wn : _weightedNodes)
    {
        _value += wn.value();
    }

    switch (_activationType)
    {
    case Neuron::SIGMOID:
        _value = fastSig(_value);
        break;
    
    default:
        throw;
    }
}

void Neuron::setActivationType(ActivationType type)
{
    _activationType = type;
}

typename Neuron::ActivationType 
    Neuron::activationType() const
{
    return _activationType;
}

const Node* Neuron::input(size_t i) const
{
    return _weightedNodes[i].node;
}

void Neuron::setInput(Node* node, size_t i)
{
    _weightedNodes[i].node = node;
}