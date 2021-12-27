#include "neuron.h"

#include <cmath>

using namespace snn;

template <typename T>
const T snn::fastSig(T value)
{
    return value / (1 + std::fabs(value));
}

template <typename T, unsigned NodesAmount>
Neuron<T, NodesAmount>::Neuron()
{
    for (auto &w : _weights)
        w = 1;
}

template <typename T, unsigned NodesAmount>
const T Neuron<T, NodesAmount>::value() const
{
    return _value;
}

template <typename T, unsigned NodesAmount>
const NodeType Neuron<T, NodesAmount>::type() const
{
    return NodeType::NEURON;
}

template <typename T, unsigned NodesAmount>
void Neuron<T, NodesAmount>::calculate()
{
    _value = 0;

    for (unsigned i = 0; i < NodesAmount; ++i)
    {
        _value += _inputs[i]->value() * _weights[i];
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

template <typename T, unsigned NodesAmount>
void Neuron<T, NodesAmount>::setActivationType(ActivationType type)
{
    _activationType = type;
}

template <typename T, unsigned NodesAmount>
typename Neuron<T, NodesAmount>::ActivationType 
    Neuron<T, NodesAmount>::activationType() const
{
    return _activationType;
}

template <typename T, unsigned NodesAmount>
const Node<T>* Neuron<T, NodesAmount>::input(unsigned i) const
{
    return _inputs[i];
}

template <typename T, unsigned NodesAmount>
void Neuron<T, NodesAmount>::setInput(Node<T>* node, unsigned i)
{
    _inputs[i] = node;
}