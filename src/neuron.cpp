#include "neuron.h"

#include <stdexcept>

using namespace snn;

ConstInput Neuron::sConstInput = {};

Neuron::Neuron(size_t inputs)
{
    _weightedNodes.resize(inputs + 1, { nullptr });
    _weightedNodes.back().node = &sConstInput;
}

const snn::float_t Neuron::value() const
{
    return _value;
}

const NodeType Neuron::type() const { return NodeType::NEURON; }

std::vector<snn::float_t> Neuron::weights() const {
  std::vector<snn::float_t> weights;
  weights.reserve(_weightedNodes.size());

  for (const auto &wn : _weightedNodes)
    weights.push_back(wn.weight);

  return weights;
}

void Neuron::setActivation(Activation *act) { _activation.reset(act); }

void Neuron::calculate()
{
    _value = 0;

    for (const auto &wn : _weightedNodes)
    {
        _value += wn.value();
    }

    _value = _activation->activate(_value);
}

const Node *Neuron::input(size_t i) const { return _weightedNodes[i].node; }

void Neuron::setWeight(size_t i, float_t weight) {
  _weightedNodes[i].weight = weight;
}

void Neuron::setWeights(const std::vector<float_t> &weights) {
  if (weights.size() != _weightedNodes.size())
    throw std::invalid_argument{
        "Neuron::setWeights. Amount of weights mismatch."};

  for (size_t i = 0; i < weights.size(); ++i)
    setWeight(i, weights[i]);
}

void Neuron::setInput(Node* node, size_t i)
{
    _weightedNodes[i].node = node;
}
