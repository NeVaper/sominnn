#include "nn.h"

#include <sstream>
#include <stdexcept>

#include "datainput.h"
#include "neuron.h"

using namespace snn;

NN::NN(const std::vector<size_t> &layersWidths) {
  if (layersWidths.size() < 2)
    throw std::invalid_argument{"SNN::SNN. To few layers."};

  _layers.push_back({});
  for (size_t i = 0; i < layersWidths[0]; ++i)
    _layers.back().emplace_back(new snn::DataInput);

  for (size_t i = 1; i < layersWidths.size(); ++i) {
    _layers.push_back({});
    for (size_t j = 0; j < layersWidths[i]; ++j) {

      auto neuron = new snn::Neuron(layersWidths[i - 1]);

      for (size_t k = 0; k < layersWidths[i - 1]; ++k) {
        neuron->setInput(_layers[i - 1][k].get(), k);
      }

      _layers.back().emplace_back(neuron);
    }
  }
}

void NN::setInputValues(const std::vector<snn::float_t> &inputValues) {
  if (inputValues.size() != _layers.front().size())
    throw std::invalid_argument{
        "SNN::setInputValues. Invalid input vector size."};

  for (size_t i = 0; i < _layers.front().size(); ++i) {
    auto input = dynamic_cast<DataInput *>(_layers.front()[i].get());
    input->setValue(inputValues[i]);
  }
}

void NN::calculate() {
  for (auto lIt = _layers.begin() + 1; lIt < _layers.end(); ++lIt)
    for (auto &node : *lIt) {
      auto neuron = dynamic_cast<Neuron *>(node.get());
      neuron->calculate();
    }
}

const std::string NN::toString() const {
  std::stringstream ss;

  for (const auto &layer : _layers)
    ss << layer.size() << ';';

  ss << '\n';

  for (auto it = _layers.begin() + 1; it < _layers.end(); ++it) {
    for (const auto &node : *it) {
      auto neuron = dynamic_cast<Neuron *>(node.get());
      const auto weights = neuron->weights();
      for (const auto &w : weights) {
        ss << w << ';';
      }
    }
    ss << '\n';
  }

  return ss.str();
}

const std::vector<snn::float_t> NN::readOutput() const {
  std::vector<snn::float_t> ret;

  for (const auto &el : _layers.back())
    ret.push_back(el->value());

  return ret;
}
