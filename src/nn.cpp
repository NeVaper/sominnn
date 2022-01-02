#include "nn.h"

#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "datainput.h"
#include "neuron.h"

using namespace snn;

namespace {

inline const std::vector<size_t> readDimensions(std::istream &is) {
  std::vector<size_t> dimensions;

  size_t d = 1;
  while (true) {
    is >> d;
    if (d == 0)
      break;

    dimensions.push_back(d);
  };

  return dimensions;
}

inline const std::vector<snn::float_t> readNeuronWeights(std::istream &is,
                                                         size_t size) {
  std::vector<snn::float_t> weights;
  weights.reserve(size);

  for (size_t i = 0; i < size; ++i) {
    snn::float_t num;
    is >> num;
    weights.push_back(num);
  }

  return weights;
}

template <typename Numeric, typename Generator = std::mt19937>
Numeric random(Numeric from, Numeric to) {
  thread_local static Generator gen(std::random_device{}());

  using dist_type =
      typename std::conditional<std::is_integral<Numeric>::value,
                                std::uniform_int_distribution<Numeric>,
                                std::uniform_real_distribution<Numeric> >::type;

  thread_local static dist_type dist;

  return dist(gen, typename dist_type::param_type{from, to});
}

} // namespace

NN::NN(const std::vector<size_t> &layersWidths, bool randomize) {
  setup(layersWidths, randomize);
}

NN::NN(std::istream &is) { fromStream(is); }

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

void NN::toStream(std::ostream &os) const {
  const auto p = os.precision();

  os.precision(std::numeric_limits<snn::float_t>::max_digits10);

  for (const auto &layer : _layers)
    os << layer.size() << ' ';

  os << '0' << '\n';

  for (auto it = _layers.begin() + 1; it < _layers.end(); ++it) {
    for (const auto &node : *it) {
      auto neuron = dynamic_cast<Neuron *>(node.get());
      const auto weights = neuron->weights();
      for (const auto &w : weights) {
        os << w << ' ';
      }
      os << '\n';
    }
    os << '\n';
  }

  os.precision(p);
}

void NN::fromStream(std::istream &is) {
  const auto dimensions = readDimensions(is);

  setup(dimensions);

  for (size_t i = 1; i < dimensions.size(); ++i) {
    for (size_t j = 0; j < dimensions[i]; ++j) {
      auto neuron = dynamic_cast<Neuron *>(_layers[i][j].get());
      neuron->setWeights(readNeuronWeights(is, dimensions[i - 1] + 1));
    }
  }
}

const std::vector<snn::float_t> NN::readOutput() const {
  std::vector<snn::float_t> ret;

  for (const auto &el : _layers.back())
    ret.push_back(el->value());

  return ret;
}

void NN::setup(const std::vector<size_t> &layersWidths, bool randomize) {
  if (layersWidths.size() < 2)
    throw std::invalid_argument{"SNN::SNN. To few layers."};

  _layers.push_back({});
  for (size_t i = 0; i < layersWidths[0]; ++i) {
    _layers.back().emplace_back(new snn::DataInput);
  }

  for (size_t i = 1; i < layersWidths.size(); ++i) {
    _layers.push_back({});
    for (size_t j = 0; j < layersWidths[i]; ++j) {

      auto neuron = new snn::Neuron(layersWidths[i - 1]);

      for (size_t k = 0; k < layersWidths[i - 1]; ++k) {
        neuron->setInput(_layers[i - 1][k].get(), k);

        if (randomize)
          neuron->setWeight(k, random<snn::float_t>(-10, 10));
      }

      if (randomize)
        neuron->setWeight(layersWidths[i - 1], random<snn::float_t>(-10, 10));

      _layers.back().emplace_back(neuron);
    }
  }
}
