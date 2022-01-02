#pragma once

#include <memory>
#include <vector>

#include "activation.h"
#include "constinput.h"
#include "node.h"
#include "typedefs.h"

namespace snn {

struct WeightedNode {
  Node *node;
  snn::float_t weight = 1;

  snn::float_t value() const { return node->value() * weight; }
};

class Neuron : public Node {
public:
  explicit Neuron(size_t inputs);
  virtual ~Neuron() = default;

  virtual const snn::float_t value() const override;
  virtual const NodeType type() const override;

  void setInput(Node *node, size_t i);
  void setWeight(size_t i, snn::float_t weight);
  void setWeights(const std::vector<float_t> &weights);

  std::vector<snn::float_t> weights() const;

  void setActivation(Activation *act);

  void calculate();

private:
  static ConstInput sConstInput;

  const Node *input(size_t i) const;

  std::vector<WeightedNode> _weightedNodes;

  snn::float_t _value = 0;

  std::unique_ptr<Activation> _activation{new ActivationFastSigmoid};
};
}; // namespace snn
