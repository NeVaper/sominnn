#pragma once

#include <ios>
#include <memory>
#include <vector>

#include "node.h"
#include "typedefs.h"

namespace snn
{

class NN
{
public:
  explicit NN(const std::vector<size_t> &layersWidths, bool randomize = false);
  explicit NN(std::istream &is);

  void setInputValues(const std::vector<snn::float_t> &inputValues);
  void calculate();

  void toStream(std::ostream &os) const;
  void fromStream(std::istream &is);

  const std::vector<snn::float_t> readOutput() const;

private:
  void setup(const std::vector<size_t> &layersWidths, bool randomize = false);

  std::vector<std::vector<std::unique_ptr<Node>>> _layers;
};
}; // namespace snn
