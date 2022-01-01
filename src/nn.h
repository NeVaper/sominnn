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
    explicit NN(const std::vector<size_t> &layers);

    void setInputValues(const std::vector<snn::float_t> &inputValues);
    void calculate();

    const std::string toString() const;
    void fromString(const std::string &str);

    const std::vector<snn::float_t> readOutput() const;

  private:
    std::vector<std::vector<std::unique_ptr<Node>>> _layers; 
};

};
