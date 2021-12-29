#pragma once

#include <vector>
#include <memory>

#include "node.h"
#include "typedefs.h"

namespace snn
{

class NN
{
public:
    explicit NN(const std::vector<size_t> &layers);

    void setInputValues(const std::vector<snn::float_t> inputValues);

private:
    std::vector<std::vector<std::unique_ptr<Node>>> _layers; 
};

};