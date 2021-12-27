#pragma once

#include <vector>

#include "neuron.h"
#include "typedefs.h"

namespace snn
{

class SNN
{
public:
    explicit SNN(std::vector<size_t> layers);

    void setInputValues(std::vector<snn::float_t> inputValues);


};

};

#include "nn.inl"