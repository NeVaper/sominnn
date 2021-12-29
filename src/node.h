#pragma once

#include "typedefs.h"

namespace snn
{

enum class NodeType
{
    CONST,
    DATA,
    NEURON,
};

class Node
{
public:
    virtual const snn::float_t value() const = 0;
    virtual const NodeType type() const = 0;
};

};