#pragma once

#include "node.h"

#include "typedefs.h"

namespace snn
{

class ConstInput : public Node
{
public:
    virtual const snn::float_t value() const { return 1; }
    virtual const NodeType type() const { return NodeType::CONST; };
};

};