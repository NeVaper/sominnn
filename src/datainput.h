#pragma once

#include "typedefs.h"

namespace snn
{

class DataInput : public Node
{
public:
    virtual const snn::float_t value() const { return _value; }
    virtual const NodeType type() const { return NodeType::DATA; }

    void setValue(snn::float_t value) { _value = value; }

private:
    snn::float_t _value = 0;
};

};