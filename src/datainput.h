#pragma once

#include "typedefs.h"

class DataInput
{
public:
    const snn::float_t value() const { return _value; }

    void setValue(snn::float_t value) { _value = value; }

private:
    snn::float_t _value = 0;
};