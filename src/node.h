#pragma once

enum class NodeType
{
    CONST,
    DATA,
    NEURON,
};

template <typename T>
class Node
{
public:
    virtual const T value() const = 0;
    virtual const NodeType type() const = 0;
};