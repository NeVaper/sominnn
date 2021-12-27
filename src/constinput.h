#include "node.h"

template <typename T>
class ConstInput : public Node<T>
{
public:
    virtual const T value() const { return 1; }
    virtual const NodeType type() const { return NodeType::CONST; };
};