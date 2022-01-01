#pragma once

#include <vector>
#include <cstdlib>

#include "typedefs.h"
#include "node.h"
#include "constinput.h"

namespace snn
{
    const snn::float_t fastSig(snn::float_t input);

    struct WeightedNode
    {
        Node* node;
        snn::float_t weight = 1;

        snn::float_t value() const
        { 
            return node->value() * weight;
        }
    };

    class Neuron : public Node
    {
    public:
        enum ActivationType
        {
            SIGMOID
        };

        explicit Neuron(size_t inputs);
        virtual ~Neuron() = default;

        virtual const snn::float_t value() const override;
        virtual const NodeType type() const override;

        void setInput(Node *node, size_t i);

        std::vector<snn::float_t> weights() const;

        void setActivationType(ActivationType type);
        ActivationType activationType() const;

        void calculate();

      private:
        const Node *input(size_t i) const;

        static ConstInput sConstInput;

        std::vector<WeightedNode> _weightedNodes;

        snn::float_t _value = 0;

        ActivationType _activationType = SIGMOID;
    };
};
