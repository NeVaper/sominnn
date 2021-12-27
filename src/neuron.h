#pragma once

#include <array>

#include "node.h"

namespace snn
{
    template <typename T>
    const T fastSig(T input);

    template <typename T, unsigned NodesAmount>
    class Neuron : public Node<T>
    {
    public:
        enum ActivationType
        {
            SIGMOID
        };

        explicit Neuron();
        virtual ~Neuron() = default;

        virtual const T value() const override;
        virtual const NodeType type() const override;

        const Node<T>* input(unsigned i) const;
        void setInput(Node<T>* node, unsigned i);

        void setActivationType(ActivationType type);
        ActivationType activationType() const;

        void calculate();

    private:
        std::array<Node<T>*, NodesAmount> _inputs;
        std::array<T, NodesAmount> _weights;

        T _value = 0;

        ActivationType _activationType = SIGMOID;
    };
};

#include "neuron.inl"