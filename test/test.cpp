#include "../src/neuron.h"
#include "../src/constinput.h"
#include "../src/nn.h"

#include <iostream>
#include <array>

bool neuron_creation()
{
    std::cout << "Neuron creation.\n";
    
    bool passed = true;
    snn::Neuron neuron(32);

    std::cout << "Neuron creation passed.\n";
    return passed;
}

bool neuron_calculation()
{
    std::cout << "Neuron calculation.\n";

    constexpr size_t inAmount = 32;

    std::array<snn::ConstInput, inAmount> inputs;
    snn::Neuron neuron(inAmount);
    neuron.setActivationType(neuron.SIGMOID);

    for (size_t i = 0; i < inAmount; ++i)
    {
        neuron.setInput(&inputs[i], i);
    }

    neuron.calculate();

    bool passed = neuron.value() == snn::fastSig(inAmount + 1);

    if (passed) std::cout << "Neuron calculation passed.\n";
    else std::cout << "Neuron calculation FAIL.\n";

    return passed;
}

bool nn_creation()
{
    std::cout << "NN creation.\n";
    
    bool passed = true;
    snn::NN neuron({ 3, 5, 5, 10, 100 });

    std::cout << "NN creation passed.\n";
    return passed;
}

int main()
{
    std::cout << "\n--__TESTS__--------------------\n";

    bool passed = true;
    passed &= neuron_creation();
    passed &= neuron_calculation();
    passed &= nn_creation();

    if (passed) std::cout << "All passed.\n";
    else std::cout << "THERE ARE FAILS.\n";

    std::cout << "--__FINISHED__-----------------\n\n";
}