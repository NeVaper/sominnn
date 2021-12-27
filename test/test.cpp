#include "../src/neuron.h"
#include "../src/constinput.h"

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

    std::array<ConstInput, inAmount> inputs;
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

int main()
{
    bool passed = true;
    passed &= neuron_creation();
    passed &= neuron_calculation();

    if (passed) std::cout << "All passed.\n";
    else std::cout << "THERE ARE FAILS.\n";
}