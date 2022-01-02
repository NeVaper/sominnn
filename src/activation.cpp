#include "activation.h"

#include <cmath>

snn::float_t snn::ActivationFastSigmoid::activate(snn::float_t value) const {
  return (value / (1 + std::fabs(value)) + 1) / 2;
}

snn::float_t snn::ActivationFastSigmoid::derivative(snn::float_t value) const {
  return 1 / (2 * (std::abs(value) + 1) * (std::abs(value) + 1));
}
