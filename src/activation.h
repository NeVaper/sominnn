#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "typedefs.h"

namespace snn {

class Activation {
public:
  virtual ~Activation() = default;

  virtual snn::float_t activate(snn::float_t value) const = 0;
  virtual snn::float_t derivative(snn::float_t value) const = 0;
};

class ActivationFastSigmoid : public Activation {
public:
  virtual ~ActivationFastSigmoid() = default;

  virtual snn::float_t activate(snn::float_t value) const override;
  virtual snn::float_t derivative(snn::float_t value) const override;
};

}; // namespace snn

#endif // ACTIVATION_H
