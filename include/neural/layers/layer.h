/*
 * Base Class for Layer
 *
 */

#pragma once

#include "neural/math/tensor.h"

namespace neural
{

class Layer
{
public:
    virtual TTensorPtr Forward(const TTensorPtr& a_input) const = 0;
    virtual TTensorPtr Backward(const TTensorPtr& a_origInput, const TTensorPtr& a_gradInput) = 0;

};

} // namespace neural
