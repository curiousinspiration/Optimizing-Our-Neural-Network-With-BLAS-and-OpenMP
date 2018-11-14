/*
 * ReLU Layer Definition
 *
 */

#pragma once

#include "neural/layers/layer.h"

namespace neural
{

class ReLULayer : public Layer
{
public:
    ReLULayer();
    virtual TTensorPtr Forward(const TTensorPtr& a_input) const override;
    virtual TTensorPtr Backward(const TTensorPtr& a_origInput, const TTensorPtr& a_gradInput) override;

private:

};

} // namespace neural
