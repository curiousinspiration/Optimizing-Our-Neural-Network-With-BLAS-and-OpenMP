/*
 * Linear Layer Definition
 *
 */

#pragma once

#include "neural/layers/layer.h"

namespace neural
{

class LinearLayer : public Layer
{
public:
    LinearLayer(const TTensorPtr& a_weights, bool a_hasBias = true);
    virtual TTensorPtr Forward(const TTensorPtr& a_input) const override;
    virtual TTensorPtr Backward(const TTensorPtr& a_origInput, const TTensorPtr& a_gradInput) override;

    TTensorPtr CalcAvgWeightGrad() const;
    void UpdateWeights(float a_learningRate);

private:
    bool m_hasBias;
    TMutableTensorPtr m_weights;
    std::vector<TTensorPtr> m_weightGrads;
};

} // namespace
