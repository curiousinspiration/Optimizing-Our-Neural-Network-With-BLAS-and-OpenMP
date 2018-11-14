/*
 * Relu Layer Implementation
 *
 */

#include "neural/layers/relu_layer.h"

#include <algorithm>

using namespace std;

namespace neural
{

ReLULayer::ReLULayer()
{

}

TTensorPtr ReLULayer::Forward(const TTensorPtr& a_input) const
{
    // initialize our return matrix with the input
    TMutableTensorPtr l_ret = a_input->ToMutable();
 
    // walk over rows
    for (size_t i = 0; i < a_input->Shape().at(0); ++i)
    {
        // walk over cols
        for (size_t j = 0; j < a_input->Shape().at(1); ++j)
        {
            // max(0,x)
            l_ret->SetAt({i,j}, std::max(0.0f, l_ret->At({i,j})));
        }
    }
    return l_ret;
}

TTensorPtr ReLULayer::Backward(const TTensorPtr& a_origInput, const TTensorPtr& a_gradInput)
{
    TMutableTensorPtr grad = a_gradInput->ToMutable();

    // walk over rows
    for (size_t i = 0; i < a_gradInput->Shape().at(0); ++i)
    {
        // walk over cols
        for (size_t j = 0; j < a_gradInput->Shape().at(1); ++j)
        {
            if (a_origInput->At({i,j}) < 0)
            {
                grad->SetAt({i,j}, 0.0);
            }
        }
    }
    return grad;
}

} // namespace neural
