/*
 * Squared Error Loss Implementation
 *
 */

#include "neural/loss/squared_error_loss.h"

using namespace std;

namespace neural
{

float SquaredErrorLoss::Forward(float output, float target) const
{
    float difference = target - output;
    return difference * difference; // square the difference
}

float SquaredErrorLoss::Backward(float input, float target)
{
    float grad = -2.0 * (target - input);
    m_grads.push_back(grad);
    return grad;
}

float SquaredErrorLoss::GetAvgGrad() const
{
    float sum = 0.0;
    for (size_t i = 0; i < m_grads.size(); ++i)
    {
        sum += m_grads.at(i);
    }
    return sum / ((float) m_grads.size());
}

void SquaredErrorLoss::ZeroGrad()
{
    m_grads.clear();
}

} // namespace neural
