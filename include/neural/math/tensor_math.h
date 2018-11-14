/*
 * Tensor Math Definition
 *
 */

#pragma once

#include "neural/math/tensor.h"

namespace neural
{

class TensorMath
{
public:
    static TTensorPtr Multiply(const TTensorPtr& a_lhs, const TTensorPtr& a_rhs);
    static TTensorPtr Transpose(const TTensorPtr& a_tensor);
    // Assumes matrix, adds column at the end
    static TTensorPtr AddCol(const TTensorPtr& a_tensor, float a_val);
    // Assumes matrix, removes column at the end
    static TTensorPtr RemoveCol(const TTensorPtr& a_tensor);
    // Assumes matrix, adds row at the end
    static TTensorPtr AddRow(const TTensorPtr& a_tensor, float a_val);
    // Assumes matrix, removes row at the end
    static TTensorPtr RemoveRow(const TTensorPtr& a_tensor);
};

} // namespace neural
