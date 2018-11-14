/*
 * Tensor is our basic building block for our models, inputs,
 * internal representation and outputs
 */

#pragma once

#include <vector>
#include <memory>

namespace neural
{

// We are forward declaring a Tensor so that we can use it
// in the typedefs before the class is defined
class Tensor;

// Assume most tensors are going to be const
typedef std::shared_ptr<const Tensor> TTensorPtr;

// explicitly call out if tensor is mutable
typedef std::shared_ptr<Tensor> TMutableTensorPtr;

class Tensor
{
public:
    // Pass in a Vector to represent the size of the the tensor,
    // ie. vector({0,1,2})
    Tensor(const std::vector<size_t>& a_shape);

    // Can pass in size and the data you want in the tensor
    Tensor(
        const std::vector<size_t>& a_shape,
        const std::vector<float>& a_data);

    // Creates new Tensor with shape
    static TMutableTensorPtr New(
      const std::vector<size_t>& a_shape);

    // Tensor with data
    static TMutableTensorPtr New(
      const std::vector<size_t>& a_shape,
      const std::vector<float>& a_data);

    // Tensor filled with random floats
    static TMutableTensorPtr Random(const std::vector<size_t>& a_shape, float a_min=0.0, float a_max=1.0);

    // Tensor filled with all the same value
    static TMutableTensorPtr Constant(const std::vector<size_t>& a_shape, float a_val);

    // Tensor filled with ones
    static TMutableTensorPtr Zeros(const std::vector<size_t>& a_shape);

    // Tensor filled with zeros
    static TMutableTensorPtr Ones(const std::vector<size_t>& a_shape);

    // Copies data into mutable tensor
    TMutableTensorPtr ToMutable() const;

    // Sets all the values in the tensor to this value
    void SetAll(float a_val);
  
    // Get the shape of tensor ie: 4x32x32x3
    const std::vector<size_t>& Shape() const;
    
    // Shape in a nice readable string
    static std::string ShapeStr(const std::vector<size_t>& a_shape);
    std::string ShapeStr() const;

    // Get size of raw data
    size_t Size() const;
  
    // Get raw data
    const std::vector<float>& Data() const;
    std::vector<float>& MutableData();

    // Returns value at offset at a_idx ie. {1, 2, 0}
    float At(const std::vector<size_t>& a_idx) const;

    // Set value at idx
    void SetAt(const std::vector<size_t>& a_idx, float a_val);
  
private:
    std::vector<size_t> m_shape;
    std::vector<float> m_data;
    // Precomputed stride sizes
    std::vector<size_t> m_strideSizes;
  
    size_t p_CalcSize(const std::vector<size_t>& a_shape) const;
    // Add to precompute stride sizes
    std::vector<size_t> p_ComputeStrideSizes(const std::vector<size_t>& a_tensorShape) const;

    // Add to calculate offset into data given strides
    size_t p_DataOffsetFromIdx(
        const std::vector<size_t>& a_tensorIdx) const;
};
  
} // namespace neural