/*
 * Tensor implementation
 */

#include "neural/math/tensor.h"

#include <sstream>
#include <chrono>
#include <random>

using namespace std;

namespace neural
{

Tensor::Tensor(const std::vector<size_t>& a_shape)
    : m_shape(a_shape)
    , m_strideSizes(p_ComputeStrideSizes(m_shape))
{
    m_data.resize(p_CalcSize(a_shape));
}

Tensor::Tensor(const std::vector<size_t>& a_shape,
               const std::vector<float>& a_data)
    : m_shape(a_shape)
    , m_data(a_data)
    , m_strideSizes(p_ComputeStrideSizes(m_shape))
{
    m_data.resize(p_CalcSize(a_shape));
}

TMutableTensorPtr Tensor::New(const std::vector<size_t>& a_shape)
{
    return TMutableTensorPtr(new Tensor(a_shape));
}

TMutableTensorPtr Tensor::New(
    const std::vector<size_t>& a_shape, const std::vector<float>& a_data)
{
    return TMutableTensorPtr(new Tensor(a_shape, a_data));
}

TMutableTensorPtr Tensor::Random(const std::vector<size_t>& a_shape, 
                          float a_min, float a_max)
{
    // construct a trivial random generator engine from a time-based seed:
    static unsigned l_seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine l_generator(l_seed);

    static std::normal_distribution<float> l_distribution(a_min, a_max);

    size_t l_size(1);
    for (size_t i = 0; i < a_shape.size(); ++i)
    {
        l_size *= a_shape[i];
    }
    vector<float> l_data;
    for (size_t i = 0; i < l_size; ++i)
    {
        l_data.push_back(l_distribution(l_generator));
    }

    return New(a_shape, l_data);
}

TMutableTensorPtr Tensor::Constant(const std::vector<size_t>& a_shape, float a_val)
{
    TMutableTensorPtr l_tensor = New(a_shape);
    l_tensor->SetAll(a_val);
    return l_tensor;
}

TMutableTensorPtr Tensor::Zeros(const std::vector<size_t>& a_shape)
{
    TMutableTensorPtr l_tensor = New(a_shape);
    l_tensor->SetAll(0.0);
    return l_tensor;
}

TMutableTensorPtr Tensor::Ones(const std::vector<size_t>& a_shape)
{
    TMutableTensorPtr l_tensor = New(a_shape);
    l_tensor->SetAll(1.0);
    return l_tensor;
}

TMutableTensorPtr Tensor::ToMutable() const
{
    return TMutableTensorPtr(new Tensor(m_shape, m_data));
}

void Tensor::SetAll(float a_val)
{
    for (size_t i = 0; i < p_CalcSize(m_shape); ++i)
    {
        m_data[i] = a_val;
    }
}

const std::vector<size_t>& Tensor::Shape() const
{
    return m_shape;
}

std::string Tensor::ShapeStr(const std::vector<size_t>& a_shape)
{
    string l_ret("");
    for (size_t i = 0; i < a_shape.size(); ++i)
    {
        l_ret.append(std::to_string(a_shape.at(i)));
        if (i != a_shape.size()-1) l_ret.append("x");
    }
    return l_ret;
}

std::string Tensor::ShapeStr() const
{
    return ShapeStr(m_shape);
}

size_t Tensor::Size() const
{
    return m_data.size();
}
  
const std::vector<float>& Tensor::Data() const
{
    return m_data;
}

std::vector<float>& Tensor::MutableData()
{
    return m_data;
}

float Tensor::At(const std::vector<size_t>& a_idx) const
{
    size_t l_offset = p_DataOffsetFromIdx(a_idx);
    return m_data[l_offset];
}

void Tensor::SetAt(const std::vector<size_t>& a_idx, float a_val)
{
    size_t l_offset = p_DataOffsetFromIdx(a_idx);
    m_data[l_offset] = a_val;
}

size_t Tensor::p_CalcSize(const std::vector<size_t>& a_shape) const
{
    size_t l_size(1);
    for (size_t i = 0; i < a_shape.size(); ++i)
    {
        l_size *= a_shape[i];
    }
    return l_size;
}

size_t Tensor::p_DataOffsetFromIdx(
    const std::vector<size_t>& a_tensorIdx) const
{
    if (m_shape.size() != a_tensorIdx.size())
    {
        stringstream ss;
        ss << "Tensor::DataOffsetFromIdx invalid shape size: " 
           << ShapeStr(a_tensorIdx) 
           << " for tensor " << ShapeStr(m_shape)
           << ", " << m_shape.size() << " != " << a_tensorIdx.size();
        throw(runtime_error(ss.str()));
    }

    for (size_t i = 0; i < m_shape.size(); ++i)
    {
        if (a_tensorIdx[i] >= m_shape[i])
        {
            stringstream ss;
            ss << "Tensor::DataOffsetFromIdx invalid shape idx: "
               << ShapeStr(a_tensorIdx)
               << " for tensor " << ShapeStr(m_shape)
               << ", @" << i << ": "
               << a_tensorIdx[i] << " >= " << m_shape[i];

            throw(runtime_error(ss.str()));
        }
    }

    // So we can go over the new input shape and jump by those amounts
    size_t l_offset = 0;
    for (size_t i = 0; i < a_tensorIdx.size(); ++i)
    {
        // for example if we have a shape of {4,2,2,3}
        // Now for the example above, l_strideSizes = {12, 6, 3, 1}
        l_offset += a_tensorIdx[i] * m_strideSizes[i];
    }
    return l_offset;
}

vector<size_t> Tensor::p_ComputeStrideSizes(
    const std::vector<size_t>& a_tensorShape) const
{
    // we have to calculate the stride sizes for each part of the shape
    // for example if we have a shape of {4,2,2,3}
    // and we want to access {1,0,0,0}
    // we know it will be at 1 * (2 * 2 * 3)
    // or {2,0,0,0} is 2 * (2 * 2 * 3)
    // then we move to the next column
    // for {2,1,0,0}
    // we get to the first offset ie 2 * (2 * 2 * 3),
    // then do the same for the next offset so 
    // (2 * (2 * 2 * 3)) + (1 * (2 * 3))
    vector<size_t> l_strideSizes;
    for (size_t i = 0; i < a_tensorShape.size(); ++i)
    {
        size_t l_strideSize = 1;
        for (size_t j = i+1; j < a_tensorShape.size(); ++j)
        {
            l_strideSize *= a_tensorShape.at(j);
        }
        l_strideSizes.push_back(l_strideSize);
    }
    return l_strideSizes;
}
} // namespace neural