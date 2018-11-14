/*
 * Linear Layer Implementation
 *
 */

#include "neural/layers/linear_layer.h"
#include "neural/math/tensor_math.h"

#include <glog/logging.h>

#include <sstream>

using namespace std;

namespace neural
{

LinearLayer::LinearLayer(const TTensorPtr& a_weights, bool a_hasBias)
    : m_hasBias(a_hasBias)
    , m_weights(a_weights->ToMutable()) // weights are learnable, hence mutable
{
    // if there is a bias, add an extra row to the weights
    if (m_hasBias)
    {
        m_weights = TensorMath::AddRow(m_weights, 1.0)->ToMutable();
    }
}

TTensorPtr LinearLayer::Forward(const TTensorPtr& a_input) const
{
    // Make a local copy so we can add bias if needed
    TTensorPtr l_input = a_input;

    if (m_hasBias)
    {
        // add an extra column of 1s
        l_input = TensorMath::AddCol(l_input, 1.0);
    }

    // Temporary logging for performance evaluation
//    LOG(INFO) << "Start LinearLayer::Forward mat mul " << l_input->ShapeStr() << "*" << m_weights->ShapeStr() << endl;
    TTensorPtr l_result = TensorMath::Multiply(l_input, m_weights);
//    LOG(INFO) << "End LinearLayer::Forward mat mul" << endl;
    return l_result;
}

TTensorPtr LinearLayer::Backward(const TTensorPtr& a_origInput, const TTensorPtr& a_gradInput)
{
    // orig input might have had bias
    TTensorPtr l_input = a_origInput;
    if (m_hasBias)
    {
        // add an extra column of 1s
        l_input = TensorMath::AddCol(l_input, 1.0);
    }

    // Gradient wrt weights
    // Added logging to see speed of transpose operation vs. speed of mat mul
  //  LOG(INFO) << "LinearLayer::Backward transpose inputs " << l_input->ShapeStr() << "^T" << endl;
    TTensorPtr l_inputT = TensorMath::Transpose(l_input);
  //  LOG(INFO) << "LinearLayer::Backward weights gradient computation " << l_inputT->ShapeStr() << "*" << a_gradInput->ShapeStr() << endl;
    TTensorPtr gradWrtWeights = TensorMath::Multiply(l_inputT, a_gradInput);
    m_weightGrads.push_back(gradWrtWeights);

    // Gradient wrt output
    // Added logging to see speed of transpose operation vs. speed of mat mul
    //LOG(INFO) << "LinearLayer::Backward transpose m_weights " << m_weights->ShapeStr() << "^T" << endl;
    TTensorPtr l_weightsT = TensorMath::Transpose(m_weights);
    //LOG(INFO) << "End LinearLayer::Backward output gradient computation " << a_gradInput->ShapeStr() << "*" << m_weights->ShapeStr() << "^T" << endl;
    TTensorPtr gradWrtOutput = TensorMath::Multiply(a_gradInput, l_weightsT);
    //LOG(INFO) << "End LinearLayer::Backward gradient computation" << endl;

    if (m_hasBias)
    {
        gradWrtOutput = TensorMath::RemoveCol(gradWrtOutput);
    }

    return gradWrtOutput;
}

void LinearLayer::UpdateWeights(float a_learningRate)
{
    //LOG(INFO) << "LinearLayer::UpdateWeights Start Update " << m_weights->ShapeStr() << " num grads: " << m_weightGrads.size() << endl;
    TTensorPtr l_gradient = CalcAvgWeightGrad();
    //LOG(INFO) << "LinearLayer::UpdateWeights done CalcAvgWeightGrad()" << endl;

    std::vector<float>& l_weightData = m_weights->MutableData();
    const std::vector<float>& l_gradientData = l_gradient->Data();

    #pragma omp parallel for
    for (size_t i = 0; i < l_weightData.size(); ++i)
    {
        l_weightData.at(i) -= a_learningRate * l_gradientData.at(i);
    }

    // clear gradients
    m_weightGrads.clear();
    //LOG(INFO) << "LinearLayer::UpdateWeights End Update " << m_weights->ShapeStr() << endl;
}

TTensorPtr LinearLayer::CalcAvgWeightGrad() const
{
    // Init with zeros
    TMutableTensorPtr average = Tensor::Zeros(m_weightGrads.at(0)->Shape());
    std::vector<float>& l_averageData = average->MutableData();

    // Sum up
    for (const TTensorPtr& grad : m_weightGrads)
    {
        const std::vector<float>& l_gradientData = grad->Data();

        #pragma omp parallel for
        for (size_t i = 0; i < l_gradientData.size(); ++i)
        {
            l_averageData[i] += l_gradientData[i];
        }
    }

    // Average
    float numGrads = (float)m_weightGrads.size();

    #pragma omp parallel for
    for (size_t i = 0; i < l_averageData.size(); ++i)
    {
        l_averageData[i] /= numGrads;
    }

    return average;
}

} // namespace neural
