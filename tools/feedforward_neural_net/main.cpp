/*
 * Example tool training feed forward neural network on mnist data
 *
 */


#include "neural/data/mnist_dataloader.h"
#include "neural/layers/linear_layer.h"
#include "neural/layers/relu_layer.h"
#include "neural/loss/squared_error_loss.h"

#include <glog/logging.h>

using namespace neural;
using namespace std;

float CalcAverage(const vector<float>& vals)
{
    float sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        sum += vals.at(i);
    }
    return sum / ((float)vals.size());
}

int main(int argc, char const *argv[])
{
    // Define data loader
    MNISTDataloader l_dataloader("../data/mnist/");

    // Define model
    // first linear layer is 784x300
    // 784 inputs, 300 hidden size
    LinearLayer firstLinearLayer(Tensor::Random({784, 300}, -0.01f, 0.01f));

    // Non-linear activation
    ReLULayer activationLayer;
    
    // second linear layer is 300x1
    // 300 hidden units, 1 output
    LinearLayer secondLinearLayer(Tensor::Random({300, 1}, -0.01f, 0.01f));

    // Error function
    SquaredErrorLoss loss;

    // Training loop
    float learningRate = 0.5;
    size_t numEpochs = 10;
    size_t numIters = l_dataloader.DataLength();
    for (size_t i = 0; i < numEpochs; ++i)
    {
        LOG(INFO) << "--EPOCH (" << i << ")--" << endl;
        vector<float> errorAcc;
        for (size_t j = 0; j < numIters; ++j)
        {
            LOG(INFO) << "--ITER (" << i << "," << j << ")--" << endl;
            // Get training example
            TMutableTensorPtr input, output;
            l_dataloader.DataAt(j, input, output);
            float targetOutput = output->At({0,0});

            // Forward pass
            TTensorPtr output0 = firstLinearLayer.Forward(input);
            TTensorPtr output1 = activationLayer.Forward(output0);
            TTensorPtr y_pred = secondLinearLayer.Forward(output1);
            float yPredVal = y_pred->At({0,0});
            LOG(INFO) << "Got prediction: " << yPredVal << " for target " << targetOutput << endl;

            // Calc Error
            float error = loss.Forward(yPredVal, targetOutput);
            errorAcc.push_back(error);
            LOG(INFO) << "Calculated error for example [" << i << "]: " << error << endl;

            // Compute average error for last 100 examples
            if (j % 100 == 0)
            {
                float avgError = CalcAverage(errorAcc);
                LOG(INFO) << "avgError = " << avgError << endl;
                errorAcc.clear();
            }

            // Backward pass
            float errorGrad = loss.Backward(yPredVal, targetOutput);
            TTensorPtr y_predGrad = secondLinearLayer.Backward(output1, Tensor::New({1,1}, {errorGrad}));
            TTensorPtr grad1 = activationLayer.Backward(output0, y_predGrad);
            TTensorPtr grad0 = firstLinearLayer.Backward(input, grad1);

            // Gradient Descent
            secondLinearLayer.UpdateWeights(learningRate);
            firstLinearLayer.UpdateWeights(learningRate);
        }
    }

    return 0;
}