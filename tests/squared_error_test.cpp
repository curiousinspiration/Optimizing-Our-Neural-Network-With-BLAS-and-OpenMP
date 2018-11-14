/*
 * Squared Error Test
 *
 */

#include "neural/loss/squared_error_loss.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(SquaredErrorTest, TestForward)
{
    float output = 1.5;
    float target = 2.0;

    SquaredErrorLoss loss;

    // (target - output)^2
    float error = loss.Forward(output, target);
    EXPECT_EQ(0.25, error);
}

TEST(SquaredErrorTest, TestBackward)
{
    float input = 1.5;
    float target = 2.0;

    SquaredErrorLoss loss;

    // -2 * (target - input)
    float grad = loss.Backward(input, target);
    EXPECT_EQ(-1.0, grad);
}

TEST(SquaredErrorTest, TestBackwardAvgGrad)
{
    SquaredErrorLoss loss;
    {   // scoped so we can use same var names
        float input = 1.5;
        float target = 2.0;
        float grad = loss.Backward(input, target);
        EXPECT_EQ(-1.0, grad);
    }

    {   // scoped so we can use same var names
        float input = 1.0;
        float target = 2.0;
        float grad = loss.Backward(input, target);
        EXPECT_EQ(-2.0, grad);
    }

    // make sure avg gradient returns average
    float avg_grad = loss.GetAvgGrad();
    EXPECT_EQ(-1.5, avg_grad);
}
