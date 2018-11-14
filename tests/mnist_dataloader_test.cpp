/*
 * MNIST Dataloader Test
 *
 */

#include "neural/data/mnist_dataloader.h"

#include <gtest/gtest.h>

#include <iostream>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(MNISTDataloaderTest, TestDataLengthTrain)
{
    bool l_isTrain = true;
    std::string l_path("../data/mnist");
    MNISTDataloader l_dataloader(l_path, l_isTrain);
    EXPECT_EQ(60000, l_dataloader.DataLength());
}

TEST(MNISTDataloaderTest, TestDataLengthTest)
{
    bool l_isTrain = false;
    std::string l_path("../data/mnist");
    MNISTDataloader l_dataloader(l_path, l_isTrain);
    EXPECT_EQ(10000, l_dataloader.DataLength());
}

TEST(MNISTDataloaderTest, TestDataAt)
{
    bool l_isTrain = true;
    std::string l_path("../data/mnist");
    MNISTDataloader l_dataloader(l_path, l_isTrain);

    {
        TMutableTensorPtr l_input, l_output;
        l_dataloader.DataAt(0, l_input, l_output);

        // Make sure valid pointers are returned
        EXPECT_TRUE(l_input);
        EXPECT_TRUE(l_output);

        // Test the shapes
        EXPECT_EQ(2, l_input->Shape().size());
        EXPECT_EQ(2, l_output->Shape().size());

        // 28*28 = 784
        EXPECT_EQ(1, l_input->Shape().at(0));
        EXPECT_EQ(784, l_input->Shape().at(1));
        
        // Just the singular value
        EXPECT_EQ(1, l_output->Shape().at(0));
        EXPECT_EQ(1, l_output->Shape().at(1));

        // first image is a 5
        EXPECT_EQ(5.0f, l_output->At({0, 0}));
    }

    {
        TMutableTensorPtr l_input, l_output;
        l_dataloader.DataAt(2, l_input, l_output);

        // Make sure valid pointers are returned
        EXPECT_TRUE(l_input);
        EXPECT_TRUE(l_output);

        // Test the shapes
        EXPECT_EQ(2, l_input->Shape().size());
        EXPECT_EQ(2, l_output->Shape().size());

        // 28*28 = 784
        EXPECT_EQ(1, l_input->Shape().at(0));
        EXPECT_EQ(784, l_input->Shape().at(1));
        
        // Just the singular value
        EXPECT_EQ(1, l_output->Shape().at(0));
        EXPECT_EQ(1, l_output->Shape().at(1));

        // third image is a ...
        EXPECT_EQ(4.0f, l_output->At({0, 0}));
    }
}
