/*
 * Tensor Math Test
 *
 */

#include "neural/math/tensor_math.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(TensorMathTest, TestMatMul)
{
    TTensorPtr lhs = Tensor::New({2,2}, {
        4.0, 3.0,
        2.0, 1.0
    });

    TTensorPtr rhs = Tensor::New({2,2}, {
        1.0, 2.0,
        3.0, 4.0
    });
    
    TTensorPtr result = TensorMath::Multiply(lhs, rhs);
    EXPECT_EQ(2, result->Shape().at(0));
    EXPECT_EQ(2, result->Shape().at(1));

    /*
    (0,0) = 4*1 + 3*3 = 13
    (0,1) = 4*2 + 3*4 = 20
    (1,0) = 2*1 + 1*3 = 5
    (1,1) = 2*2 + 1*4 = 8
    */

    EXPECT_EQ(13.0, result->At({0,0}));
    EXPECT_EQ(20.0, result->At({0,1}));
    EXPECT_EQ(5.0,  result->At({1,0}));
    EXPECT_EQ(8.0,  result->At({1,1}));
}

TEST(TensorMathTest, TestTranspose)
{
    TTensorPtr mat = Tensor::New({2,3}, {
        5.0, 4.0, 3.0,
        2.0, 1.0, 0.0
    });

    
    TTensorPtr transpose = TensorMath::Transpose(mat);
    EXPECT_EQ(3, transpose->Shape().at(0));
    EXPECT_EQ(2, transpose->Shape().at(1));

    /*
    5.0, 2.0,
    4.0, 1.0,
    3.0, 0.0
    */

    EXPECT_EQ(5.0, transpose->At({0,0}));
    EXPECT_EQ(2.0, transpose->At({0,1}));
    EXPECT_EQ(4.0, transpose->At({1,0}));
    EXPECT_EQ(1.0, transpose->At({1,1}));
    EXPECT_EQ(3.0, transpose->At({2,0}));
    EXPECT_EQ(0.0, transpose->At({2,1}));
}

TEST(TensorMathTest, TestAddCol)
{
    TTensorPtr mat = Tensor::New({3,5}, {
        5.0, 4.0, 3.0, 2.0, 1.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        -1.0, -2.0, -3.0, -4.0, -5.0
    });

    
    TTensorPtr newMat = TensorMath::AddCol(mat, 1.0);
    EXPECT_EQ(3, newMat->Shape().at(0));
    EXPECT_EQ(6, newMat->Shape().at(1));

    EXPECT_EQ( 5.0, newMat->At({0,0}));
    EXPECT_EQ( 4.0, newMat->At({0,1}));
    EXPECT_EQ( 3.0, newMat->At({0,2}));
    EXPECT_EQ( 2.0, newMat->At({0,3}));
    EXPECT_EQ( 1.0, newMat->At({0,4}));
    EXPECT_EQ( 1.0, newMat->At({0,5}));

    EXPECT_EQ( 1.0, newMat->At({1,0}));
    EXPECT_EQ( 2.0, newMat->At({1,1}));
    EXPECT_EQ( 3.0, newMat->At({1,2}));
    EXPECT_EQ( 4.0, newMat->At({1,3}));
    EXPECT_EQ( 5.0, newMat->At({1,4}));
    EXPECT_EQ( 1.0, newMat->At({1,5}));
    
    EXPECT_EQ(-1.0, newMat->At({2,0}));
    EXPECT_EQ(-2.0, newMat->At({2,1}));
    EXPECT_EQ(-3.0, newMat->At({2,2}));
    EXPECT_EQ(-4.0, newMat->At({2,3}));
    EXPECT_EQ(-5.0, newMat->At({2,4}));
    EXPECT_EQ( 1.0, newMat->At({2,5}));
}

TEST(TensorMathTest, TestRemoveCol)
{
    TTensorPtr mat = Tensor::New({3,5}, {
        5.0, 4.0, 3.0, 2.0, 1.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        -1.0, -2.0, -3.0, -4.0, -5.0
    });

    
    TTensorPtr newMat = TensorMath::RemoveCol(mat);
    EXPECT_EQ(3, newMat->Shape().at(0));
    EXPECT_EQ(4, newMat->Shape().at(1));

    EXPECT_EQ( 5.0, newMat->At({0,0}));
    EXPECT_EQ( 4.0, newMat->At({0,1}));
    EXPECT_EQ( 3.0, newMat->At({0,2}));
    EXPECT_EQ( 2.0, newMat->At({0,3}));
    EXPECT_EQ( 1.0, newMat->At({1,0}));
    EXPECT_EQ( 2.0, newMat->At({1,1}));
    EXPECT_EQ( 3.0, newMat->At({1,2}));
    EXPECT_EQ( 4.0, newMat->At({1,3}));
    EXPECT_EQ(-1.0, newMat->At({2,0}));
    EXPECT_EQ(-2.0, newMat->At({2,1}));
    EXPECT_EQ(-3.0, newMat->At({2,2}));
    EXPECT_EQ(-4.0, newMat->At({2,3}));
}

TEST(TensorMathTest, TestAddRow)
{
    TTensorPtr mat = Tensor::New({3,5}, {
        5.0, 4.0, 3.0, 2.0, 1.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        -1.0, -2.0, -3.0, -4.0, -5.0
    });

    
    TTensorPtr newMat = TensorMath::AddRow(mat, 1.0);
    EXPECT_EQ(4, newMat->Shape().at(0));
    EXPECT_EQ(5, newMat->Shape().at(1));

    EXPECT_EQ( 5.0, newMat->At({0,0}));
    EXPECT_EQ( 4.0, newMat->At({0,1}));
    EXPECT_EQ( 3.0, newMat->At({0,2}));
    EXPECT_EQ( 2.0, newMat->At({0,3}));
    EXPECT_EQ( 1.0, newMat->At({0,4}));

    EXPECT_EQ( 1.0, newMat->At({1,0}));
    EXPECT_EQ( 2.0, newMat->At({1,1}));
    EXPECT_EQ( 3.0, newMat->At({1,2}));
    EXPECT_EQ( 4.0, newMat->At({1,3}));
    EXPECT_EQ( 5.0, newMat->At({1,4}));
    
    EXPECT_EQ(-1.0, newMat->At({2,0}));
    EXPECT_EQ(-2.0, newMat->At({2,1}));
    EXPECT_EQ(-3.0, newMat->At({2,2}));
    EXPECT_EQ(-4.0, newMat->At({2,3}));
    EXPECT_EQ(-5.0, newMat->At({2,4}));

    EXPECT_EQ(1.0, newMat->At({3,0}));
    EXPECT_EQ(1.0, newMat->At({3,1}));
    EXPECT_EQ(1.0, newMat->At({3,2}));
    EXPECT_EQ(1.0, newMat->At({3,3}));
    EXPECT_EQ(1.0, newMat->At({3,4}));
}

TEST(TensorMathTest, TestRemoveRow)
{
    TTensorPtr mat = Tensor::New({3,5}, {
        5.0, 4.0, 3.0, 2.0, 1.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        -1.0, -2.0, -3.0, -4.0, -5.0
    });

    TTensorPtr newMat = TensorMath::RemoveRow(mat);
    EXPECT_EQ(2, newMat->Shape().at(0));
    EXPECT_EQ(5, newMat->Shape().at(1));

    EXPECT_EQ( 5.0, newMat->At({0,0}));
    EXPECT_EQ( 4.0, newMat->At({0,1}));
    EXPECT_EQ( 3.0, newMat->At({0,2}));
    EXPECT_EQ( 2.0, newMat->At({0,3}));
    EXPECT_EQ( 1.0, newMat->At({0,4}));

    EXPECT_EQ( 1.0, newMat->At({1,0}));
    EXPECT_EQ( 2.0, newMat->At({1,1}));
    EXPECT_EQ( 3.0, newMat->At({1,2}));
    EXPECT_EQ( 4.0, newMat->At({1,3}));
    EXPECT_EQ( 5.0, newMat->At({1,4}));
}

