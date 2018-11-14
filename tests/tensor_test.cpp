/*
 * Tensor test
 *
 */

#include "neural/math/tensor.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(TensorTest, TestConstructorNoData)
{
    // Create 3x4x5 tensor
    Tensor t({3,4,5});

    // Make sure vector data is correct size
    EXPECT_EQ(60, t.Data().size());

    // Make sure helper function returns correct siz
    EXPECT_EQ(60, t.Size());

    // Make sure we saved the shape
    vector<size_t> l_shape = t.Shape();
    EXPECT_EQ(3, l_shape.at(0));
    EXPECT_EQ(4, l_shape.at(1));
    EXPECT_EQ(5, l_shape.at(2));
}

TEST(TensorTest, TestConstructorData)
{
    // Create 2x2 tensor with data:
    /*
      1.0 2.0
      3.0 4.0
    */
    Tensor t({2,2}, {1.0, 2.0, 3.0, 4.0});

    // Make sure we saved the size
    vector<size_t> l_shape = t.Shape();
    EXPECT_EQ(2, l_shape.at(0));
    EXPECT_EQ(2, l_shape.at(1));
    
    // Make sure data is valid
    const vector<float>& l_data = t.Data();
    EXPECT_EQ(4, l_data.size());
  
    EXPECT_EQ(1.0, l_data.at(0));
    EXPECT_EQ(2.0, l_data.at(1));
    EXPECT_EQ(3.0, l_data.at(2));
    EXPECT_EQ(4.0, l_data.at(3));
}

TEST(TensorTest, TestGetAtIdx)
{
    // Fake image batch data of 2x2 color images
    // batch_size x img_width x img_height x img_channels
    vector<size_t> l_size = {4,2,2,3};

    Tensor t(l_size, {
        // img 0
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0,       // pixel (0, 0), pixel (0, 1)
        6.0, 7.0, 8.0, 9.0, 10.0, 11.0,     // pixel (1, 0), pixel (1, 1)

        // img 1 
        12.0, 13.0, 14.0, 15.0, 16.0, 17.0, // pixel (0, 0), pixel (0,1)
        18.0, 19.0, 20.0, 21.0, 22.0, 23.0, // pixel (1, 0), pixel (1, 1)

        // img 2
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, // pixel (0, 0), pixel (0, 1)
        30.0, 31.0, 32.0, 33.0, 34.0, 35.0, // pixel (1, 0), pixel (1, 1)

        // img 3
        36.0, 37.0, 38.0, 39.0, 40.0, 41.0, // pixel (0, 0), pixel (0, 1)
        42.0, 43.0, 44.0, 45.0, 46.0, 47.0, // pixel (1, 0), pixel (1, 1)
    });

    EXPECT_EQ(0.0,  t.At({0, 0, 0, 0})); // first pixel
    EXPECT_EQ(9.0,  t.At({0, 1, 1, 0})); // image 0, row 1, col 1, channel, 0
    EXPECT_EQ(14.0, t.At({1, 0, 0, 2})); // image 1, row 0, col 0, channel, 2
    EXPECT_EQ(28.0, t.At({2, 0, 1, 1})); // image 2, row 0, col 1, channel, 1
    EXPECT_EQ(42.0, t.At({3, 1, 0, 0})); // image 3, row 1, col 0, channel, 0
}

