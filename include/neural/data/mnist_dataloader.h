/*
 * MNISTDataloader
 */

#pragma once

#include "neural/math/tensor.h"

#include <string>

namespace neural
{

class MNISTDataloader
{
public:
    MNISTDataloader(
        const std::string& a_path,
        bool a_isTrain = true);

    // Total number of examples
    size_t DataLength() const;

    // Get specific example
    bool DataAt(
        size_t i,
        TMutableTensorPtr& a_outInput,
        TMutableTensorPtr& a_outOutput) const;

private:
    // Total number of examples
    size_t m_numData;

    // Image sizes
    size_t m_imageWidth;
    size_t m_imageHeight;

    // Files for images and labels
    std::string m_imageFile;
    std::string m_labelFile;

    // Helpers functions
    bool p_FileExists(const std::string& a_file) const;
    int32_t p_ReverseInt(int32_t a_int) const;
    size_t p_ReadNumImages(const std::string& a_file) const;
    size_t p_ReadImageWidth(const std::string& a_file) const;
    size_t p_ReadImageHeight(const std::string& a_file) const;
    size_t p_ReadIntAt(const std::string& a_file, size_t a_idx) const;
    float p_TransformToInterval(
        float a_input, float a_oldMin, float a_oldMax,
        float a_newMin, float a_newMax) const;
};

} // namespace neural
