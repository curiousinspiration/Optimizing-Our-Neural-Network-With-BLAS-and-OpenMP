/*
 * MNISTDataloader Implementation
 */

#include "neural/data/mnist_dataloader.h"

#include <glog/logging.h>

#include <sstream>
#include <fstream>

using namespace std;

namespace neural
{

MNISTDataloader::MNISTDataloader(const std::string& a_path, bool a_isTrain)
{
    // Determine the file prefix depending on if it is train or test data
    string l_filePrefix = "train";
    if (!a_isTrain)
    {
        l_filePrefix = "t10k";
    }

    // Define the image file path
    m_imageFile = a_path + "/" + l_filePrefix + "-images-idx3-ubyte";
    if (!p_FileExists(m_imageFile))
    {
        LOG(ERROR) << "Image file does not exist: "
                   << m_imageFile << endl;
        return;
    }

    // Define the label file path
    m_labelFile = a_path + "/" + l_filePrefix + "-labels-idx1-ubyte";
    if (!p_FileExists(m_labelFile))
    {
        LOG(ERROR) << "Label file does not exist: "
                   << m_labelFile << endl;
        return;
    }

    LOG(INFO) << "Got Image File: " << m_imageFile << endl;
    LOG(INFO) << "Got Label File: " << m_labelFile << endl;

    // Read num data
    m_numData = p_ReadNumImages(m_imageFile);
    LOG(INFO) << "Got Number of Images: " << m_numData << endl;

    // Read image sizes
    m_imageWidth = p_ReadImageWidth(m_imageFile);
    m_imageHeight = p_ReadImageHeight(m_imageFile);

    LOG(INFO) << "Image size: " << m_imageWidth << "x" << m_imageHeight << endl;
}

size_t MNISTDataloader::DataLength() const
{
    return m_numData;
}

bool MNISTDataloader::DataAt(
    size_t a_dataIdx,
    TMutableTensorPtr& a_outInput,
    TMutableTensorPtr& a_outOutput) const
{
    /*
    http://yann.lecun.com/exdb/mnist/

    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    0016     unsigned byte   ??               pixel 
    0017     unsigned byte   ??               pixel 
    ........ 
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
    0004     32 bit integer  60000            number of items 
    0008     unsigned byte   ??               label 
    0009     unsigned byte   ??               label 
    ........ 
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    */
    
    if (a_dataIdx >= DataLength())
    {
        LOG(ERROR) << "MNISTDataloader::DataAt cannot access data at ["
                   << a_dataIdx << "] >= " << DataLength() << endl;
        return false;
    }

    // Read image data
    vector<uint8_t> l_imageData;
    size_t l_bytesPerData = m_imageWidth * m_imageWidth;
    l_imageData.resize(l_bytesPerData);
    {
        size_t l_headerBytes = sizeof(int32_t) * 4; // skip magic number, num images, and image sizes
        size_t l_dataOffset = (a_dataIdx * l_bytesPerData) + l_headerBytes;
        ifstream l_infile;
        l_infile.open(m_imageFile, ios::binary | ios::in); 
        l_infile.seekg(l_dataOffset, ios::beg); // move n bytes into the file 
        l_infile.read(reinterpret_cast<char*>(l_imageData.data()), sizeof(uint8_t) * l_bytesPerData);
        l_infile.close();
    }

    // Read label data
    uint8_t l_label;
    {
        size_t l_headerBytes = sizeof(int32_t) * 2;
        size_t l_dataOffset = l_headerBytes + a_dataIdx;
        ifstream l_infile;
        l_infile.open(m_labelFile, ios::binary | ios::in); 
        l_infile.seekg(l_dataOffset, ios::beg); // move n bytes into the file 
        l_infile.read(reinterpret_cast<char*>(&l_label), sizeof(uint8_t));
        l_infile.close();
    }

    a_outInput = Tensor::Zeros({1, m_imageWidth*m_imageHeight})->ToMutable();
    a_outOutput = Tensor::Zeros({1, 1})->ToMutable();

    float l_class = (float)l_label;

    a_outOutput->SetAt({0, 0}, l_class);

    for (size_t i = 0; i < m_imageHeight; ++i)
    {
        for (size_t j = 0; j < m_imageWidth; ++j)
        {
            size_t l_imgDataOffset = (i * m_imageWidth) + j;
            float l_val = (float)l_imageData.at(l_imgDataOffset);
            a_outInput->SetAt({0, l_imgDataOffset}, p_TransformToInterval(l_val, 0.0, 255.0, -1.0, 1.0));
        }
    }
    return true;
}

/*
http://yann.lecun.com/exdb/mnist/
All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors.
Users of Intel processors and other low-endian machines must flip the bytes of the header.

p_ReverseInt() impl from
https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
*/
int32_t MNISTDataloader::p_ReverseInt(int32_t a_int) const
{
    unsigned char c1, c2, c3, c4;

    c1 = a_int & 255;
    c2 = (a_int >> 8) & 255;
    c3 = (a_int >> 16) & 255;
    c4 = (a_int >> 24) & 255;

    return ((int32_t)c1 << 24) + ((int32_t)c2 << 16) + ((int32_t)c3 << 8) + c4;
}

size_t MNISTDataloader::p_ReadNumImages(const std::string& a_file) const
{
    /*
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    */
    return p_ReadIntAt(a_file, 1); // num images is the 2nd line
}

size_t MNISTDataloader::p_ReadImageHeight(const std::string& a_file) const
{
    /*
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    */
    return p_ReadIntAt(a_file, 2); // height is num rows @ 2
}

size_t MNISTDataloader::p_ReadImageWidth(const std::string& a_file) const
{
    /*
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    */
    return p_ReadIntAt(a_file, 3); // width is num cols @ 3
}

size_t MNISTDataloader::p_ReadIntAt(const std::string& a_file, size_t a_idx) const
{
    /*
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    */
    size_t l_dataOffset = sizeof(int32_t) * a_idx; // skip header
    ifstream l_infile;
    l_infile.open(a_file, ios::binary | ios::in); 
    l_infile.seekg(l_dataOffset, ios::beg); // move n bytes into the file 
    int32_t l_val;
    l_infile.read(reinterpret_cast<char*>(&l_val), sizeof(int32_t));
    l_infile.close();
    l_val = p_ReverseInt(l_val);
    return l_val;
}

bool MNISTDataloader::p_FileExists(const std::string& a_file) const
{
    std::ifstream l_file(a_file);
    return l_file.good();
}

float MNISTDataloader::p_TransformToInterval(
    float a_input, float a_oldMin, float a_oldMax,
    float a_newMin, float a_newMax) const
{
    return (((a_input - a_oldMin) * (a_newMax - a_newMin)) / (a_oldMax - a_oldMin)) + a_newMin;
}

} // namespace neural
