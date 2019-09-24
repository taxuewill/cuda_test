#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "constants.h"
#include "../cuda/rgb2yuv.cuh"
using namespace std;
int main()
{
    cv::Mat image = cv::imread(IMAGE_BOLDT);
    // display original image
	cv::namedWindow("Original Image");
    cv::imshow("Original Image",image);
    uint8_t * pOutput = new uint8_t[image.cols*image.rows*3/2];
    rgb2NV12((char *)image.data,pOutput,image.cols,image.rows);
    std::ofstream * pFile = new std::ofstream("./boldt_nv12.yuv");
    pFile->write((char *)pOutput,image.cols*image.rows*3/2);
    delete pFile;
    cv::waitKey(10 * 1000);
    return 0;
}