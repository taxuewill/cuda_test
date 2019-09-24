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
    char * nv12Data = new char[image.cols*image.rows*3/2];
    ifstream * inputFile = new ifstream("./boldt_nv12.yuv");
    ofstream * outputFile = new ofstream("./rgba.data");
    inputFile->read((char *)nv12Data,image.cols*image.rows*3/2);
    char * rgbaData = new char[image.cols*image.rows*4];
    cudaNV12ToRGBA(nv12Data,image.cols,rgbaData,image.cols*4,image.cols,image.rows);
    outputFile->write((char *)rgbaData,image.cols*image.rows*4);

    delete inputFile;
    delete outputFile;
    cv::waitKey(10 * 1000);
    return 0;
}