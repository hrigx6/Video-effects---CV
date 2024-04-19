
#ifndef filters_hpp
#define filters_hpp

#include <stdio.h>
#include <opencv2/core.hpp>

int greyscale(cv::Mat &src, cv::Mat &dst);

int Sepia(cv::Mat &src, cv::Mat &dst);

int SepiaWithVignette(cv::Mat &src, cv::Mat &dst);

int blur5x5_1( cv::Mat &src, cv::Mat &dst );

int blur5x5_2( cv::Mat &src, cv::Mat &dst );

int sobelX3x3( cv::Mat &src, cv::Mat &dst );

int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

int animate( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

int Canny(cv::Mat &src, cv::Mat &dst, int a, int b);

void isolateStrongColor(const cv::Mat& inputImage, cv::Mat& outputImage, const cv::Scalar& targetColor, int threshold);

void invertColors(const cv::Mat& src, cv::Mat& dst);

void highPassFilter(const cv::Mat& src, cv::Mat& dst);
#endif /* filters_hpp */
