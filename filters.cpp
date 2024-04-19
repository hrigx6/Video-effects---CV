// Author: Hrigved Suryawanshi & Haard shah (1/19/24)
// CODE: Implemention of filters 

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>

// Function to convert a color image to greyscale
int greyscale(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Compute the average intensity across channels
            dst.at<uchar>(i, j) = (src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[1] + src.at<cv::Vec3b>(i, j)[2]) / 3;
        }
    }
    return 0;
}

// Function to apply Sepia tone effect
int Sepia(cv::Mat &src, cv::Mat &dst) {
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    cv::transform(dst, dst, cv::Matx33f(0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131));
    cv::cvtColor(dst, dst, cv::COLOR_RGB2BGR);
    return 0;
}

// Function to apply Sepia tone effect with Vignetting
int SepiaWithVignette(cv::Mat &src, cv::Mat &dst) {
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);

    // Apply Sepia transformation
    cv::transform(dst, dst, cv::Matx33f(0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131));

    // Convert back to BGR
    cv::cvtColor(dst, dst, cv::COLOR_RGB2BGR);

    // Add Vignetting
    int rows = dst.rows;
    int cols = dst.cols;
    float maxDistance = sqrt(static_cast<float>(rows * rows + cols * cols)) / 2.0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Calculate distance from the center
            float distance = sqrt(static_cast<float>((i - rows / 2) * (i - rows / 2) + (j - cols / 2) * (j - cols / 2)));

            // Vignetting factor (inverse relation with distance)
            float vignetteFactor = 1.0 - (distance / maxDistance);

            // Ensure the factor is between 0 and 1
            vignetteFactor = std::max(0.0f, std::min(1.0f, vignetteFactor));

            // Apply vignetting to each channel separately
            for (int c = 0; c < dst.channels(); ++c) {
                dst.at<cv::Vec3b>(i, j)[c] = static_cast<uchar>(dst.at<cv::Vec3b>(i, j)[c] * vignetteFactor);
            }
        }
    }

    return 0;
}

// Function to apply a 5x5 Gaussian blur filter
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    // Clone the source image for the destination
    dst = src.clone();

    // 5x5 blur kernel weights
    int kernel[5][5] = {{1, 2, 4, 2, 1},
                        {2, 4, 8, 4, 2},
                        {4, 8, 16, 8, 4},
                        {2, 4, 8, 4, 2},
                        {1, 2, 4, 2, 1}};

    // Iterate through the inner part of the image
    for (int y = 2; y < src.rows - 2; ++y) {
        for (int x = 2; x < src.cols - 2; ++x) {
            // Separate channels
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0;

                // Apply the 5x5 kernel
                for (int i = -2; i <= 2; ++i) {
                    for (int j = -2; j <= 2; ++j) {
                        sum += src.at<cv::Vec3b>(y + i, x + j)[c] * kernel[i + 2][j + 2];
                    }
                }

                // Update the destination pixel value
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(sum / 136);  // 136 is the sum of the kernel values
            }
        }
    }

    return 0;
}

// Function to apply a 5x5 Gaussian blur filter using 1D kernels
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    // Ensure the input image is not empty
    if (src.empty()) {
        return -1;
    }

    // Ensure the input image has 3 channels (BGR)
    if (src.channels() != 3) {
        return -1;
    }

    // Clone the source image for the destination
    dst = src.clone();

    // 1x5 blur kernel weights
    int kernel1x5[5] = {1, 2, 4, 2, 1};

    // Apply horizontal blur using a 1x5 filter
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 2; x < src.cols - 2; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0;
                uchar *ptr = src.ptr<uchar>(y) + (x - 2) * 3;  // Pointer to the pixel of interest
                for (int i = 0; i < 5; ++i) {
                    sum += ptr[i * 3 + c] * kernel1x5[i];
                }
                dst.ptr<uchar>(y)[x * 3 + c] = static_cast<uchar>(sum / 10);  // 10 is the sum of the kernel values
            }
        }
    }

    // Apply vertical blur using a 1x5 filter
    for (int y = 2; y < src.rows - 2; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0;
                for (int i = -2; i <= 2; ++i) {
                    sum += dst.ptr<uchar>(y + i)[x * 3 + c] * kernel1x5[i + 2];
                }
                dst.ptr<uchar>(y)[x * 3 + c] = static_cast<uchar>(sum / 10);  // 10 is the sum of the kernel values
            }
        }
    }

    return 0;
}

// Function to apply Sobel filter for edge detection (X-axis)
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst.create(src.size(), CV_16SC3);

    int sobelFilterH[3][3] = {{-1, 0, 1},
                               {-2, 0, 2},
                               {-1, 0, 1}};

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                int sumX = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        if (y + i >= 0 && y + i < src.rows && x + j >= 0 && x + j < src.cols) {
                            sumX += src.at<cv::Vec3b>(y + i, x + j)[channel] * sobelFilterH[i + 1][j + 1];
                        }
                    }
                }
                dst.at<cv::Vec3s>(y, x)[channel] = static_cast<short>(sumX);
            }
        }
    }

    return 0;
}

// Function to apply Sobel filter for edge detection (Y-axis)
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst.create(src.size(), CV_16SC3);

    int sobelFilterY[3][3] = {{-1, -2, -1},
                               {0, 0, 0},
                               {1, 2, 1}};

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                int sumY = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        if (y + i >= 0 && y + i < src.rows && x + j >= 0 && x + j < src.cols) {
                            sumY += src.at<cv::Vec3b>(y + i, x + j)[channel] * sobelFilterY[i + 1][j + 1];
                        }
                    }
                }
                dst.at<cv::Vec3s>(y, x)[channel] = static_cast<short>(sumY);
            }
        }
    }

    return 0;
}

// Function to compute the magnitude of two images
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    dst.create(sx.size(), CV_8UC3);  // Assuming you want the output to be an 8-bit image
    cv::Vec3s result;

    double maxMagnitude = 0.0;

    // Find the maximum magnitude for normalization
    for (int i = 0; i < sx.rows; i++) {
        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                double squareSum = pow(sx.at<cv::Vec3s>(i, j)[c], 2) + pow(sy.at<cv::Vec3s>(i, j)[c], 2);
                double magnitude = sqrt(squareSum);
                maxMagnitude = std::max(maxMagnitude, magnitude);
            }
        }
    }

    // Normalize and assign values to the output
    for (int i = 0; i < sx.rows; i++) {
        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                double squareSum = pow(sx.at<cv::Vec3s>(i, j)[c], 2) + pow(sy.at<cv::Vec3s>(i, j)[c], 2);
                double magnitude = sqrt(squareSum);

                // Normalize and convert to 8-bit
                result[c] = cv::saturate_cast<uchar>((magnitude / maxMagnitude) * 255.0);
            }
            dst.at<cv::Vec3b>(i, j) = result;
        }
    }

    return 0;
}
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || src.channels() != 3 || levels <= 0) {
        return -1;  // Invalid input
    }

    cv::GaussianBlur(src, dst, cv::Size(5, 5), 0);

    double bucketSize = 255.0 / levels;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                double xt = src.at<cv::Vec3b>(i, j)[c] / bucketSize;
                double xf = std::round(xt) * bucketSize;
                dst.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(xf);
            }
        }
    }

    return 0;
}

int Canny(cv::Mat &src, cv::Mat &dst, int a, int b){
    greyscale(src, dst);
    cv::Canny(dst, dst, a, b);
    return 0;
    
}

int animate(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold) {
    cv::Mat xPic, yPic, magImg, temp;

    xPic.create(src.size(), CV_32FC3);
    yPic.create(src.size(), CV_32FC3);
    magImg.create(src.size(), CV_32FC3);
    temp.create(src.size(), src.type());
    dst.create(src.size(), src.type());

    sobelX3x3(src, xPic);
    sobelY3x3(src, yPic);
    magnitude(xPic, yPic, magImg);
    blurQuantize(src, temp, levels);

    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            float avg = (magImg.at<cv::Vec3f>(i, j)[0] + magImg.at<cv::Vec3f>(i, j)[1] + magImg.at<cv::Vec3f>(i, j)[2]) / 3.0f;

            // Adjust the threshold to control the amount of cartoon effect
            if (avg > magThreshold) {
                // Increase the black intensity for a more pronounced effect
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            } else {
                dst.at<cv::Vec3b>(i, j) = temp.at<cv::Vec3b>(i, j);
            }
        }
    }

    return 0;
}

void isolateStrongColor(const cv::Mat& inputImage, cv::Mat& outputImage, const cv::Scalar& targetColor, int threshold) {
    outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC3);

    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);

            // Convert BGR to HSV
            cv::Mat3b hsvPixel;
            cv::cvtColor(cv::Mat(1, 1, CV_8UC3, pixel), hsvPixel, cv::COLOR_BGR2HSV);

            // Check if pixel hue is within a certain range of the target hue
            int hueDiff = std::min(std::abs(hsvPixel(0, 0)[0] - targetColor[0]), 180 - std::abs(hsvPixel(0, 0)[0] - targetColor[0]));
            if (hueDiff < threshold) {
                outputImage.at<cv::Vec3b>(y, x) = pixel;
            }
        }
    }
}
//function to invert colours of the image
void invertColors(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) {
        return;
    }

    dst.create(src.size(), src.type());

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            for (int c = 0; c < src.channels(); ++c) {
                dst.at<cv::Vec3b>(i, j)[c] = 255 - src.at<cv::Vec3b>(i, j)[c];
            }
        }
    }
}
//function for high pass filter
void highPassFilter(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) {
        return;
    }

    dst.create(src.size(), src.type());

    for (int i = 1; i < src.rows - 1; ++i) {
        for (int j = 1; j < src.cols - 1; ++j) {
            for (int c = 0; c < src.channels(); ++c) {
                int sum = 8 * src.at<cv::Vec3b>(i, j)[c]
                          - src.at<cv::Vec3b>(i - 1, j - 1)[c] - src.at<cv::Vec3b>(i - 1, j)[c] - src.at<cv::Vec3b>(i - 1, j + 1)[c]
                          - src.at<cv::Vec3b>(i, j - 1)[c] - src.at<cv::Vec3b>(i, j + 1)[c]
                          - src.at<cv::Vec3b>(i + 1, j - 1)[c] - src.at<cv::Vec3b>(i + 1, j)[c] - src.at<cv::Vec3b>(i + 1, j + 1)[c];

                dst.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(std::max(0, std::min(255, sum / 8)));
            }
        }
    }
}

