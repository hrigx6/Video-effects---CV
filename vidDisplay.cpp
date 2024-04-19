// Author: Hrigved Suryawanshi & Hard shah (1/19/24)
// CODE: Read and display a video with implemented filters 

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "filters.cpp"  // Assuming this file contains your filter functions
#include "filters.h"   // Assuming this file contains your filter function declarations
#include "faceDetect.h" // Assuming this file contains your face detection functions

using namespace cv;

int main(int argc, char *argv[]) {
    bool faceDetection = false; // Flag to enable/disable face detection
    cv::VideoCapture *capdev;
    std::vector<cv::Rect> faces; // Vector to store detected faces
    cv::Mat grey;  // Declare the grey variable

    // Open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    // Get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // Identifies a window
    cv::Mat frame;
    cv::Mat filter;
    cv::Mat sobelXResult, sobelYResult;
    int colorState = 0; // Variable to track color state (if needed)

    for (;;) {
        *capdev >> frame; // Get a new frame from the camera, treat as a stream

        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        cv::imshow("Video", frame);

        if (faceDetection) {
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);  // Convert frame to greyscale
            cv::Rect last(0, 0, 0, 0);
            detectFaces(grey, faces); // Detect faces in the greyscale frame
            drawBoxes(frame, faces);  // Draw rectangles around detected faces

            if (faces.size() > 0) {
                last.x = (faces[0].x + last.x) / 2;
                last.y = (faces[0].y + last.y) / 2;
                last.width = (faces[0].width + last.width) / 2;
                last.height = (faces[0].height + last.height) / 2;
            }

            cv::imshow("Video", frame);
        }

        // See if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;  // Break the loop if 'q' key is pressed
        } else if (key == 'g') {
            cv::Mat filter;
            cv::cvtColor(frame, filter, cv::COLOR_RGBA2GRAY);
            printf("Greyscale_Video");
            cv::imshow("Greyscale_Video", filter); // Display the greyscale video
        } else if (key == 'h') {
            greyscale(frame, filter);
            printf("Alternate_Greyscale");
            cv::imshow("Alternate_Greyscale", filter);
        } else if (key == 'o') {
            Sepia(frame, filter);
            cv::imshow("sepia_Video", filter);
        } else if (key == 'v') {
            SepiaWithVignette(frame, filter);
            cv::imshow("sepia_X_vignette", filter);
        } else if (key == 'b') {
            blur5x5_1(frame, filter);
            cv::imshow("blur_1", filter);
        } else if (key == 'n') {
            blur5x5_2(frame, filter);
            cv::imshow("blur_2", filter);
        } else if (key == 'x') {
            sobelX3x3(frame, sobelXResult);
            cv::convertScaleAbs(sobelXResult, sobelXResult);  // Convert to 8-bit for display
    	    cv::imshow("Sobel X", sobelXResult);  // Corrected from filter to sobelXResult
        } else if (key == 'y') {
    	    sobelY3x3(frame, sobelYResult);
    	    cv::convertScaleAbs(sobelYResult, sobelYResult);  // Convert to 8-bit for display
   	    cv::imshow("Sobel Y", sobelYResult);  // Corrected from filter to sobelYResult
        } 
        // applying gradient magnitude filter
          else if (key == 'm') {
            sobelX3x3(frame, sobelXResult);
            sobelY3x3(frame, sobelYResult);
            magnitude(sobelXResult, sobelYResult, filter);
            cv::convertScaleAbs( filter, frame );
            cv::imshow("Gradient_Magnitude_Video", filter);
        } else if (key == 'l') {
            blurQuantize(frame, filter, 15);
            cv::imshow("Blur_Quantize_Video", filter);
        
        } else if (key == 'a') {
            animate(frame, filter, 15, 15);
            cv::imshow("animated_Video", filter);
        } else if (key == 'e') {
            Canny(frame, filter, 80, 160);
            cv::imshow("Canny_Video", filter);
        } else if (key == 'z') {
    		// Convert the frame to HSV before calling the isolate function
    	     cv::Mat hsvFrame;
    	     cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

    		// Call the function to isolate color
    	     cv::Mat isolatedFrame;
    	     cv::Scalar targetColor(0, 255, 255);  //  Red color
    	     int threshold = 5;
    	     isolateStrongColor(frame, isolatedFrame, targetColor, threshold);

    	     cv::imshow("Color Isolation", isolatedFrame);
}        
          else if (key == 'f') {
             faceDetection = !faceDetection; // Toggle face detection on/off
        } else if (key == 'i') {
             invertColors(frame, filter);
             cv::imshow( "invertedcolors" , filter );
        } else if (key == 'p') {
    	     highPassFilter(frame, filter);
   	     cv::imshow("High Pass Filter", filter);
}
    }

    // Clean up resources
    delete capdev;

    return 0;
}

