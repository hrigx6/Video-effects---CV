
# Video Special Effects

## Description
The Video Special Effects project is an implementation of various image processing filters and effects using the C/C++ OpenCV package. The project aims to provide hands-on experience with basic image manipulation tasks, live video streaming, face detection, and the application of creative filters and effects to enhance visual elements in real-time video streams.

## Repository Structure
- **vidDisplay.cpp**: The main file responsible for running the filters and displaying the video stream.
- **filters.cpp**: Contains the implementations of various image processing functions and filters.
- **filter.h**: Header file to source filters.cpp in the main file.

## Implemented Filters and Effects
1. **Greyscale Conversion**: Converts live video frames to greyscale.
2. **Alternative Greyscale Conversion**: Converts live video frames to greyscale using an alternative method.
3. **Sepia Tone Filter**: Applies a sepia tone effect to live video frames, giving them an antique camera look.
4. **5x5 Gaussian Blur Filter**:
    - **Implementation 1**: Naïve implementation of a 5x5 blur filter from scratch.
    - **Implementation 2**: 5x5 Gaussian blur using separable 1x5 filters.
5. **3x3 Sobel X and Sobel Y Filters**: Applies Sobel filters to detect edges in live video frames.
6. **Gradient Magnitude Image**: Generates a gradient magnitude image from the X and Y Sobel images.
7. **Blur and Quantize Color Image**: Blurs and quantizes live video frames into a fixed number of levels.
8. **Face Detection**: Detects faces in live video frames and draws bounding boxes around them.
9. **Additional Effects**:
    - **Canny Edge Filter**: Detects significant edges in live video frames using the Canny edge detection technique.
    - **Animated Effect**: Stylizes and simplifies live video frames, resembling cartoons.
    - **Color Isolation**: Highlights specific colors in live video frames while desaturating the rest.

## Usage
1. Compile the main file `vidDisplay.cpp` along with `filters.cpp` using your preferred C/C++ compiler.
2. Ensure that the OpenCV library is properly linked during compilation.
3. Run the compiled executable to apply various filters and effects to live video streams.

