cmake_minimum_required(VERSION 2.8)
project(DisplayImage)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imgDisplay src/imgDisplay.cpp)
target_link_libraries(imgDisplay PRIVATE ${OpenCV_LIBS})

add_executable(vidDisplay src/vidDisplay.cpp src/faceDetect.cpp)  # Add faceDetect.cpp here
target_link_libraries(vidDisplay PRIVATE ${OpenCV_LIBS})

