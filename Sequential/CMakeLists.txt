cmake_minimum_required(VERSION 2.8.4)
project(Sequential)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(Sequential ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( Sequential ${OpenCV_LIBS} )