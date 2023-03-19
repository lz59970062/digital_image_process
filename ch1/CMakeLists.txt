cmake_minimum_required(VERSION 3.10)

project(cvch1)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
# include 
include_directories(&{OpenCV_INCLUDE_DIRS})

add_executable(cvch1 main.cpp)
#link 

target_link_libraries(cvch1 &{OpenCV_LIBS}qw
)