cmake_minimum_required(VERSION 3.10)
project(test)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
# include 
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS} .)
add_executable(test main.cpp  myfunc.cpp histogram.hpp)
#link 
target_link_libraries(test ${OpenCV_LIBS})