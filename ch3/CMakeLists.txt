cmake_minimum_required(VERSION 3.10)
project(cvch3)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
# include 
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS} .)
add_executable(cvch3 main.cpp  myfunc.cpp)
#link 
target_link_libraries(cvch3 ${OpenCV_LIBS})