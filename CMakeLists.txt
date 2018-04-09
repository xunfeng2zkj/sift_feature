cmake_minimum_required(VERSION 2.8.11)

#global variables
set(PROJ_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_BUILD_TYPE"Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g -ggdb")  
#compilation flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG"${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC -msse4.1 -02 - fopenmp")
add_definitions("-std=c++11")

#output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

#global dependencies
set(OpenCV_DIR /home/zhukj1/soft/opencv-2.4.13/share/OpenCV)
find_package(OpenCV 2 REQUIRED)

set(LIB_DEPS ${OpenCV_LIBS})

include_directories(
    #	${OpenCV_INCLUDE_DIRS}
	/home/zhukj1/soft/opencv-2.4.13/include
	/home/zhukj1/soft/opencv-2.4.13/include/opencv
	/home/zhukj1/soft/opencv-2.4.13/include/opencv2
	/home/zhukj1/UBSelect/include/	
)
link_directories(
)
add_executable(test
	src/demo.cpp
	)
target_link_libraries(test ${LIB_DEPS})
