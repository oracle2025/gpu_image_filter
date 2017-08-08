cmake_minimum_required(VERSION 3.1)
project(gpu_image_filter)

find_package( OpenCV CONFIG REQUIRED )

#find_package( OpenCL REQUIRED )

find_package( Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

#include_directories(${OpenCL_INCLUDE_DIRS})

include_directories("${PROJECT_BINARY_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include/proxy-opencl")

#set(CLEW_LIBS ${PROJECT_BINARY_DIR}/lib/libclew.a)
set(CLEW_LIBS clew dl)
link_directories("${PROJECT_BINARY_DIR}/lib")

include(ExternalProject)

ExternalProject_Add(libclew
	GIT_REPOSITORY https://github.com/oracle2025/clew.git
	INSTALL_DIR ${PROJECT_BINARY_DIR}
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DINSTALL_CL_HEADER=ON)

add_subdirectory(src)

