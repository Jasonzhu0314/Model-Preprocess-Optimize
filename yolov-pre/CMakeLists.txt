cmake_minimum_required(VERSION 3.18)
project(yolo-pre)

# 设置opencv
find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)
# 设置 CUDA 架构
# set(CUDA_ARCH_LIST Auto CACHE STRING "List of CUDA architectures (e.g. 7.0, 7.5, 8.0)")
# string(REPLACE "." "" CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_ARCH_FLAGS "70")
set(CUDA_NVCC_FLAGS "-gencode arch=compute_${CUDA_ARCH_FLAGS},code=sm_${CUDA_ARCH_FLAGS}")


set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB SOURCES_FILE ${SOURCE_DIR}/*)
include_directories("${PROJECT_SOURCE_DIR}/include")

find_package(Boost REQUIRED COMPONENTS filesystem)

# set(BOOST_ROOT "/usr/lib")
# set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/x86_64-linux-gnu")

# find_package(Boost REQUIRED)
# if(Boost_FOUND)
    # include_directories(${Boost_INCLUDE_DIRS})
# endif()
# message(STATUS ${Boost_LIBRARIES})

cuda_add_executable(yolo-pre ${SOURCES_FILE})

target_link_libraries(yolo-pre ${OpenCV_LIBS})
target_link_libraries(yolo-pre Boost::filesystem)




