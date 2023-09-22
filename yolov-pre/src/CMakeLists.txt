# 设置opencv
find_package(OpenCV REQUIRED)

# find_package(CUDA REQUIRED)
# 设置 CUDA 架构
# set(CUDA_ARCH_LIST Auto CACHE STRING "List of CUDA architectures (e.g. 7.0, 7.5, 8.0)")
# string(REPLACE "." "" CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SOURCES_FILE ${SOURCE_DIR}/*.c*)

message(${SOURCES_FILE})
cuda_add_library(cvpre SHARED ${SOURCES_FILE})

target_include_directories(cvpre PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cvpre ${OpenCV_LIBS})
# find_package(Boost REQUIRED COMPONENTS filesystem)
# target_link_libraries(yolo-pre Boost::filesystem)


