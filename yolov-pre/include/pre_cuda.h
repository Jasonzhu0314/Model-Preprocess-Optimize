#pragma once
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"

#define CHECK_CUDA_ERROR(call) \
do { \
    const cudaError_t error=call; \
    if (error != cudaSuccess) { \
        printf("ERROR: %s:%d,", __FILE__,__LINE__);\
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));\
        exit(1);\
    } \
} while(0)

#define CHECK_RUN() \
do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        printf("CUDA 错误：%s\n", cudaGetErrorString(error)); \
    } \
} while(0)

namespace cudapre {

void gpu_resize(
    uint8_t* image, 
    uint8_t* outImage,
    cudaStream_t stream, 
    uint32_t in_width,
    uint32_t in_height,
    uint32_t width = 640,
    uint32_t height = 640
);

void gpu_letterbox(uint8_t* image, 
                   uint8_t* outImage,
                   cudaStream_t stream, 
                   uint32_t in_width,
                   uint32_t in_height,
                   uint32_t width = 640,
                   uint32_t height = 640
                   );



void cpu_resize(uint8_t* src, 
                uint8_t* dst,
                uint32_t src_width, 
                uint32_t src_height,
                uint32_t out_width,
                uint32_t out_height);


void cpu_copymakeborder(uint8_t* src, 
                uint8_t* dst,
                uint32_t out_width,
                uint32_t out_height,
                uint32_t in_width,
                uint32_t in_height,
                uint8_t border_value);

void gpu_copymakeborder(uint8_t *image, 
                uint8_t* outImage,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height,
                uint32_t out_width,
                uint32_t out_height);

void gpu_normalize(uint8_t *image, 
                float* out_image,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height,
                float c1, float c2, float c3);

void gpu_hwc2chw(float* image, 
                float* out_image,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height);
};
