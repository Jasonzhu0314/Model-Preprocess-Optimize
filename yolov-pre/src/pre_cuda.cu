#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include "pre_cuda.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"


inline int divUp(int a, int b) 
{
    assert(b > 0);
    return ceil((float) a / b);
};


__device__ inline void copymakeborder_op(
    uint8_t* src, uint8_t* dst,
    int top, int left, int out_width, 
    int out_height, uint8_t border_value
) {

}

__global__ void copymakeborder_kernel(
                uint8_t *image, uint8_t* out_image,
                uint32_t in_width, uint32_t in_height,
                uint32_t out_width, uint32_t out_height, 
                int top,
                int left,
                uint8_t border_val) 
{
    const int sx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx < out_width && sy < out_height) {
        uint8_t* out_ptr = out_image + sy * out_width * 3 + sx * 3;
        if (sx < left || sy < top || sx >= in_width + left || sy >= in_height + top) {
            out_ptr[0] = border_val;
            out_ptr[1] = border_val;
            out_ptr[2] = border_val;
        } else {
            uint8_t* in_ptr = image + (sy - top) * in_width * 3 + (sx - left) * 3;
            out_ptr[0] = in_ptr[0];
            out_ptr[1] = in_ptr[1];
            out_ptr[2] = in_ptr[2];
        }
    }

}

__global__ void resize_op(uint8_t* src, uint8_t* dst,
                            float scale_x, float scale_y, int src_width, 
                            int src_height, int out_width, int out_height) 
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;

    if ((dst_x < out_width) && (dst_y < out_height))
    {
        //y coordinate
        // 原图的y坐标, +0.5到像素坐标中心，否则是像素的左上角
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        // 左上角的y, 向下取整
        int   sy = floor(fy);
        fy -= sy;
        //  防止越界
        sy = max(0, min(sy, src_height - 2));
        //row pointers
        // sy,sx*3--BGR---BGR--BGR--sy,(sx+1)*3+1
        // ----------------------------------
        // sy+1,sx*3-BGR---BGR---BGR--sy+1,(sx+1)*3+1

        const uint8_t *aPtr = src + sy * src_width * 3;     //start of upper row
        const uint8_t *bPtr = src + (sy + 1) * src_width * 3; //start of lower row
        //compute source data position and weight for [x0] components
            float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
            int   sx = floor(fx);
            fx -= sx;
            fx *= ((sx >= 0) && (sx < src_width - 1));
            sx = max(0, min(sx, src_width - 2));
        
        uint32_t sp = sx * 3;
        uint32_t sp_right = (sx + 1) * 3;
        uint32_t dp = dst_y * out_width * 3 + dst_x * 3;
        for (int i = 0; i < 3; i++) {
            dst[dp + i]
                = uint8_t((1.0f - fx) * (aPtr[sp + i] * (1.0f - fy) + bPtr[sp + i] * fy)
                            + fx * (aPtr[sp_right + i] * (1.0f - fy) + bPtr[sp_right + i] * fy));
        }

    }
}

__global__ void normalize_kernel(
                uint8_t *image, float* out_image,
                uint32_t in_width, uint32_t in_height,
                float c1, float c2, float c3) 
{
    const int src_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (src_x < in_width && src_y < in_height) {
        float* out_ptr = out_image + src_y * in_width * 3 + src_x * 3;
        uint8_t* in_ptr = image + src_y * in_width * 3 + src_x * 3;
        out_ptr[0] = float(in_ptr[0]) * c1;
        out_ptr[1] = float(in_ptr[1]) * c2;
        out_ptr[2] = float(in_ptr[2]) * c3;
    }
}

__global__ void reformat_kernel(float *image, 
                float* R,
                float* G,
                float* B,
                uint32_t in_width, 
                uint32_t in_height) 
{
    const int sx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx < in_width && sy < in_height) {
        float* in_ptr = image + sy * in_width * 3 + sx * 3;
        float* b_ptr = B + sy * in_width + sx;
        float* g_ptr = G + sy * in_width + sx;
        float* r_ptr = R + sy * in_width + sx;
        *b_ptr = in_ptr[0];
        *g_ptr = in_ptr[1];
        *r_ptr = in_ptr[2];
    }
}


__global__ void fusion_kernel(
    uint8_t *image, float* R, float* G, float* B,
    uint32_t in_width, uint32_t in_height,
    float c1, float c2, float c3, int top, int left,
    uint32_t out_width, uint32_t out_height, uint8_t border_val
) {
    const int sx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx < out_width && sy < out_height) {
        float* r_ptr = B + sy * out_width + sx;
        float* g_ptr = G + sy * out_width + sx;
        float* b_ptr = R + sy * out_width + sx;
        if (sx < left || sy < top || sx >= in_width + left || sy >= in_height + top) {
            // 边界处理
            *b_ptr = float(border_val) * c1;
            *g_ptr = float(border_val) * c2;
            *r_ptr = float(border_val) * c3;
        } else {
            uint8_t* in_ptr = image + (sy - top) * in_width * 3 + (sx - left) * 3;
            *b_ptr = float(in_ptr[0]) * c1;
            *g_ptr = float(in_ptr[1]) * c2;
            *r_ptr = float(in_ptr[2]) * c3;
        }
    }
}


inline __device__ void resize_pixel(
    uint8_t *image, int x, int y,
    float* r_ptr, float* g_ptr, float* b_ptr,
    float scale_x, float scale_y, 
    uint32_t in_width, uint32_t in_height,
    float c1, float c2, float c3
) {

    //y coordinate
        // 原图的y坐标, +0.5到像素坐标中心，否则是像素的左上角
        float fy = (float)((y + 0.5f) * scale_y - 0.5f);
        // 左上角的y, 向下取整
        int   sy = floor(fy);
        fy -= sy;
        //  防止越界
        sy = max(0, min(sy, in_height - 2));
        //row pointers
        // sy,sx*3--BGR---BGR--BGR--sy,(sx+1)*3+1
        // ----------------------------------
        // sy+1,sx*3-BGR---BGR---BGR--sy+1,(sx+1)*3+1

        const uint8_t *aPtr = image + sy * in_width * 3;     //start of upper row
        const uint8_t *bPtr = image + (sy + 1) * in_width * 3; //start of lower row
        //compute source data position and weight for [x0] components
            float fx = (float)((x + 0.5f) * scale_x - 0.5f);
            int   sx = floor(fx);
            fx -= sx;
            fx *= ((sx >= 0) && (sx < in_width - 1));
            sx = max(0, min(sx, in_width - 2));
        
        uint32_t sp = sx * 3;
        uint32_t sp_right = (sx + 1) * 3;
        // uint32_t dp = dst_y * out_width * 3 + dst_x * 3;
        // for (int i = 0; i < 3; i++) {
            *b_ptr = float(uint8_t((1.0f - fx) * (aPtr[sp + 0] * (1.0f - fy) + bPtr[sp + 0] * fy)
                            + fx * (aPtr[sp_right + 0] * (1.0f - fy) + bPtr[sp_right + 0] * fy))) * c1;

            *g_ptr = float(uint8_t((1.0f - fx) * (aPtr[sp + 1] * (1.0f - fy) + bPtr[sp + 1] * fy)
                            + fx * (aPtr[sp_right + 1] * (1.0f - fy) + bPtr[sp_right + 1] * fy))) * c2;

            *r_ptr = float(uint8_t((1.0f - fx) * (aPtr[sp + 2] * (1.0f - fy) + bPtr[sp + 2] * fy)
                            + fx * (aPtr[sp_right + 2] * (1.0f - fy) + bPtr[sp_right + 2] * fy))) * c3;
}


__global__ void fusion_all_kernel(
    uint8_t *image, float* R, float* G, float* B,
    uint32_t in_width, uint32_t in_height,
    float c1, float c2, float c3, int top, int left,
    uint32_t out_width, uint32_t out_height, uint8_t border_val,
    uint32_t resized_width, uint32_t resized_height,
    float scale_x, float scale_y
) {
    const int sx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx < out_width && sy < out_height) {
        float* b_ptr = B + sy * out_width + sx;
        float* g_ptr = G + sy * out_width + sx;
        float* r_ptr = R + sy * out_width + sx;
        if (sx < left || sy < top || sx >= resized_width + left || sy >= resized_height + top) {
            // 边界处理 ,大于缩放的边界直接赋值
            *b_ptr = float(border_val) * c1;
            *g_ptr = float(border_val) * c2;
            *r_ptr = float(border_val) * c3;
        } else {
            // 缩放中心处理
            resize_pixel(image, sx - left, sy - top, r_ptr, g_ptr, b_ptr, 
                        scale_x, scale_y, in_width, in_height, c1, c2, c3);
        }
    }
}

namespace cudapre {
// const nvcv::Tensor &inTensor, uint32_t batchSize, int inputLayerWidth, int inputLayerHeight,
                // cudaStream_t stream, const nvcv::Tensor &outTensor

void gpu_resize(uint8_t *image, 
                uint8_t* outImage,
                cudaStream_t stream,
                uint32_t src_width, 
                uint32_t src_height,
                uint32_t out_width,
                uint32_t out_height) 
{
    float scale_x = ((float)src_width) / out_width;
    float scale_y = ((float)src_height) / out_height;

    const int batch_size = 1;
    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);

    resize_op<<<gridSize, blockSize, 0, stream>>>(image, outImage, 
                                                scale_x, scale_y, src_width, 
                                                src_height, out_width, out_height);

    CHECK_RUN();
}


void gpu_copymakeborder(uint8_t *image, 
                uint8_t* outImage,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height,
                uint32_t out_width,
                uint32_t out_height,
                uint8_t border_val)
{
    int top = std::round(float(out_height - in_height) / 2 - 0.1f);
    int left = std::round(float(out_width - in_width) / 2 - 0.1f);

    const int batch_size = 1;
    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);

    copymakeborder_kernel<<<gridSize, blockSize, 0, stream>>>
                (image, outImage, in_width, in_height, out_width, out_height, top, left, border_val);
    CHECK_RUN();

}

void gpu_normalize(uint8_t *image, 
                float* out_image,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height,
                float c1, float c2, float c3)
{
    const int batch_size = 1;
    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(in_width, blockSize.x), divUp(in_height, blockSize.y), batch_size);

    normalize_kernel<<<gridSize, blockSize, 0, stream>>>
                (image, out_image, in_width, in_height, c1, c2, c3);
    CHECK_RUN();

}

void gpu_hwc2chw(float* image, 
                float* out_image,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height) 
{
    const int batch_size = 1;
    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(in_width, blockSize.x), divUp(in_height, blockSize.y), batch_size);
    float* B = out_image;
    float* G = out_image + in_width * in_height;
    float* R = out_image + in_width * in_height * 2;

    reformat_kernel<<<gridSize, blockSize, 0, stream>>>
                (image, R, G, B, in_width, in_height);
    CHECK_RUN();
}

void gpu_fusion(uint8_t *image,
                float* out_image,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height,
                uint32_t out_width,
                uint32_t out_height,
                float c1, float c2, float c3,
                uint8_t border_val) 
{
    int top = std::round(float(out_height - in_height) / 2 - 0.1f);
    int left = std::round(float(out_width - in_width) / 2 - 0.1f);

    const int batch_size = 1;
    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);

    float* B = out_image;
    float* G = out_image + out_width * out_height;
    float* R = out_image + out_width * out_height * 2;

    fusion_kernel<<<gridSize, blockSize, 0, stream>>>(
        image, R, G, B, in_width, in_height, c1, c2, c3,
        top, left, out_width, out_height, border_val
    );
}


void gpu_fusion_all(uint8_t *image,
                float* out_image,
                cudaStream_t stream,
                uint32_t in_width, 
                uint32_t in_height,
                uint32_t resized_width,
                uint32_t resized_height,
                uint32_t out_width,
                uint32_t out_height,
                float c1, float c2, float c3,
                uint8_t border_val) 
{
    float scale_x = ((float)in_width) / resized_width;
    float scale_y = ((float)in_height) / resized_height;

    const int batch_size = 1;
    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);


    int top = std::round(float(out_height - resized_height) / 2 - 0.1f);
    int left = std::round(float(out_width - resized_width) / 2 - 0.1f);

    float* B = out_image;
    float* G = out_image + out_width * out_height;
    float* R = out_image + out_width * out_height * 2;

    fusion_all_kernel<<<gridSize, blockSize, 0, stream>>>(
        image, R, G, B, in_width, in_height, c1, c2, c3,
        top, left, out_width, out_height, border_val,
        resized_width, resized_height, scale_x, scale_y
    );
}


void cpu_resize(uint8_t* src, 
                uint8_t* dst,
                uint32_t src_width, 
                uint32_t src_height,
                uint32_t out_width,
                uint32_t out_height) {
    
    float scale_x = ((float)src_width) / out_width;
    float scale_y = ((float)src_height) / out_height;
    // printf("scale_x: %f, scale_y:%f\n", scale_x, scale_y);

    for (int dst_y = 0; dst_y < out_height; dst_y++) {
        for (int dst_x = 0; dst_x < out_width; dst_x++) {

            //float space for weighted addition
            // using work_type = cuda::ConvertBaseTypeTo<float, uint8_t>;

            //y coordinate
            // 原图的y坐标, +0.5到像素坐标中心，否则是像素的左上角
            double fy = double((dst_y + 0.5f) * scale_y - 0.5f);
            // 左上角的y, 向下取整
            int   top_y = std::round(fy);
            fy -= top_y;
            //  防止越界
            top_y = max(0, min(top_y, src_height - 2));

            //row pointers
            // top_y,left_y---BGR--BGR--BGR--top_y,left_x+1
            // ----------------------------------
            // top_y+1,left_y---BGR--BGR--BGR--top_y+1,left_x+1

            // BGRBGR

            const uint8_t *aPtr = src + top_y * src_width * 3;     //start of upper row
            const uint8_t *bPtr = src + (top_y + 1) * src_width * 3; //start of lower row

            //compute source data position and weight for [x0] components
                double fx = double((dst_x + 0.5f) * scale_x - 0.5f);
                int   left_x = std::round(fx);
                fx -= left_x;
                // fx *= ((left_x >= 0) && (left_x < src_width - 1));
                left_x = max(0, min(left_x, src_width - 2));
            
            uint32_t sp_left = left_x * 3;
            uint32_t sp_right = (left_x + 1) * 3;
            uint32_t dp = dst_y * out_width * 3 + dst_x * 3;
            for (int i = 0; i < 3; i++) {
                dst[dp + i]
                    = uint8_t((1.0f - fx) * (aPtr[sp_left + i] * (1.0f - fy) + bPtr[sp_left + i] * fy)
                             + fx * (aPtr[sp_right + i] * (1.0f - fy) + bPtr[sp_right + i] * fy));
            }
        }
    }
}



};

