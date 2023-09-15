#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include "pre_opencv.h"
#include "pre_cuda.h"
#include "utils.h"

#define MAX_HEIGHT 1920
#define MAX_WIDTH 1080

inline void convertChwToHwc(cv::Mat& in, float* out, int height, int width) {
    std::vector<cv::Mat> channelsArray;
    cv::split(in, channelsArray);
    int chanelLength = height * width;
    for (int c = 0; c < 3; ++c) {
        memcpy(out, channelsArray[c].data, height * width * sizeof(float));
        out += height * width;
    }
}

void cv_letterbox_normalize(const cv::Mat& img, cv::Mat& resize_img, 
                        cv::Mat& res, std::vector<float>& chw,
                        int width, int height)
{
    int channel = 3;

    cv::Mat float_res(height, width, CV_32FC3);
    float* float_res_ptr = float_res.ptr<float>();

    double cpu_start = cpuSecond();
    {
        resize_op(img, resize_img, cv::Size(width, height));
        copymakeborder_op(resize_img, res);
        res.convertTo(float_res, CV_32FC3, 1.0 / 255.0);
        convertChwToHwc(float_res, chw.data(), height, width);
    }

    double cpu_end = cpuSecond();

    printf("opencv letterbox time: %.2fms \n", (cpu_end - cpu_start) * 1000);

    std::string opencv_save_path = "../imgs/opencv_image.jpg";
    cv::imwrite(opencv_save_path, res);
}


void gpu_letterbox_normalize(cv::Mat &resize_out, cv::Mat& res, 
                    std::vector<float>& gpu_out,
                    std::vector<std::string>& image_paths,
                    int out_h, int out_w) 
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 定义比较大的图片，用于处理动态输入
    const int channel = 3;
    const int nums = MAX_WIDTH * MAX_HEIGHT * channel;

    uint8_t* data_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&data_dev, nums));

    // 最终copymakeborder和normlize的最后输出
    const int out_nums = out_h * out_w * 3;

    uint8_t* resize_data_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&resize_data_dev, out_nums));

    uint8_t* res_data_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&res_data_dev, out_nums));

    float* gpu_out_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_out_dev, out_nums * sizeof(float)));

    float* gpu_chw_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_chw_dev, out_nums * sizeof(float)));

    for (const auto& img_path : image_paths) {
        cv::Mat img = cv::imread(img_path);
        cv::Size shape = img.size();
        float r = std::min((float)out_h / (float)shape.height,
                       (float)out_w / (float)shape.width);

        int resize_width = (int)std::round((float)shape.width * r);
        int resize_height = (int)std::round((float)shape.height * r);
        resize_out = cv::Mat(resize_height, resize_width, CV_8UC3);

        printf("gpu resize h:%d, gpu resize w: %d\n", resize_height, resize_width);

        double cpu_start = cpuSecond();
        {
            // resize
            CHECK_CUDA_ERROR(cudaMemcpyAsync(data_dev, img.data, nums, cudaMemcpyHostToDevice, stream));
            cudapre::gpu_resize(data_dev, resize_data_dev, stream, 
                                img.cols, img.rows, resize_width, resize_height);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(resize_out.data, resize_data_dev, 
                                            resize_width * resize_height * 3, cudaMemcpyDeviceToHost, stream));

            // copymakeborder
            CHECK_CUDA_ERROR(cudaMemcpyAsync(res_data_dev, res.data, out_nums, cudaMemcpyHostToDevice, stream));
            cudapre::gpu_copymakeborder(resize_data_dev, res_data_dev, stream, 
                                        resize_width, resize_height, out_w, out_h);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(res.data, res_data_dev, out_nums, cudaMemcpyDeviceToHost, stream));

            // normalize
            cudapre::gpu_normalize(res_data_dev, gpu_out_dev, stream, 
                                    out_w, out_h, 1.0/255.0, 1.0/255.0, 1.0/255.0);

            // hwc2chw
            cudapre::gpu_hwc2chw(gpu_out_dev, gpu_chw_dev, stream, out_w, out_h);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(gpu_out.data(), gpu_chw_dev, out_nums * sizeof(float), 
                                            cudaMemcpyDeviceToHost, stream));

            cudaStreamSynchronize(stream);
        }

        double cpu_end = cpuSecond();
        printf("gpu resize time: %.2fms \n", (cpu_end - cpu_start) * 1000);
        break;
    }

    CHECK_CUDA_ERROR(cudaFree(resize_data_dev));
    CHECK_CUDA_ERROR(cudaFree(res_data_dev));
    CHECK_CUDA_ERROR(cudaFree(gpu_out_dev));
    cudaStreamDestroy(stream);

    std::string gpu_save_path = "../imgs/gpu_image.jpg";
    cv::imwrite(gpu_save_path, res);
}


int main() {
    int desiredDeviceId = 1;
    cudaSetDevice(desiredDeviceId);

    std::vector<std::string> image_paths;
    fs::path folderPath("../imgs/image2"); // 替换成您的文件夹路径
    // 获取所有图像文件的完整路径
    getAllImageFilesInFolder(folderPath, image_paths);


    const int out_w = 640;
    const int out_h = 640;
    cv::Scalar color(114, 114, 114);

    cv::Mat opencv_resize_res;
    cv::Mat opencv_res(out_h, out_w, CV_8UC3, color);
    std::vector<float> cpu_out(out_h * out_w * 3);

        for (const auto& img_path : image_paths) {
            std::cout << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            cv_letterbox_normalize(img, opencv_resize_res, 
                                opencv_res, cpu_out, out_w, out_h);
            break;
        }

    cv::Mat gpu_res(out_h, out_w, CV_8UC3, color);
    std::vector<float> gpu_out(out_h * out_w * 3);
    cv::Mat gpu_resize_res;
        gpu_letterbox_normalize(gpu_resize_res, gpu_res, gpu_out, image_paths, out_h, out_w);

    // 比较结果
    int max_inter = cmp_mat(opencv_resize_res, gpu_resize_res, gpu_resize_res.cols, gpu_resize_res.rows);
    printf("resize max inter:%d\n", max_inter);
    max_inter = cmp_mat(opencv_res, gpu_res, gpu_res.cols, gpu_res.rows);
    printf("letterbox inter:%d\n", max_inter);
    float vector_inter = cmp_vector(cpu_out, gpu_out, out_w, out_h, 1e-5);
    printf("final inter:%f\n", vector_inter);

    return 0;
}