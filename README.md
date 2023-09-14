# Model-Infer-Optimize
记录之前使用过的naiive方法对模型推理前处理过程的优化，偏工程方向，也是为了更好的理解内部原理，万一哪天需要用的到的时候能用，


## TODO
目前的yolo系列前处理主要使用的是letterbox，我在网上找到了很多关于letterbox的版本的实现方式，通过自己的整理和编写，测试代码的实现，主要还是想着以后集成到TensorRT中，加速模型推理，因为经过之前的测试发现，有时候图片预处理不当会影响模型推理

- [x] cpu-opencv: 缩放+copymakeborder+normalize+通道转换
- [x] cuda: 缩放+copymakeborder+normalize+cpu+通道转换
- [ ] cuda-opnecv: 缩放+copymakeborder+normalize+cpu+通道转换
- [ ] cuda-仿射变换

其实也看过CV-CUDA的代码，但是我在想一个问题，CV-CUDA的letterbox的每个阶段都启动一个核函数，相当于多次启动核函数，是否会有kernel launch overhead问题，回头找时间测试一下

## environment

```
opencv
cuda==11.8
boost
```

### install
opencv：没有编译cuda版本的代码

cuda安装，网上教程挺多的，也可以直接使用tensorRT的镜像，后面可以通过docker设置的CPU，限制CPU运行数量

boost：主要是使用filesystem读取目录文件，安装参考`sudo apt-get install libboost-all-dev`，如果不需要可把代码中的关于boost中的代码删除


### reference

[opencv的双线性插值原理](https://zhuanlan.zhihu.com/p/513569382)

[use-tensorrt-c-api-with-opencv](https://www.dotndash.net/2023/03/09/using-tensorrt-with-opencv-cuda.html#use-tensorrt-c-api-with-opencv)

[CV-CUDA](https://github.com/CVCUDA/CV-CUDA)

[Kernel Launch](https://zhuanlan.zhihu.com/p/544492099?utm_id=0)

[CV-CUDA-yolov前处理demo](https://zhuanlan.zhihu.com/p/637458406)
