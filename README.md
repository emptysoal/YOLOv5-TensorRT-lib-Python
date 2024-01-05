# 打包YOLO+TensorRT+Cuda为SO库，并通过Python调用

## 项目简介

### 项目目标

- 把 `TensorRT` `C++` `api`推理 `YOLOv5`的代码，打包成动态链接库，并通过 `Python` 调用。

**这样做的原因**：

1. 使用 `TensorRT` 的 `C++` `api`优化 `YOLO ` 模型，可以显著提升目标检测的速度，但 `C++` 代码不方便扩展为网络通信协议接口；

2. `Python` 作为最流行的胶水语言，拥有很多成熟的通信协议库，可以方便的进行各种网络协议通信，赋能于各种各样的服务；

因此，若将`C++`的模型推理代码编译为动态链接库，再使用`Python`封装，那就既有推理速度的优势，又具备可扩展性的便利

​		可能有人问为什么不直接使用 `TensorRT Python Api`，看似可以达到相同的效果，作者认为本项目有以下优势，比直接使用`TensorRT Python Api` 更好：

1. 使用 `TensorRT Python Api` 时，前处理和后处理是使用 `Python` 完成的，不过也是通过numpy实现的，速度方面没有太大差距；
2. 作者使用 Cuda 编程加速预处理，提升预处理速度，之后一同打包到动态链接库，推理速度更快。

### 项目概述

- 使用 `TensorRT-v8.2` 的 `C++` `api`，加速`YOLOv5-v5.0` 目标检测；
- 在 `Linux x86_64` 上进行部署；
- 在 `Jetson`系列嵌入式设备上也是可行的，把本项目中的 `CMakeLists.txt` 文件中头文件、库文件相关目录更换即可；

**大致实现过程如下：**

1. 作者把 `YOLOv5` 的 **`TensorRT` 推理封装成 `C++` 的类**，关键代码如下：

```c++
class YoloDetecter
{
public:
    YoloDetecter(const std::string trtFile, const int gpuId);
    ~YoloDetecter();
    float* inference(cv::Mat& img);
};
```

2. 接着作者使用 **`C` 类型的函数再次封装**上面的类，关键代码如下：

```c++
#ifdef __cplusplus
extern "C" {
#endif

YoloDetecter* YoloDetecter_new(char* trtFile, int gpuId){
    return new YoloDetecter(std::string(trtFile), gpuId);
}

float* inference_one(YoloDetecter* instance, const uchar* srcImgData, const int srcH, const int srcW){
    cv::Mat srcImg(srcH, srcW, CV_8UC3);
    memcpy(srcImg.data, srcImgData, srcH * srcW * 3 * sizeof(uchar));
    return instance->inference(srcImg);
}

void destroy(YoloDetecter* instance) { delete instance; }

#ifdef __cplusplus
}
#endif
```

3. 把封装后的代码**生成动态链接库**，`CMakeLists.txt` 中的关键部分如下

```cmake
# ====== yolo infer shared lib ======
cuda_add_library(yolo_infer SHARED 
    ${PROJECT_SOURCE_DIR}/src/preprocess.cu 
    ${PROJECT_SOURCE_DIR}/src/yololayer.cu 
    ${PROJECT_SOURCE_DIR}/src/yolo_infer.cpp
    ${PROJECT_SOURCE_DIR}/main.cpp
)
target_link_libraries(yolo_infer nvinfer cudart ${OpenCV_LIBS})
```

4. 再接着作者使用 **`Python` 封装一个检测类**，类当中调用的是上述 `C/C++` 代码，关键部分代码如下：

```python
class YoloDetector:
    def __init__(self, trt_file, gpu_id=0):
        self.yolo_infer_lib = ctypes.cdll.LoadLibrary("./lib/libyolo_infer.so")
        self.cpp_yolo_detector = self.yolo_infer_lib.YoloDetecter_new(trt_file.encode('utf-8'), gpu_id)

    def release(self):
        self.yolo_infer_lib.destroy(self.cpp_yolo_detector)

    def infer(self, image):
        out_data = self.yolo_infer_lib.inference_one(self.cpp_yolo_detector, image, height, width)
        out_data = as_array(out_data).copy().reshape(-1)
```

5. 最后，使用者不必关心具体的实现，仅仅使用下面的 2 行代码，即可实现 `Python` 对 `YOLOv5+TensorRT` `C++` 代码的调用

```python
# 实例化目标检测类
yolo_infer = YoloDetector(trt_file=plan_path, gpu_id=0)
# 使用目标检测实例推理
detect_res = yolo_infer.infer(img)
```

## 项目效果

![result_01](samples/_10008.jpg)

![result_02](samples/_10002.jpeg)

## 环境要求

- 作者自己所使用的基本环境如下：

| Ubuntu | CUDA | cuDNN | TensorRT | OpenCV |
| ------ | ---- | ----- | -------- | ------ |
| 20.04  | 11.6 | 8.4   | 8.2.4    | 4.5.0  |

想要方便点的话，可以直接拉取一个 `docker` 镜像：

```bash
docker pull nvcr.io/nvidia/tensorrt:22.04-py3
```

然后在镜像中编译安装 OpenCV，具体可参考下面链接中的环境构建部分：

https://github.com/emptysoal/TensorRT-v8-YOLOv5-v5.0

- python 第三方库环境

```bash
pip install numpy==1.22.3
pip install opencv-python==3.4.16.59
```

## 模型转换

把 `YOLO`检测模型，转换成`TensorRT`的序列化文件，后缀 `.plan`（作者的习惯，也可以是`.engine`或其他）

### 原模型下载

- 链接：https://pan.baidu.com/s/1YG-A8dXL4zWvecsD6mW2ug 
- 提取码：y2oz

下载并解压后，模型文件说明：

```bash
模型文件目录
    └── YOLOv5-v5.0  # 该目录中存放的是 YOLOv5 目标检测网络的模型
        ├── yolov5s.pt  # 官方 PyTorch 格式的模型文件
        └── para.wts  # 根据 yolov5s.pt 导出的 wts 格式模型文件
```

也可以直接从官方`YOLOv5-v5.0`处下载 `yolov5s.pt`，然后直接进入到下面的模型转换

### YOLO模型转换

- 将上述 `yolov5s.pt` 转为 `model.plan`，或 `para.wts`转为 `model.plan`
- 具体转换方法参考下面链接，也是作者自己发布的一个项目

- https://github.com/emptysoal/TensorRT-v8-YOLOv5-v5.0/tree/main

完成之后便可得到 `model.plan` ，为检测网络的 `TensorRT` 序列化模型文件。

## 运行项目

- 开始编译并运行
- 按如下步骤

```bash
# 创建用于存储 TensoRT 模型的目录
mkdir resources
# 把上面转换得到的 plan 文件复制到目录 resources 中
cp {TensorRT-v8-YOLOv5-v5.0}/model.plan ./resources

mkdir images  # 向其中放入用于推理的图片文件

mkdir build
cd build
cmake ..
make
# 以上执行完成后，会生成 lib 目录，其中存放着 C++ 代码所生成的动态链接库

# 运行 python 推理代码，即可调用动态链接库完成快速推理
python main.py
# 检测结果会保存到output目录下
```

- 运行后可以看到如下日志：

```bash
Succeeded getting serialized engine!
Succeeded loading engine!
Model load cost: 1.2174 s
Infer 001.jpg cost: 0.0077 s
Infer 002.jpg cost: 0.0054 s
Infer 003.jpg cost: 0.0043 s
```

## 项目参考

在`Python`如何调用`C++`代码部分，主要参考了下面的链接：

- https://geek-docs.com/python/python-ask-answer/601_python_how_can_i_use_c_class_in_python.html
- https://blog.csdn.net/qq_41554005/article/details/128292116

## 其他项目

作者的其他一些项目，若感兴趣，欢迎交流

[基于TensorRT v8部署加速YOLOv5-v5.0](https://github.com/emptysoal/TensorRT-v8-YOLOv5-v5.0)

[Deepsort+YOLOv5的TensorRT加速部署](https://github.com/emptysoal/Deepsort-YOLOv5-TensorRT)

[CUDA编程加速图像预处理](https://github.com/emptysoal/cuda-image-preprocess)

[TensorRT各种API对模型加速效果的对比实验](https://github.com/emptysoal/tensorrt-experiment)

