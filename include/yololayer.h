#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>
#include "public.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 640;  // yolov5's input height and width must be divisible by 32.
    static constexpr int INPUT_W = 640;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection
    {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2DynamicExt
    {
    public:
        YoloLayerPlugin() = delete;
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int32_t getNbOutputs() const noexcept override
        {
            return 1;
        }

        DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;

        int32_t initialize() noexcept override;

        void terminate() noexcept override {};

        size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override
        {
            return 0;
        }

        int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const *inputs, void* const *outputs, void* workspace, cudaStream_t stream) noexcept override;

        size_t getSerializationSize() const noexcept override;

        void serialize(void *buffer) const noexcept override;

        bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
        {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        void destroy() noexcept override;

        IPluginV2DynamicExt* clone() const noexcept override;

        void setPluginNamespace(const char* pluginNamespace) noexcept override;

        const char* getPluginNamespace() const noexcept override;

        DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;

        void attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;

        void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;

        void detachFromContext() noexcept override;

    private:
        void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        const PluginFieldCollection* getFieldNames() noexcept override;

        IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

        IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(const char* pluginNamespace) noexcept override
        {
            mNamespace = pluginNamespace;
        }

        const char* getPluginNamespace() const noexcept override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };

    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

}  // namespace nvinfer1

#endif  // _YOLO_LAYER_H
