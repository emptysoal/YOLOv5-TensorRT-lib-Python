#include <assert.h>
#include "yololayer.h"

namespace Tn
{
    template<typename T> 
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> 
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float) * CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }

    YoloLayerPlugin::~YoloLayerPlugin()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CHECK(cudaFree(mAnchor[ii]));
        }
        CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void *buffer) const noexcept
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const noexcept
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int32_t YoloLayerPlugin::initialize() noexcept
    {
        return 0;
    }

    DimsExprs YoloLayerPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
    {
        //output the result to channel
        int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);
        // return Dims32{4, {1, totalsize + 1, 1, 1}};
        return DimsExprs{4, {exprBuilder.constant(1), exprBuilder.constant(totalsize + 1), exprBuilder.constant(1), exprBuilder.constant(1)}};
    }

    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const noexcept
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
    {
        return DataType::kFLOAT;
    }

    void YoloLayerPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
    {
    }

    void YoloLayerPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
    {
    }

    void YoloLayerPlugin::detachFromContext() noexcept
    {
    }

    const char* YoloLayerPlugin::getPluginType() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const noexcept
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() noexcept
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2DynamicExt* YoloLayerPlugin::clone() const noexcept
    {
        YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data)
    {
        return 1.0f / (1.0f + expf(-data));
    }

    __global__ void CalDetection(const float* input, float* output, int noElements,
        const int netWidth, const int netHeight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes, int outputElem)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid * bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < CHECK_COUNT; k++)
        {
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; i++)
            {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob)
                {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float* res_count = output + bnIdx * outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= maxoutobject) return;
            char* data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
            Detection* det = (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            // pytorch:
            //  y = x[i].sigmoid()
            //  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            //  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            //  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth
            det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netWidth / yoloWidth;
            det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netHeight / yoloHeight;

            // W: (Pw * e^tw) / FeaturemapW * netwidth
            // v5: https://github.com/ultralytics/yolov5/issues/471
            det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
            det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
            det->conf = box_prob * max_cls_prob;
            det->class_id = class_id;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize)
    {
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
        }
        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width * yolo.height * batchSize;
            if (numElem < mThreadCount) mThreadCount = numElem;

            //printf("Net: %d  %d \n", mYoloV5NetWidth, mYoloV5NetHeight);
            CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
                (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem);
        }
    }

    int32_t YoloLayerPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const *inputs, void* const *outputs, void* workspace, cudaStream_t stream) noexcept
    {
        forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, 1);  // define batchsize = 1
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const noexcept
    {
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames() noexcept
    {
        return &mFC;
    }

    IPluginV2DynamicExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
    {
        assert(fc->nbFields == 2);
        assert(strcmp(fc->fields[0].name, "netinfo") == 0);
        assert(strcmp(fc->fields[1].name, "kernels") == 0);
        int *p_netinfo = (int*)(fc->fields[0].data);
        int class_count = p_netinfo[0];
        int input_w = p_netinfo[1];
        int input_h = p_netinfo[2];
        int max_output_object_count = p_netinfo[3];
        std::vector<Yolo::YoloKernel> kernels(fc->fields[1].length);
        memcpy(&kernels[0], fc->fields[1].data, kernels.size() * sizeof(Yolo::YoloKernel));
        YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2DynamicExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}  // namespace nvinfer1
