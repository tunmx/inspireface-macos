#include <iostream>
#include <inspireface.h>
#include <chrono>
int main()
{
    // 全局配置InspireFace资源文件, 只需要执行一次, 建议放在APP启动时执行
    HResult ret;
    ret = HFLaunchInspireFace("models/Tracking-CoreML6-E0.isf");
    if (ret != HSUCCEED)
    {
        std::cout << "Load Resource error: " << ret << std::endl;
        return ret;
    }
    // 配置InspireFace的CoreML扩展文件, 只需要执行一次, 建议放在APP启动时执行
    HFInspireFaceConfiguringIOSExtensionCoreMLPackagePath("models/Tracking-CoreML6-E0.mlmodelc");

    // 读取一个图像, 仅在这个示例下使用, 在你的APP中, 你需要从你的APP的图像或视频去获取数据
    HFImageBitmap image;
    ret = HFCreateImageBitmapFromFilePath("images/kun.jpg", 3, &image);
    if (ret != HSUCCEED)
    {
        std::cout << "Create Image Bitmap error: " << ret << std::endl;
        return ret;
    }

    HFImageBitmapData data;
    ret = HFImageBitmapGetData(image, &data);
    if (ret != HSUCCEED)
    {
        std::cout << "Get Image Bitmap Data error: " << ret << std::endl;
        return ret;
    }

    HOption option = HF_ENABLE_NONE;
    // 如果你是使用视频流模式，建议打开人脸跟踪模式
    HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
    // 最大检测人脸数
    HInt32 maxDetectNum = 1;
    // 检测人脸等级, 通常是160、192、256、320、640，越大检测越精确，但耗时越长，如果是前置摄像头，建议使用160-256之间即可
    HInt32 detectPixelLevel = 256;
    // 创建InspireFace会话
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED)
    {
        std::cout << "Create FaceContext error: " << ret << std::endl;
        return ret;
    }

    // 设置跟踪预览大小 通常跟检测器的像素等级一致即可
    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    // 设置最小人脸像素大小 通常设置为0即可 如果你需要过滤小脸可以适当调高
    HFSessionSetFilterMinimumFacePixelSize(session, 0);

    // 准备一个图像参数结构体用于配置
    HFImageData imageParam = {0};
    imageParam.data = (uint8_t *)data.data;     // 数据缓冲区
    imageParam.width = data.width;              // 目标视图宽度
    imageParam.height = data.height;            // 目标视图高度
    imageParam.rotation = HF_CAMERA_ROTATION_0; // 数据源旋转
    imageParam.format = HF_STREAM_BGR;          // 数据源格式

    // 创建一个图像数据流
    HFImageStream imageHandle = {0};
    ret = HFCreateImageStream(&imageParam, &imageHandle);
    if (ret != HSUCCEED)
    {
        std::cout << "Create ImageStream error: " << ret << std::endl;
        return ret;
    }

    // 执行HF_FaceContextRunFaceTrack 在图像中捕获人脸信息
    HFMultipleFaceData multipleFaceData = {0};
    // 运行100次, 模拟多帧跟踪
    int loop = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loop; ++i)
    {
        // 执行人脸跟踪
        ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
        if (ret != HSUCCEED)
        {
            std::cout << "Execute HFExecuteFaceTrack error: " << ret << std::endl;
            return ret;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "run " << loop << " times, average time: " << duration.count() / loop << " seconds" << std::endl;
    // 打印检测到的人脸数
    auto faceNum = multipleFaceData.detectedNum;
    std::cout << "Num of face: " << faceNum << std::endl;

    // 遍历检测到的人脸
    for (int index = 0; index < faceNum; ++index)
    {
        // 绘制人脸矩形框
        HFaceRect rect = multipleFaceData.rects[index];
        HFColor rectColor = {124, 125, 0};
        HFImageBitmapDrawRect(image, rect, rectColor, 2);
        // 获取人脸5个关键点
        HPoint2f landmarks[5];
        ret = HFGetFaceFiveKeyPointsFromFaceToken(multipleFaceData.tokens[index], landmarks, 5);
        if (ret != HSUCCEED)
        {
            std::cout << "Get face five key points error: " << ret << std::endl;
            return ret;
        }
        for (int i = 0; i < 5; ++i)
        {
            HFColor landmarkColor = {0, 0, 255};
            HPoint2i p = {(int)landmarks[i].x, (int)landmarks[i].y};
            HFImageBitmapDrawCircle(image, p, 2, landmarkColor, 2);
        }
    }
    // 将图像写入文件
    ret = HFImageBitmapWriteToFile(image, "result.jpg");
    if (ret != HSUCCEED)
    {
        printf("Write image bitmap to file error: %lu\n", ret);
    }
    std::cout << "Write image bitmap to " << "result.jpg" << std::endl;

    // 释放图像
    ret = HFReleaseImageBitmap(image);
    if (ret != HSUCCEED)
    {
        printf("Release image bitmap error: %lu\n", ret);
    }
    // 释放图像数据流
    ret = HFReleaseImageStream(imageHandle);
    if (ret != HSUCCEED)
    {
        printf("Release image stream error: %lu\n", ret);
    }
    // 释放InspireFace会话
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED)
    {
        printf("Release session error: %lu\n", ret);
        return ret;
    }

    return 0;
}