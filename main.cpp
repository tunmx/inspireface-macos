#include <iostream>
#include <inspireface.h>

int main()
{
    HResult ret;
    ret = HFLaunchInspireFace("models/Tracking-CoreML6-E0.isf");
    if (ret != HSUCCEED)
    {
        std::cout << "Load Resource error: " << ret << std::endl;
        return ret;
    }
    HFInspireFaceConfiguringIOSExtensionCoreMLPackagePath("models/Tracking-CoreML6-E0.mlmodelc");

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

    // Enable the functions in the pipeline: mask detection, live detection, and face quality detection
    HOption option = HF_ENABLE_NONE;
    // Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without tracking
    HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
    // Maximum number of faces detected
    HInt32 maxDetectNum = 20;
    // Face detection image input level
    HInt32 detectPixelLevel = 256;
    // Handle of the current face SDK algorithm context
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED)
    {
        std::cout << "Create FaceContext error: " << ret << std::endl;
        return ret;
    }

    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    HFSessionSetFilterMinimumFacePixelSize(session, 0);

    // Prepare an image parameter structure for configuration
    HFImageData imageParam = {0};
    imageParam.data = (uint8_t *)data.data;     // Data buffer
    imageParam.width = data.width;              // Target view width
    imageParam.height = data.height;            // Target view width
    imageParam.rotation = HF_CAMERA_ROTATION_0; // Data source rotate
    imageParam.format = HF_STREAM_BGR;          // Data source format

    // Create an image data stream
    HFImageStream imageHandle = {0};
    ret = HFCreateImageStream(&imageParam, &imageHandle);
    if (ret != HSUCCEED)
    {
        std::cout << "Create ImageStream error: " << ret << std::endl;
        return ret;
    }

    // Execute HF_FaceContextRunFaceTrack captures face information in an image
    HFMultipleFaceData multipleFaceData = {0};
    int loop = 100;
    for (int i = 0; i < loop; ++i)
    {
        ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
        if (ret != HSUCCEED)
        {
            std::cout << "Execute HFExecuteFaceTrack error: " << ret << std::endl;
            return ret;
        }
    }
    // Print the number of faces detected
    auto faceNum = multipleFaceData.detectedNum;
    std::cout << "Num of face: " << faceNum << std::endl;

    return 0;
}