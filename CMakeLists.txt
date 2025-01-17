project(test)
cmake_minimum_required(VERSION 3.10)

enable_language(OBJC)
enable_language(OBJCXX)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message(STATUS "Target architecture: x86")
    set(ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm)|(ARM)|(aarch64)|(AARCH64)")
    message(STATUS "Target architecture: ARM") 
    set(ARCH "arm64")
else()
    message(WARNING "Unknown target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH "unknown")
endif()

include_directories(${CMAKE_SOURCE_DIR}/inspireface-lib/include)

# MacOS需要链接的库
find_library(FOUNDATION_LIBRARY Foundation)
find_library(COREML_LIBRARY CoreML)
find_library(ACCELERATE_LIBRARY Accelerate)

# 设置InspireFace库路径
set(INSPIREFACE_LIB ${CMAKE_SOURCE_DIR}/inspireface-lib/${ARCH}/libInspireFace.a)
# 设置MNN库路径
set(MNN_LIB ${CMAKE_SOURCE_DIR}/inspireface-lib/${ARCH}/libMNN.a)
# 设置需要链接的库
set(NEED_LIBRARIES ${FOUNDATION_LIBRARY} ${COREML_LIBRARY} ${ACCELERATE_LIBRARY} ${INSPIREFACE_LIB} ${MNN_LIB})

# 创建可执行文件
add_executable(test main.cpp)
target_link_libraries(test ${NEED_LIBRARIES})