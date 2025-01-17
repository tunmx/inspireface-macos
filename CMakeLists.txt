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

find_library(FOUNDATION_LIBRARY Foundation)
find_library(COREML_LIBRARY CoreML)
find_library(ACCELERATE_LIBRARY Accelerate)


set(INSPIREFACE_LIB ${CMAKE_SOURCE_DIR}/inspireface-lib/${ARCH}/libInspireFace.a)
set(MNN_LIB ${CMAKE_SOURCE_DIR}/inspireface-lib/${ARCH}/libMNN.a)
set(NEED_LIBRARIES ${FOUNDATION_LIBRARY} ${COREML_LIBRARY} ${ACCELERATE_LIBRARY} ${INSPIREFACE_LIB} ${MNN_LIB})


add_executable(test main.cpp)
target_link_libraries(test ${NEED_LIBRARIES})