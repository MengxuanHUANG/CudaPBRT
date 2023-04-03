cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CudaPBRT LANGUAGES CXX CUDA)
 
include(CTest)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Enable C++20 for host code
set(CMAKE_CXX_STANDARD 20)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 12 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)

set(EXTERNAL "external")

set(GLFW_ROOT_DIR ${EXTERNAL})
set(GLFW_USE_STATIC_LIBS ON)
find_package(GLFW REQUIRED)

set(GLEW_ROOT_DIR ${EXTERNAL})
set(GLEW_USE_STATIC_LIBS ON)
find_package(GLEW REQUIRED)

find_package(OpenGL REQUIRED)

set(GLM_ROOT_DIR ${EXTERNAL})
find_package(GLM REQUIRED)

set(IMGUI_ROOT_DIR ${EXTERNAL})
find_package(IMGUI REQUIRED)

message(${IMGUI_INCLUDE_DIRS})

# ImGui Library
file(GLOB imgui_headers 
    "${IMGUI_INCLUDE_DIRS}/imgui/*.cpp" 
    "${IMGUI_INCLUDE_DIRS}/imgui/backends/imgui_impl_glfw.cpp"
    "${IMGUI_INCLUDE_DIRS}/imgui/backends/imgui_impl_opengl3.cpp"
)

file(GLOB imgui_sources 
    "${IMGUI_INCLUDE_DIRS}/imgui/*.h"
    "${IMGUI_INCLUDE_DIRS}/imgui/backends/imgui_impl_glfw.h"
    "${IMGUI_INCLUDE_DIRS}/imgui/backends/imgui_impl_opengl3.h"
)

list(SORT headers)
list(SORT sources)

add_library(ImGui STATIC ${imgui_headers} ${imgui_sources})
target_include_directories(ImGui PUBLIC
    ${GLFW_INCLUDE_DIR} 
    "${IMGUI_INCLUDE_DIRS}/imgui"
)



# CudaPBRT library Projects

# set ENGIN_SRC_ROOT_DIR
set(ENGIN_SRC_ROOT_DIR "Engine/src")

# set library
set(ENGINE_LIBRARIES ImGui ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
# set include direction
set(ENGINE_INCLUDE_DIR
    ${ENGIN_SRC_ROOT_DIR}
    ${GLM_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIR} 
    ${GLFW_INCLUDE_DIR} 
    ${IMGUI_INCLUDE_DIRS}
    "${IMGUI_INCLUDE_DIRS}/imgui/backends"
)

file(GLOB_RECURSE headers 
    "${ENGIN_SRC_ROOT_DIR}/*.h" 
    "${ENGIN_SRC_ROOT_DIR}/*.cuh"
)
file(GLOB_RECURSE sources 
    "${ENGIN_SRC_ROOT_DIR}/*.cpp" 
    "${ENGIN_SRC_ROOT_DIR}/*.cu"
)

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

CUDA_ADD_LIBRARY(${CMAKE_PROJECT_NAME} ${headers} ${sources})

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
    ${ENGINE_INCLUDE_DIR}
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${ENGINE_LIBRARIES}
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC 
    CUDA_PBRT_DEBUG = 1
)

# SandBox project
# set SANDBOX_SRC_ROOT_DIR
set(SANDBOX_SRC_ROOT_DIR "SandBox/src")
# set library
set(SANDBOX_LIBRARIES ImGui ${CMAKE_PROJECT_NAME} ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
# set library
set(SANDBOX_INCLUDE_DIR
    ${ENGIN_SRC_ROOT_DIR}
    ${SANDBOX_SRC_ROOT_DIR}
    ${GLM_INCLUDE_DIRS} 
    ${GLEW_INCLUDE_DIR} 
    ${GLFW_INCLUDE_DIR} 
    ${IMGUI_INCLUDE_DIRS}
    "${IMGUI_INCLUDE_DIRS}/imgui/backends"
)

file(GLOB_RECURSE sandboxHeaders 
    "${SANDBOX_SRC_ROOT_DIR}/*.h" 
    "${SANDBOX_SRC_ROOT_DIR}/*.cuh"
)
file(GLOB_RECURSE sandboxSources 
    "${SANDBOX_SRC_ROOT_DIR}/*.cpp" 
    "${SANDBOX_SRC_ROOT_DIR}/*.cu"
)

list(SORT sandboxHeaders)
list(SORT sandboxSources)

source_group(Headers FILES ${sandboxHeaders})
source_group(Sources FILES ${sandboxSources})

CUDA_ADD_EXECUTABLE(SandBox ${sandboxHeaders} ${sandboxSources})

target_include_directories(SandBox PUBLIC
    ${SANDBOX_INCLUDE_DIR}
)

target_link_libraries(SandBox
    ${SANDBOX_LIBRARIES}
)
