#pragma once

#include <assert.h>
#include <memory>

#define GLEW_STATIC

#ifdef CUDA_PBRT_DEBUG
#define ASSERT(x) assert((x))
#else
#define ASSERT(x)
#endif

#define SMART_POINTERS
#define uPtr std::unique_ptr
#define sPtr std::shared_ptr
#define wPtr std::weak_ptr

#define mkU std::make_unique
#define mkS std::make_shared

#define BIT(x) 1 << (x)


