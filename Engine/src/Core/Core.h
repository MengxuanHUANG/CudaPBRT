#pragma once

#include <assert.h>
#include <memory>

#ifndef GLEW_STATIC 
#define GLEW_STATIC
#endif // !GLEW_STATIC 

#ifndef ASSERT
	#ifdef CUDA_PBRT_DEBUG
		#define ASSERT(x) assert((x))
	#else
		#define ASSERT(x)
	#endif
#endif

#ifndef SMART_POINTERS
#define SMART_POINTERS
#define uPtr std::unique_ptr
#define sPtr std::shared_ptr
#define wPtr std::weak_ptr

#define mkU std::make_unique
#define mkS std::make_shared

#endif

#ifndef BIT
#define BIT(x) 1 << (x)
#endif // !BIT(x)


