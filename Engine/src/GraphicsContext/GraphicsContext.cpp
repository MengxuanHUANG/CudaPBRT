#include "GraphicsContext.h"
#include "GLEWContext.h"

namespace CudaPBRT
{
	uPtr<GraphicsContext> GraphicsContext::CreateGraphicsContext(void* window)
	{
		// TODO: create context based on global parameters

		return mkU<GLEWContext>(window);
	}
}