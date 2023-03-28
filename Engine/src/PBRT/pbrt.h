#pragma once

#ifndef MachineEpsilon
#define MachineEpsilon 0.5f * std::numeric_limits<float>::epsilon()
#endif // !MachineEpsilon

namespace CudaPBRT
{
	inline float gamma(int n)
	{
		return n * MachineEpsilon / (1.f - n * MachineEpsilon);
	}
}
