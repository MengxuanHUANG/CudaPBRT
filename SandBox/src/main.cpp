#include "Core/Application.h"

#include "SandBox.h"

using namespace CudaPBRT;

int main()
{
	SandBox app({1000, 1000, "SandBox"});
	app.Run();

    return 0;
}