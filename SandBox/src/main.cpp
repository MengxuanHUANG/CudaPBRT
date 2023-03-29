#include "Core/Application.h"

#include "SandBox.h"

using namespace CudaPBRT;

int main()
{
	SandBox app({540, 540, "SandBox"});
	app.Run();

    return 0;
}