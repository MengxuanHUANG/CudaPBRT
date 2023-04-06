#include "Core/Application.h"

#include "SandBox.h"

using namespace CudaPBRT;

int main()
{
	SandBox app({700, 700, "SandBox"});
	app.Run();

    return 0;
}