#include "Core/Application.h"
#include "Window/Window.h"

#include "SandBox.h"

using namespace CudaPBRT;

int main()
{
	SandBox app({1960, 1300, "SandBox"});
	app.Run();

    return 0;
}