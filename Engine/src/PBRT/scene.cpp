#include "scene.h"

namespace CudaPBRT
{
	CPUScene::CPUScene(const char* path)
	{
		LoadSceneFromJSON(path);
	}

	void CPUScene::LoadSceneFromJSON(const char* path)
	{
		// clear previous scene data
		ClearScene();

		// try to open json file

	}

	void CPUScene::LoadObj(const char* path, int material_id, const Transform& transform)
	{
		// try to open obj file

		// read mesh data
	}
}