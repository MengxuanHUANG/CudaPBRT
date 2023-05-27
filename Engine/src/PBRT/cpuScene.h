#pragma once

#include "gpuScene.h"
#include "Camera/CameraController.h"

#include <json/json.hpp>

using JSON = nlohmann::json;

namespace CudaPBRT
{
    class PerspectiveCamera;
    class PerspectiveCameraController;

    struct ObjectData
    {
        // triangles start and end id
        int start_id;
        int end_id;

        // translation, rotation, and scale
        Transform transform;

        // material_id
        int material_id;
    };

    /*
    ** Scene data that will be maintenanced on CPU
    */

    class CPUScene
    {
    public:
        CPUScene();
        CPUScene(const char* path);
        ~CPUScene()
        {
            ClearScene();
        }

        inline void ClearScene()
        {
            m_GPUScene.FreeDataOnCuda();

            m_Textures.clear();
            shapeData.clear();
            materialData.clear();
            lightData.clear();
        }

        bool LoadCameraFromJSON(JSON& json_data);

        bool LoadSceneFromJSON(const char* path);

        bool LoadObj(const char* path, ObjectData& obj_data);

        void CreateBoundingBox(std::vector<ShapeData>& shapeData, std::vector<glm::vec3>& vertices);

    public:
        uPtr<PerspectiveCamera> camera;
        uPtr<PerspectiveCameraController> camController;

        GPUScene m_GPUScene;

        std::vector<ObjectData> objectData;

        // textures
        std::vector<uPtr<CudaTexture>> m_Textures;

        // shapes' data
        std::vector<ShapeData> shapeData;

        // materials' data
        std::unordered_map<std::string, int> materials_map;
        std::vector<MaterialData> materialData;

        // lights' data
        std::vector<LightData> lightData;

    protected:
        // shapes' data
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec2> uvs;
        std::vector<TriangleData> triangles;
    };
}