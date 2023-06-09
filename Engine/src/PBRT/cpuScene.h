#pragma once

#include "gpuScene.h"
#include "Sampler/alias.h"
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
        int start_id = -1;
        int end_id = -1;

        // translation, rotation, and scale
        Transform transform;

        // material_id
        int material_id = 0;
    };

    struct TempTriangleLight
    {
        int obj_id = -1;
        float Lv = 0.f;
        bool double_side = false;
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

            m_AliasSamplers_1D.clear();
        }

        bool LoadCameraFromJSON(const JSON& json_data);
        bool LoadMaterialFromJSON(const JSON& json_data);
        unsigned int LoadShapeFromJSON(const JSON& json_data);
        bool LoadLightFromJSON(const JSON& json_data, 
                                std::vector<TempTriangleLight>& temp_triangles_lights);

        bool LoadSceneFromJsonFile(const char* path);

        bool LoadMeshFromFile(const char* path, ObjectData& obj_data);

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

        // alias sampler
        std::vector<uPtr<Cuda_AlaisSampler_1D>> m_AliasSamplers_1D;

    protected:
        // shapes' data
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec2> uvs;
        std::vector<TriangleData> triangles;
    };
}