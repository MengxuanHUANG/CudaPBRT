#include <iostream>

#include "TestCuda.cuh"

int main()
{
    std::cout << "Hello CUDA PBRT!" << std::endl;

    glm::mat4 m1{
        {1, 0, 0, 0},
        {0, 1, 1, 0},
        {1, 0, 2, 0},
        {0, 3, 0, 1}
    };

    glm::mat4 m2{
        {1, 1, 0, 0},
        {0, 1, 1, 0},
        {1, 0, 1, 0},
        {0, 0, 0, 1}
    };

    glm::mat4 result;

    executeCuda(&m1, &m2, &result);

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::cout << result[i][j];
        }
        std::cout << std::endl;
    }

    return 0;
}