#include "CubeLineBox.h"
#include <viewer/Framework/RenderSystem.h>
#include <viewer/RenderObject/LineSegment.h>
#include <viewer/Renderer/SolidColorRenderer.h>
#include <viewer/GLFWApp.h>

namespace viewer {

CubeLineBox::CubeLineBox(const std::vector<float>& bbox, float dist, glm::vec3 color)
{
    std::vector<unsigned int> grid_sizes;
    std::vector<float>        outter_box = bbox;
    for (unsigned int i = 0; i < 3; ++i)
    {
        grid_sizes.push_back(( unsigned int )floor((bbox[i + 3] - bbox[i]) / dist - 0.5) + 2);
    }
    //   6------7
    //  /|     /|
    // 4------5 |
    // | |    | |
    // | 2----|-3
    // |/     |/
    // 0------1
    float                            dist_half  = dist * 0.5f;
    unsigned int                     cube_count = 0;
    std::vector<glm::vec3>           lineVerts;
    std::vector<std::pair<int, int>> lineInd;

    for (unsigned int i = 0; i <= grid_sizes[0]; ++i)
    {
        for (unsigned int j = 0; j <= grid_sizes[1]; ++j)
        {
            for (unsigned int k = 0; k <= grid_sizes[2]; ++k)
            {

                float x = bbox[0] - dist_half + dist * i;
                float y = bbox[1] - dist_half + dist * j;
                float z = bbox[2] - dist_half + dist * k;

                if (i == 0 || i == grid_sizes[0] || j == 0 || j == grid_sizes[1] || k == 0)
                {
                    lineVerts.push_back(glm::vec3(x - dist_half, y - dist_half, z - dist_half));
                    lineVerts.push_back(glm::vec3(x + dist_half, y - dist_half, z - dist_half));
                    lineVerts.push_back(glm::vec3(x - dist_half, y - dist_half, z + dist_half));
                    lineVerts.push_back(glm::vec3(x + dist_half, y - dist_half, z + dist_half));
                    lineVerts.push_back(glm::vec3(x - dist_half, y + dist_half, z - dist_half));
                    lineVerts.push_back(glm::vec3(x + dist_half, y + dist_half, z - dist_half));
                    lineVerts.push_back(glm::vec3(x - dist_half, y + dist_half, z + dist_half));
                    lineVerts.push_back(glm::vec3(x + dist_half, y + dist_half, z + dist_half));

                    lineInd.push_back({ cube_count * 8 + 0, cube_count * 8 + 1 });
                    lineInd.push_back({ cube_count * 8 + 0, cube_count * 8 + 2 });
                    lineInd.push_back({ cube_count * 8 + 1, cube_count * 8 + 3 });
                    lineInd.push_back({ cube_count * 8 + 2, cube_count * 8 + 3 });
                    lineInd.push_back({ cube_count * 8 + 4, cube_count * 8 + 5 });
                    lineInd.push_back({ cube_count * 8 + 4, cube_count * 8 + 6 });
                    lineInd.push_back({ cube_count * 8 + 5, cube_count * 8 + 7 });
                    lineInd.push_back({ cube_count * 8 + 6, cube_count * 8 + 7 });
                    lineInd.push_back({ cube_count * 8 + 0, cube_count * 8 + 4 });
                    lineInd.push_back({ cube_count * 8 + 1, cube_count * 8 + 5 });
                    lineInd.push_back({ cube_count * 8 + 2, cube_count * 8 + 6 });
                    lineInd.push_back({ cube_count * 8 + 3, cube_count * 8 + 7 });
                    ++cube_count;
                }
            }
        }
    }
    m_line_segment = std::make_shared<LineSegment>(lineVerts, lineInd);
    m_line_segment->AddRenderer(std::make_shared<SolidColorRenderer>(color, true));
    GLFWApp::GetInstance()->GetRenderSystem()->AddRenderObject(m_line_segment);
}

}  // namespace viewer
