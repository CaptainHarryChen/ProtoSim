#pragma once
#include <vector>
#include <memory>
#include <string>
#include <Render/Renderer.h>

class TextureRenderer : public Renderer
{
    static const int MAX_LIGHTS = 4; // limited by the shader

public:
    /// @brief Construct a new Mesh object with textures
    /// @param textures an array of 5 strings, representing the path to albedo, normal, metallic, roughness, ao textures
    TextureRenderer(std::vector<std::string> textures);

    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object) override;

protected:
    unsigned int aoID;
    unsigned int albedoID;
    unsigned int normalID;
    unsigned int metallicID;
    unsigned int roughnessID;

private:
    unsigned int loadTexture(const char *path);
};
