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

    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos, const RenderObject *object) const override;

protected:
    unsigned int m_ao_id;
    unsigned int m_albedo_id;
    unsigned int m_normal_id;
    unsigned int m_metallic_id;
    unsigned int m_roughness_id;

private:
    unsigned int LoadTexture(const char *path) const;
};
