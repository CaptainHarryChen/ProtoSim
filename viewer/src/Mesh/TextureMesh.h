#pragma once
#include <vector>
#include <memory>
#include <string>
#include <Mesh/Mesh.h>

class Shader;

class TextureMesh : public Mesh
{
    static const int MAX_LIGHTS = 4; // limited by the shader

public:
    /// @brief Construct a new Mesh object with textures
    /// @param vertices 
    /// @param indices 
    /// @param textures an array of 5 strings, representing the path to albedo, normal, metallic, roughness, ao textures
    TextureMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<std::string> textures);

    virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos) override;
	virtual void Draw(const CameraInfo &camera, const std::vector<LightInfo> &light_infos, const std::vector<ShadowMappingInfo> &shadow_mapping_infos) override;

protected:
    std::shared_ptr<Shader> shader;

    unsigned int aoID;
	unsigned int albedoID;
	unsigned int normalID;
	unsigned int metallicID;
	unsigned int roughnessID;

private:
    unsigned int loadTexture(const char *path);
};
