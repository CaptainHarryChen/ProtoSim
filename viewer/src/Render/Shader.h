#pragma once
#include <string>
#include <glm/glm.hpp>

class Shader
{
public:
    Shader(std::string type, bool enableGeometry = false);
    virtual ~Shader() = default;

    unsigned int shaderProgram;
    std::string shaderName;

    void use();
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setMat4(const std::string &name, glm::mat4 value) const;
    void setVec3(const std::string &name, glm::vec3 value) const;
    void setVec4(const std::string &name, glm::vec4 value) const;

private:
    void checkCompileErrors(unsigned int shader, std::string type);
};
