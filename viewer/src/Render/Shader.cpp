#include "Shader.h"
#include <iostream>
#include <fstream>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <viewer_config.h>

Shader::Shader(std::string type, bool enable_geometry)
{
    m_shader_name = type;
    std::string viewer_dir(VIEWER_DIR);

    m_shader_program = glCreateProgram();

    std::ifstream vstream(viewer_dir + "/shader/" + type + ".vs.glsl");
    std::string vstring = std::string(std::istreambuf_iterator<char>(vstream), std::istreambuf_iterator<char>());
    const char *vs = vstring.c_str();
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vs, nullptr);
    glCompileShader(vertexShader);
    checkCompileErrors(vertexShader, "VERTEX");
    glAttachShader(m_shader_program, vertexShader);

    unsigned int geometryShader;
    if (enable_geometry == true)
    {
        // std::cout << "Enable geometry shader" << std::endl;
        std::ifstream gstream(viewer_dir + "/shader/" + type + ".gs.glsl");
        std::string gstring = std::string(std::istreambuf_iterator<char>(gstream), std::istreambuf_iterator<char>());
        const char *gs = gstring.c_str();
        geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometryShader, 1, &gs, nullptr);
        glCompileShader(geometryShader);
        checkCompileErrors(geometryShader, "GEOMETRY");
        glAttachShader(m_shader_program, geometryShader);
    }

    std::ifstream fstream(viewer_dir + "/shader/" + type + ".fs.glsl");
    std::string fstring = std::string(std::istreambuf_iterator<char>(fstream), std::istreambuf_iterator<char>());
    const char *fs = fstring.c_str();
    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fs, nullptr);
    glCompileShader(fragmentShader);
    checkCompileErrors(fragmentShader, "FRAGMENT");
    glAttachShader(m_shader_program, fragmentShader);

    glLinkProgram(m_shader_program);
    checkCompileErrors(m_shader_program, "PROGRAM");

    glDeleteShader(vertexShader);
    if (enable_geometry == true)
        glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);
}

void Shader::use()
{
    glUseProgram(m_shader_program);
};

void Shader::setBool(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(m_shader_program, name.c_str()), (int)value);
}

void Shader::setInt(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(m_shader_program, name.c_str()), value);
}

void Shader::setFloat(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(m_shader_program, name.c_str()), value);
}

void Shader::setMat4(const std::string &name, glm::mat4 value) const
{
    glUniformMatrix4fv(glGetUniformLocation(m_shader_program, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::setVec3(const std::string &name, glm::vec3 value) const
{
    glUniform3f(glGetUniformLocation(m_shader_program, name.c_str()), value.x, value.y, value.z);
}

void Shader::setVec4(const std::string &name, glm::vec4 value) const
{
    glUniform4f(glGetUniformLocation(m_shader_program, name.c_str()), value.x, value.y, value.z, value.w);
}

void Shader::checkCompileErrors(unsigned int shader, std::string type)
{
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM")
    {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << m_shader_name << " " << type << "\n"
                      << infoLog << std::endl;
        }
    }
    else
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << m_shader_name << " " << type << "\n"
                      << infoLog << std::endl;
        }
    }
}
