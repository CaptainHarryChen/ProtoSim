#include <Mesh/SurfaceMesh.h>
#include <Render/Shader.h>
#include <stb_image.h>

SurfaceMesh::SurfaceMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<glm::vec3> material, bool cubic)
{
    // std::cout << isCubic << std::endl;
    isCubic = cubic;
    this->vertices = vertices;
    this->indices = indices;
    this->material = material;
    setupMesh();
    if (cubic)
        shader = std::make_shared<Shader>("pbr", true);
    else
        shader = std::make_shared<Shader>("pbr2D", true);
    shaderEdge = std::make_shared<Shader>("edge");
}

SurfaceMesh::SurfaceMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<std::string> textures, bool cubic)
{
    // std::cout << isCubic << std::endl;
    isCubic = cubic;
    assert(textures.size() == 5);
    this->vertices = vertices;
    this->indices = indices;
    this->textures = textures;
    setupMesh();
    if (cubic)
        shader = std::make_shared<Shader>("pbr_texture", true);
    else
        shader = std::make_shared<Shader>("pbr_texture2D", true);
    shaderEdge = std::make_shared<Shader>("edge");
}

SurfaceMesh::SurfaceMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices) // light
{
    this->vertices = vertices;
    this->indices = indices;
    setupMesh();
    shader = std::make_shared<Shader>("light_cube");
    shaderEdge = std::make_shared<Shader>("edge");
}

void SurfaceMesh::BindTexture()
{
    // order: albedo, normal, metallic, roughness, ao
    assert(textures.size() == 5);
    albedoID = loadTexture(textures[0].c_str());
    normalID = loadTexture(textures[1].c_str());
    metallicID = loadTexture(textures[2].c_str());
    roughnessID = loadTexture(textures[3].c_str());
    aoID = loadTexture(textures[4].c_str());
    textureBinded = true;
    // std::cout << ">>> SURFACE MESH\t texture binded" << std::endl;
}

void SurfaceMesh::ActivateTexture()
{
    assert(textureBinded == true);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, albedoID);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normalID);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, metallicID);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, roughnessID);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, aoID);
}
void SurfaceMesh::DrawEdge(
    glm::mat4 model,
    glm::mat4 view,
    glm::mat4 projection,
    glm::vec3 Color)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    shaderEdge->use();
    shaderEdge->setMat4("model", model);
    shaderEdge->setMat4("view", view);
    shaderEdge->setMat4("projection", projection);
    shaderEdge->setVec3("color", Color);
    Draw();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void SurfaceMesh::Draw()
{
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void SurfaceMesh::UpdateVertices(std::vector<Vertex> data)
{
    glBindVertexArray(VAO);
    vertices = data;
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(Vertex), &vertices[0]);
    glBindVertexArray(0);
}

// pbr, pbr texture
void SurfaceMesh::Draw(
    std::vector<unsigned int> depthMap3D,
    glm::mat4 model,
    glm::mat4 view,
    glm::mat4 projection,
    std::vector<glm::vec3> lightPos,
    std::vector<glm::vec3> lightColor,
    glm::vec3 viewPos,
    std::vector<bool> lightsOn,
    float far_plane,
    bool enableShadow)
{
    // std::cout << enableShadow << std::endl;
    shader->use();
    shader->setMat4("model", model);
    shader->setMat4("view", view);
    shader->setMat4("projection", projection);
    shader->setVec3("viewPos", viewPos);
    shader->setFloat("far_plane", far_plane);
    shader->setBool("enableShadow", enableShadow);
    for (int i = 0; i < lightPos.size(); ++i)
    {
        shader->setVec3("lightPos[" + std::to_string(i) + "]", lightPos[i]);
        shader->setVec3("lightColor[" + std::to_string(i) + "]", lightColor[i]);
        shader->setBool("lightsOn[" + std::to_string(i) + "]", lightsOn[i]);
    }
    if (textureBinded == true)
    {
        ActivateTexture();
        for (unsigned int i = 0; i < depthMap3D.size(); ++i)
        {
            glActiveTexture(GL_TEXTURE5 + i);
            glBindTexture(GL_TEXTURE_CUBE_MAP, depthMap3D[i]);
        }
        shader->setInt("albedoMap", 0);
        shader->setInt("normalMap", 1);
        shader->setInt("metallicMap", 2);
        shader->setInt("roughnessMap", 3);
        shader->setInt("aoMap", 4);
        for (unsigned int i = 0; i < depthMap3D.size(); ++i)
        {
            shader->setInt("depthMap3D[" + std::to_string(i) + "]", 5 + i);
        }
    }
    else
    {
        shader->setVec3("albedoIn", material[0]);
        shader->setFloat("metallicIn", material[1][0]);
        shader->setFloat("roughnessIn", material[1][1]);
        shader->setFloat("aoIn", material[1][2]);
        for (unsigned int i = 0; i < depthMap3D.size(); ++i)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_CUBE_MAP, depthMap3D[i]);
        }
        for (unsigned int i = 0; i < depthMap3D.size(); ++i)
        {
            shader->setInt("depthMap3D[" + std::to_string(i) + "]", i);
        }
    }
    Draw();
}

void SurfaceMesh::Draw(
    std::vector<unsigned int> depthMap2D,
    glm::mat4 model,
    glm::mat4 view,
    glm::mat4 projection,
    std::vector<glm::vec3> lightPos,
    std::vector<glm::vec3> lightColor,
    glm::vec3 viewPos,
    std::vector<bool> lightsOn,
    std::vector<glm::mat4> lightSpaceMatrix,
    bool enableShadow)
{
    shader->use();
    shader->setMat4("model", model);
    shader->setMat4("view", view);
    shader->setMat4("projection", projection);
    shader->setVec3("viewPos", viewPos);
    shader->setBool("enableShadow", enableShadow);
    for (int i = 0; i < lightPos.size(); ++i)
    {
        shader->setVec3("lightPos[" + std::to_string(i) + "]", lightPos[i]);
        shader->setVec3("lightColor[" + std::to_string(i) + "]", lightColor[i]);
        shader->setMat4("lightSpaceMatrix[" + std::to_string(i) + "]", lightSpaceMatrix[i]);
        shader->setBool("lightsOn[" + std::to_string(i) + "]", lightsOn[i]);
    }
    if (textureBinded == true)
    {
        // std::cout << "texture\n";
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, albedoID);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalID);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, metallicID);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, roughnessID);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, aoID);
        for (int i = 0; i < depthMap2D.size(); ++i)
        {
            glActiveTexture(GL_TEXTURE5 + i);
            glBindTexture(GL_TEXTURE_2D, depthMap2D[i]);
        }
        shader->setInt("albedoMap", 0);
        shader->setInt("normalMap", 1);
        shader->setInt("metallicMap", 2);
        shader->setInt("roughnessMap", 3);
        shader->setInt("aoMap", 4);
        for (int i = 0; i < depthMap2D.size(); ++i)
            shader->setInt("depthMap2D[" + std::to_string(i) + "]", 5 + i);
    }
    else
    {
        // std::cout << "no texture\n";
        for (int i = 0; i < depthMap2D.size(); ++i)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, depthMap2D[i]);
        }
        shader->setVec3("albedoIn", material[0]);
        shader->setFloat("metallicIn", material[1][0]);
        shader->setFloat("roughnessIn", material[1][1]);
        shader->setFloat("aoIn", material[1][2]);
        for (int i = 0; i < depthMap2D.size(); ++i)
            shader->setInt("depthMap2D[" + std::to_string(i) + "]", i);
    }
    Draw();
}

// light
void SurfaceMesh::Draw(glm::mat4 model, glm::mat4 view, glm::mat4 projection, glm::vec3 lightPos, glm::vec3 lightColor, float scale)
{
    shader->use();
    glm::mat4 __model = model;
    __model = glm::translate(__model, lightPos);
    __model = glm::scale(__model, glm::vec3(scale));
    shader->setMat4("model", __model);
    shader->setMat4("view", view);
    shader->setMat4("projection", projection);
    shader->setVec3("lightColor", lightColor);
    Draw();
}

void SurfaceMesh::setupMesh()
{
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, Normal));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, TexCoords));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

unsigned int SurfaceMesh::loadTexture(char const *path)
{
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_set_flip_vertically_on_load(true);

    int texture_width, texture_height, nrChannels;
    unsigned char *texture_data = stbi_load(path, &texture_width, &texture_height, &nrChannels, 0);
    GLenum format;
    if (nrChannels == 1)
        format = GL_RED;
    if (nrChannels == 3)
        format = GL_RGB;
    if (nrChannels == 4)
        format = GL_RGBA;

    glTexImage2D(GL_TEXTURE_2D, 0, format, texture_width, texture_height, 0, format, GL_UNSIGNED_BYTE, texture_data);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(texture_data);

    return texture;
}
