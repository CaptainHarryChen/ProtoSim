#include "Background.h"
#include <Render/Shader.h>

Background::Background()
{
    SetUp();
}

Background::Background(glm::vec3 t, glm::vec3 m, glm::vec3 b)
{
    Update(t, m, b);
    SetUp();
}

void Background::Update(glm::vec3 t, glm::vec3 m, glm::vec3 b)
{
    __SCREEN_T = t;
    __SCREEN_M = m;
    __SCREEN_B = b;

    float _[] =
        {
            -1.0f, -1.0f, __SCREEN_B.x, __SCREEN_B.y, __SCREEN_B.z,
            1.0f, -1.0f, __SCREEN_B.x, __SCREEN_B.y, __SCREEN_B.z,
            -1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,

            -1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
            1.0f, -1.0f, __SCREEN_B.x, __SCREEN_B.y, __SCREEN_B.z,
            1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,

            -1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
            1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
            -1.0f, 1.0f, __SCREEN_T.x, __SCREEN_T.y, __SCREEN_T.z,

            -1.0f, 1.0f, __SCREEN_T.x, __SCREEN_T.y, __SCREEN_T.z,
            1.0f, 0.0f, __SCREEN_M.x, __SCREEN_M.y, __SCREEN_M.z,
            1.0f, 1.0f, __SCREEN_T.x, __SCREEN_T.y, __SCREEN_T.z};
    for (int i = 0; i < 60; ++i)
        __SCREEN_VERTICES[i] = _[i];
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(__SCREEN_VERTICES), __SCREEN_VERTICES);
}

void Background::Draw()
{
    glDepthFunc(GL_LEQUAL);
    shader->use();
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 12);
    glBindVertexArray(0);
    glDepthFunc(GL_LESS);
}

void Background::SetUp()
{
    shader = std::make_shared<Shader>("framebuffer_screen");
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(__SCREEN_VERTICES), __SCREEN_VERTICES, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(2 * sizeof(float)));
}
