# viewer

## 概述

viewer是一个轻量级实时渲染系统，提供基础的3D渲染功能，支持物理仿真的可视化需求。

## 特性

![示例图片](./data/readme.png "viewer示例")

- ✅ **网格渲染**：支持线框、纯色PBR、纹理渲染模式
- ✅ **阴影映射**：基于ShadowMapping的实时阴影
- ✅ **粒子系统**：球体渲染粒子
- ✅ **GPU数据交互**：支持CUDA显存到OpenGL VBO的数据传输
- ✅ **预设场景**：包含地面、4个可调点光源、FPS显示器
- ✅ **相机控制**：轨道相机，支持旋转、平移、缩放

## 系统架构

### 核心组件

#### RenderSystem（渲染系统）
- 管理相机、光源、渲染对象
- `Camera` 基类：提供视图矩阵和透视矩阵
- `Light` 基类：目前仅支持点光源
- `RenderObject`：可渲染物体的基类

#### RenderObject（渲染对象）
- 代表可渲染的物体（粒子、网格、线段、ImGUI窗口）
- 可包含多个`Renderer`组件
- 主要负责存储顶点数据
- 在`RenderObject::Draw()`时调用内部`Renderer`进行绘制

#### Renderer（渲染器）
- 负责具体渲染逻辑
- 存储纹理、材质信息
- 支持的类型：
  - `PbrRenderer`：纯色PBR渲染
  - `SolidColorRenderer`（线框模式）：纯色线框渲染
  - `TextureRenderer`：纹理渲染
  - `SphereRenderer`：球体渲染（用于粒子）

#### Object（对象基类）
- 处理控制事件（键盘鼠标）
- 每帧更新逻辑
- 用于打包复杂的渲染结构

#### Connector（数据连接器）
- 负责向渲染结构传输数据
- 支持CUDA device数组到Mesh的顶点位置传输
- 支持位置和颜色的device数组到ParticleBatch的更新

## 关键概念

### 渲染流程
1. **初始化**：创建RenderObject并添加Renderer
2. **数据准备**：通过Connector从仿真器传输数据到渲染对象
3. **渲染循环**：每帧调用RenderObject::Draw()，内部调用Renderer进行具体绘制

### 对象管理
- **仅渲染对象**：只需添加到RenderSystem
- **需要交互/更新的对象**：需同时作为Object添加到App中

### 数据流
```
物理仿真器 → Connector → RenderObject → Renderer → 屏幕
```

## 快速开始

### 基础用法示例

参考example/app_example

```cpp
// 初始化应用窗口
auto app = GLFWApp::GetInstance("App", 1600, 900);
std::shared_ptr<RenderSystem> render_system = app->GetRenderSystem();

// 设置相机
auto camera = std::make_shared<OrbitCamera>(glm::radians(-45.0f), glm::radians(-45.0f), 10.0f);
app->SetCamera(camera);
app->AddObject(camera);

// 添加地面
auto floor = std::make_shared<Floor>(100.0f);
render_system->AddRenderObject(floor->GetMesh());

// 加载网格模型 - 线框渲染
auto venus = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/venus.obj", 0.01f, 
                                    glm::vec3(-1.0f, 0.0f, 0.0f), 
                                    glm::vec3(glm::radians(-90.0f), 0.0f, 0.0f), 
                                    std::vector<glm::vec3>({glm::vec3(1.0f), glm::vec3(0.1f)}));
venus->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0f, 0.0f, 0.0f), true)); // 线框渲染器
render_system->AddRenderObject(venus);

// 加载带纹理的网格
auto squirel = MeshLoader::LoadMesh(std::string(VIEWER_DIR) + "/data/objs/CartoonSquirelModel/CartoonSquirelModel.obj", 
                                    0.5f, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f),
                                    std::string(VIEWER_DIR) + "/data/objs/CartoonSquirelModel/DiffSqurel.tga");
render_system->AddRenderObject(squirel);

// 添加光源场景
auto light_scene = std::make_shared<LightScene>();
app->AddObject(light_scene);

// 添加线段
auto lines = std::make_shared<LineSegment>(
    std::vector<glm::vec3>({/* 顶点坐标 */}), 
    std::vector<std::pair<int, int>>({{/* 线段连接 */}})
);
lines->AddRenderer(std::make_shared<SolidColorRenderer>(glm::vec3(1.0f, 1.0f, 1.0f), true));
render_system->AddRenderObject(lines);

// 添加粒子系统
auto particles = std::make_shared<ParticleBatch>(std::vector<Particle>({
    {glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)}, 
    {glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)}, 
    {glm::vec3(0.0f, 6.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)}
}));
particles->AddRenderer(std::make_shared<SphereRenderer>(glm::vec2(0.1f), 0.2f));
render_system->AddRenderObject(particles);

// 运行应用
app->Run();
```

### 与物理仿真集成

将物理仿真求解器包装为Object，在每帧更新中执行仿真步进并通过Connector传输数据：

```cpp
virtual void Update(double delta_time) override
{
    if (m_play)
    {
        // 执行仿真步进并计算耗时
        auto start = std::chrono::high_resolution_clock::now();
        m_solver->Step();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(end - start);
        m_fps_monitor->UpdateFPS((double)duration.count());

        // 从求解器向渲染器传输数据
        for (auto& connector : m_connectors)
        {
            connector->TransferData();
        }
    }
}
```

Connector设置：

```cpp
auto connector = std::make_shared<viewer::MeshConnector<Real>>(mesh, position_ptr);
m_connectors.push_back(connector);
```

## 目录结构
```
viewer/
├── Connector/         # 数据传输更新器
├── Event/             # 事件结构体
├── Framework/         # 底层框架
├── GUI/               # 用户界面
├── Object/            # 对象类（相机、地面、光源等）
├── RenderObject/      # 渲染对象（网格、粒子等）
├── Renderer/          # 渲染器实现
└── utils/             # 工具类（模型加载器等）

```
