cmake_minimum_required(VERSION 3.20)
project(rig_editor CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# raylib (window, input, 3D rendering, model loading)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 6.0
    GIT_SHALLOW TRUE)

# Dear ImGui (no CMake support upstream, sources are compiled below)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9b
    GIT_SHALLOW TRUE)

# rlImGui (raylib backend for Dear ImGui)
FetchContent_Declare(rlimgui
    GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
    GIT_TAG 1550009359ad975927f7f0e4a3f47e3f27123ea9)

# ImGuizmo (3D translate/rotate gizmo for the bones); only ImGuizmo.cpp is needed
FetchContent_Declare(imguizmo
    GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
    GIT_TAG 18cef5e031d8c6973d80284c67f60549fafd78c1
    SOURCE_SUBDIR none)   # don't run its CMakeLists (it builds extra widgets/examples)

FetchContent_MakeAvailable(raylib imgui rlimgui imguizmo)

add_library(imgui_rl STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${rlimgui_SOURCE_DIR}/rlImGui.cpp
    ${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp)
target_include_directories(imgui_rl PUBLIC ${imgui_SOURCE_DIR} ${rlimgui_SOURCE_DIR} ${imguizmo_SOURCE_DIR}/src)
target_compile_definitions(imgui_rl PUBLIC NO_FONT_AWESOME)
target_link_libraries(imgui_rl PUBLIC raylib)

add_executable(rig_editor
    src/main.cpp
    src/skeleton.cpp
    src/viewport.cpp)
target_link_libraries(rig_editor PRIVATE imgui_rl)

if(APPLE)
    target_link_libraries(rig_editor PRIVATE "-framework IOKit" "-framework Cocoa" "-framework OpenGL")
endif()
