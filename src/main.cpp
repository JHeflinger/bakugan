#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
  #include <GL/gl.h>
#elif defined(__APPLE__)
  #define GL_SILENCE_DEPRECATION
  #include <OpenGL/gl.h>
#else
  #include <GL/gl.h>
#endif
#include <openvdb/openvdb.h>
#include <cstdio>
#include <cstring>

// Headless checks used by CI: no window or GPU needed.
static int run_tests() {
    int failures = 0;

    // OpenVDB: grid access plus a compressed write/read round trip.
    openvdb::initialize();
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setName("density");
    grid->setTransform(openvdb::math::Transform::createLinearTransform(0.1));
    grid->getAccessor().setValue(openvdb::Coord(1, 2, 3), 5.0f);
    if (grid->getAccessor().getValue(openvdb::Coord(1, 2, 3)) != 5.0f) {
        std::fprintf(stderr, "FAIL: openvdb accessor value\n"); failures++;
    }
    try {
        {
            openvdb::io::File out("tiny_test.vdb");
            out.write({ grid });
            out.close();
        }
        openvdb::io::File in("tiny_test.vdb");
        in.open();
        auto loaded = openvdb::gridPtrCast<openvdb::FloatGrid>(in.readGrid("density"));
        in.close();
        std::remove("tiny_test.vdb");
        if (!loaded || loaded->getAccessor().getValue(openvdb::Coord(1, 2, 3)) != 5.0f) {
            std::fprintf(stderr, "FAIL: openvdb round trip value\n"); failures++;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: openvdb file io: %s\n", e.what()); failures++;
    }

    // Dear ImGui: build one frame with no backend.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(800, 600);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* px; int tw, th;
    io.Fonts->GetTexDataAsRGBA32(&px, &tw, &th);
    // A window is hidden on its first frame, so run a few frames.
    for (int frame = 0; frame < 3; frame++) {
        ImGui::NewFrame();
        ImGui::Begin("test");
        ImGui::Text("hello");
        ImGui::End();
        ImGui::Render();
    }
    if (ImGui::GetDrawData()->TotalVtxCount <= 0) {
        std::fprintf(stderr, "FAIL: imgui produced no geometry\n"); failures++;
    }
    ImGui::DestroyContext();

    std::printf(failures ? "TESTS FAILED (%d)\n" : "ALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--test") == 0) return run_tests();

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // required for core profiles on macOS
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui App", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::ShowDemoWindow();
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
