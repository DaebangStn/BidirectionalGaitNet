/**
 * @brief Minimal test application for ViewerAppBase
 *
 * This tests that the base class infrastructure works correctly
 * before migrating existing apps.
 */
#include "common/ViewerAppBase.h"
#include "GLfunctions.h"
#include <iostream>

class TestViewerApp : public ViewerAppBase
{
public:
    TestViewerApp() : ViewerAppBase("ViewerAppBase Test", 1280, 720) {
        std::cout << "[TestViewerApp] Created" << std::endl;
    }

protected:
    void onInitialize() override {
        std::cout << "[TestViewerApp] onInitialize called" << std::endl;
    }

    void drawContent() override {
        // Draw a simple colored cube at origin
        glEnable(GL_LIGHTING);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glEnable(GL_COLOR_MATERIAL);

        glColor4f(0.3f, 0.6f, 0.9f, 1.0f);  // Blue
        GUI::DrawCube(Eigen::Vector3d(0.5, 0.5, 0.5));

        // Draw coordinate axes
        glDisable(GL_LIGHTING);
        glLineWidth(2.0f);

        glBegin(GL_LINES);
        // X axis - red
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
        // Y axis - green
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
        // Z axis - blue
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();

        glLineWidth(1.0f);
    }

    void drawUI() override {
        // Simple test UI
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("ViewerAppBase Test")) {
            ImGui::Text("Camera Controls:");
            ImGui::BulletText("Left drag: Rotate");
            ImGui::BulletText("Right drag: Pan");
            ImGui::BulletText("Scroll: Zoom");
            ImGui::Separator();
            ImGui::Text("Keyboard:");
            ImGui::BulletText("1/2/3: View planes");
            ImGui::BulletText("R: Reset camera");
            ImGui::BulletText("G: Toggle ground");
            ImGui::BulletText("ESC: Exit");
            ImGui::Separator();

            ImGui::Text("Camera State:");
            ImGui::Text("  Zoom: %.2f", mCamera.zoom);
            ImGui::Text("  Focus: %d", static_cast<int>(mCamera.focus));
            ImGui::Text("  Ground: %s", mRenderGround ? "On" : "Off");
        }
        ImGui::End();
    }

    void keyPress(int key, int scancode, int action, int mods) override {
        // Call base class for standard shortcuts
        ViewerAppBase::keyPress(key, scancode, action, mods);

        // App-specific keys
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_T) {
                std::cout << "[TestViewerApp] T pressed - test key" << std::endl;
            }
        }
    }
};

int main(int argc, char** argv)
{
    std::cout << "========================================" << std::endl;
    std::cout << "    ViewerAppBase Test Application" << std::endl;
    std::cout << "========================================" << std::endl;

    TestViewerApp app;
    app.startLoop();

    std::cout << "[TestViewerApp] Exited cleanly" << std::endl;
    return 0;
}
