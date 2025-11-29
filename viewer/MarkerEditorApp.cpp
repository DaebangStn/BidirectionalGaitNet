#include "MarkerEditorApp.h"
#include "DARTHelper.h"
#include "UriResolver.h"
#include "Log.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <yaml-cpp/yaml.h>

using namespace dart::dynamics;

// =============================================================================
// Constructor / Destructor
// =============================================================================

MarkerEditorApp::MarkerEditorApp(int argc, char** argv)
    : mWindow(nullptr)
    , mWidth(1280)
    , mHeight(720)
    , mWindowXPos(0)
    , mWindowYPos(0)
    , mZoom(1.0)
    , mPersp(45.0)
    , mMouseDown(false)
    , mRotate(false)
    , mTranslate(false)
    , mMouseX(0)
    , mMouseY(0)
    , mSelectedMarkerIndex(-1)
    , mReferenceMarkerIndex(-1)
    , mShowLabels(true)
    , mShowPlaneXY(false)
    , mShowPlaneYZ(false)
    , mShowPlaneZX(false)
    , mPlaneOpacity(0.5f)
    , mRenderMode(RenderMode::Overlay)
    , mNewMarkerBoneIndex(0)
{
    // Initialize buffers
    std::memset(mSearchFilter, 0, sizeof(mSearchFilter));
    std::memset(mNewMarkerName, 0, sizeof(mNewMarkerName));
    std::memset(mBodyNodeFilter, 0, sizeof(mBodyNodeFilter));
    std::memset(mExportSkeletonPath, 0, sizeof(mExportSkeletonPath));
    std::strcpy(mNewMarkerName, "NewMarker");
    std::strcpy(mExportSkeletonPath, "data/skeleton/edited.yaml");
    mSelectedBodyNodeIndex = -1;

    // Initialize URI resolver for @data/ scheme
    PMuscle::URIResolver::getInstance().initialize();

    // Default paths
    mSkeletonPath = "@data/skeleton/base.xml";
    mMarkerPath = "@data/marker/default.xml";
    mExportPath = "test";  // Just the stem, will be prefixed with data/marker/ and suffixed with .xml

    // Parse command line arguments
    auto printHelp = [&]() {
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "\nOptions:\n"
                  << "  -s, --skeleton <path>  Load skeleton from YAML file\n"
                  << "  -m, --markers <path>   Load marker set from XML file\n"
                  << "  -h, --help             Show this help message\n"
                  << "\nControls:\n"
                  << "  Left drag              Rotate camera\n"
                  << "  Right drag             Pan camera\n"
                  << "  Scroll                 Zoom\n"
                  << "  Click marker list      Select marker\n"
                  << "  Shift+Click            Set reference marker (for align/mirror)\n"
                  << "  1/2/3                  Align camera to XY/YZ/ZX plane\n"
                  << "  O                      Cycle render mode (Primitive/Mesh/Overlay/Wireframe)\n"
                  << "  L                      Toggle marker labels\n"
                  << std::endl;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp();
            exit(0);
        } else if ((arg == "--skeleton" || arg == "-s") && i + 1 < argc) {
            mSkeletonPath = argv[++i];
        } else if ((arg == "--markers" || arg == "-m") && i + 1 < argc) {
            mMarkerPath = argv[++i];
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n\n";
            printHelp();
            exit(1);
        }
    }

    // Camera defaults
    mEye = Eigen::Vector3d(0.0, 0.0, 2.5);
    mUp = Eigen::Vector3d(0.0, 1.0, 0.0);
    mTrans = Eigen::Vector3d(0.0, -0.8, 0.0);

    // Load window config from render.yaml
    loadRenderConfig();

    // Initialize GLFW
    if (!glfwInit()) {
        LOG_ERROR("[MarkerEditor] Failed to initialize GLFW");
        exit(1);
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    mWindow = glfwCreateWindow(mWidth, mHeight, "Marker Editor", nullptr, nullptr);
    if (!mWindow) {
        LOG_ERROR("[MarkerEditor] Failed to create GLFW window");
        glfwTerminate();
        exit(1);
    }

    // Set window position from render.yaml
    glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);

    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOG_ERROR("[MarkerEditor] Failed to initialize GLAD");
        exit(1);
    }

    // Set up callbacks
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, cursorPosCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);
    glfwSetKeyCallback(mWindow, keyCallback);

    // Initialize ImGui
    initImGui();

    // Load skeleton and markers
    loadSkeleton(mSkeletonPath);
    if (mCharacter) {
        loadMarkers(mMarkerPath);
        updateBodyNodeNames();
    }

    LOG_INFO("[MarkerEditor] Initialized with skeleton: " << mSkeletonPath);
}

MarkerEditorApp::~MarkerEditorApp()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

// =============================================================================
// Main Loop
// =============================================================================

void MarkerEditorApp::startLoop()
{
    while (!glfwWindowShouldClose(mWindow)) {
        glfwPollEvents();

        // OpenGL rendering
        initGL();
        initLighting();
        setCamera();
        drawSkeleton();
        drawMarkers();

        // ImGui rendering
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawMarkerLabels();  // Must be after NewFrame (uses ImGui draw list)
        drawEditorPanel();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(mWindow);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void MarkerEditorApp::initGL()
{
    glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void MarkerEditorApp::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 150");
}

void MarkerEditorApp::initLighting()
{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat light_position[] = {1.0f, 2.0f, 1.5f, 0.0f};
    GLfloat light_ambient[] = {0.3f, 0.3f, 0.3f, 1.0f};
    GLfloat light_diffuse[] = {0.7f, 0.7f, 0.7f, 1.0f};

    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
}

void MarkerEditorApp::setCamera()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, mWidth, mHeight);
    gluPerspective(mPersp, (double)mWidth / (double)mHeight, 0.1, 100.0);
    gluLookAt(mEye[0], mEye[1], mEye[2], 0.0, 0.0, 0.0, mUp[0], mUp[1], mUp[2]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Trackball rotation
    mTrackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
    mTrackball.setRadius(std::min(mWidth, mHeight) * 0.4);
    mTrackball.applyGLRotation();

    // Zoom and translation
    glScalef(mZoom, mZoom, mZoom);
    glTranslatef(mTrans[0], mTrans[1], mTrans[2]);
}

// =============================================================================
// Rendering
// =============================================================================

void MarkerEditorApp::drawSkeleton()
{
    if (!mCharacter) return;

    auto skel = mCharacter->getSkeleton();

    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glColor4f(0.7f, 0.7f, 0.8f, 0.9f);

    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        // Check visibility (skip if hidden)
        if (i < mBodyNodeVisible.size() && !mBodyNodeVisible[i]) {
            continue;
        }
        drawSingleBodyNode(skel->getBodyNode(i));
    }
}

void MarkerEditorApp::drawSingleBodyNode(const BodyNode* bn)
{
    if (!bn) return;

    glPushMatrix();
    glMultMatrixd(bn->getTransform().data());

    bn->eachShapeNodeWith<VisualAspect>([this](const ShapeNode* sn) {
        if (!sn) return true;

        const auto& va = sn->getVisualAspect();
        if (!va || va->isHidden()) return true;

        glPushMatrix();
        Eigen::Affine3d tmp = sn->getRelativeTransform();
        glMultMatrixd(tmp.data());

        const auto* shape = sn->getShape().get();
        Eigen::Vector4d color(0.6, 0.6, 0.7, 1.0);

        // Render mesh (for Mesh and Overlay modes)
        if (mRenderMode == RenderMode::Mesh || mRenderMode == RenderMode::Overlay) {
            if (shape->is<MeshShape>()) {
                const auto* mesh = dynamic_cast<const MeshShape*>(shape);
                mShapeRenderer.renderMesh(mesh, false, 0.0, color);
            }
        }

        // Render primitive shapes (for Primitive, Overlay, and Wireframe modes)
        if (mRenderMode == RenderMode::Primitive || mRenderMode == RenderMode::Overlay || mRenderMode == RenderMode::Wireframe) {
            // For overlay and wireframe modes, render primitives as wireframe
            if (mRenderMode == RenderMode::Overlay || mRenderMode == RenderMode::Wireframe) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glLineWidth(2.0f);
                if (mRenderMode == RenderMode::Wireframe) {
                    glColor4f(0.8f, 0.8f, 0.8f, 1.0f);  // Light gray wireframe
                } else {
                    glColor4f(0.2f, 0.8f, 0.2f, 1.0f);  // Green wireframe for overlay
                }
            }

            if (shape->is<BoxShape>()) {
                GUI::DrawCube(static_cast<const BoxShape*>(shape)->getSize());
            } else if (shape->is<CapsuleShape>()) {
                auto* cap = static_cast<const CapsuleShape*>(shape);
                GUI::DrawCapsule(cap->getRadius(), cap->getHeight());
            } else if (shape->is<SphereShape>()) {
                GUI::DrawSphere(static_cast<const SphereShape*>(shape)->getRadius());
            } else if (shape->is<CylinderShape>()) {
                auto* cyl = static_cast<const CylinderShape*>(shape);
                GUI::DrawCylinder(cyl->getRadius(), cyl->getHeight());
            }

            // Restore fill mode and line width after wireframe
            if (mRenderMode == RenderMode::Overlay || mRenderMode == RenderMode::Wireframe) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glLineWidth(1.0f);
            }
        }

        glPopMatrix();
        return true;
    });

    glPopMatrix();
}

void MarkerEditorApp::drawMarkers()
{
    if (!mCharacter || !mCharacter->hasMarkers()) return;

    glDisable(GL_LIGHTING);

    const auto& markers = mCharacter->getMarkers();
    for (size_t i = 0; i < markers.size(); ++i) {
        // Skip hidden markers
        if (i < mMarkerVisible.size() && !mMarkerVisible[i]) continue;

        Eigen::Vector3d pos = markers[i].getGlobalPos();
        if (!pos.array().isFinite().all()) continue;

        // Selected marker: yellow, larger; Reference marker: cyan; Others: red
        if (static_cast<int>(i) == mSelectedMarkerIndex) {
            glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
            GUI::DrawSphere(pos, 0.02);
        } else if (static_cast<int>(i) == mReferenceMarkerIndex) {
            glColor4f(0.3f, 0.8f, 1.0f, 1.0f);  // Cyan for reference
            GUI::DrawSphere(pos, 0.018);
        } else {
            glColor4f(1.0f, 0.3f, 0.3f, 0.9f);
            GUI::DrawSphere(pos, 0.0125);
        }
    }

    // Draw axis planes at selected marker
    if (mSelectedMarkerIndex >= 0 && mSelectedMarkerIndex < static_cast<int>(markers.size())) {
        Eigen::Vector3d pos = markers[mSelectedMarkerIndex].getGlobalPos();
        if (pos.array().isFinite().all()) {
            const double planeSize = 1000.0;  // Very large planes

            // Draw transparent planes
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_FALSE);  // Don't write to depth buffer

            // XY plane (Blue tint - Z constant)
            if (mShowPlaneXY) {
                glBegin(GL_QUADS);
                glColor4f(0.0f, 0.0f, 1.0f, mPlaneOpacity);
                glVertex3d(pos[0] - planeSize, pos[1] - planeSize, pos[2]);
                glVertex3d(pos[0] + planeSize, pos[1] - planeSize, pos[2]);
                glVertex3d(pos[0] + planeSize, pos[1] + planeSize, pos[2]);
                glVertex3d(pos[0] - planeSize, pos[1] + planeSize, pos[2]);
                glEnd();
            }

            // YZ plane (Red tint - X constant)
            if (mShowPlaneYZ) {
                glBegin(GL_QUADS);
                glColor4f(1.0f, 0.0f, 0.0f, mPlaneOpacity);
                glVertex3d(pos[0], pos[1] - planeSize, pos[2] - planeSize);
                glVertex3d(pos[0], pos[1] + planeSize, pos[2] - planeSize);
                glVertex3d(pos[0], pos[1] + planeSize, pos[2] + planeSize);
                glVertex3d(pos[0], pos[1] - planeSize, pos[2] + planeSize);
                glEnd();
            }

            // ZX plane (Green tint - Y constant)
            if (mShowPlaneZX) {
                glBegin(GL_QUADS);
                glColor4f(0.0f, 1.0f, 0.0f, mPlaneOpacity);
                glVertex3d(pos[0] - planeSize, pos[1], pos[2] - planeSize);
                glVertex3d(pos[0] + planeSize, pos[1], pos[2] - planeSize);
                glVertex3d(pos[0] + planeSize, pos[1], pos[2] + planeSize);
                glVertex3d(pos[0] - planeSize, pos[1], pos[2] + planeSize);
                glEnd();
            }

            glDepthMask(GL_TRUE);

            // Draw axis lines
            const double axisLength = 1000.0;
            const float lineWidth = 1.5f;

            glLineWidth(lineWidth);
            glBegin(GL_LINES);
            // X axis - Red
            glColor4f(1.0f, 0.0f, 0.0f, 0.7f);
            glVertex3d(pos[0] - axisLength, pos[1], pos[2]);
            glVertex3d(pos[0] + axisLength, pos[1], pos[2]);
            // Y axis - Green
            glColor4f(0.0f, 1.0f, 0.0f, 0.7f);
            glVertex3d(pos[0], pos[1] - axisLength, pos[2]);
            glVertex3d(pos[0], pos[1] + axisLength, pos[2]);
            // Z axis - Blue
            glColor4f(0.0f, 0.0f, 1.0f, 0.7f);
            glVertex3d(pos[0], pos[1], pos[2] - axisLength);
            glVertex3d(pos[0], pos[1], pos[2] + axisLength);
            glEnd();
            glLineWidth(1.0f);
        }
    }

    // Store matrices for later label drawing (after ImGui::NewFrame)
    if (mShowLabels) {
        glGetDoublev(GL_MODELVIEW_MATRIX, mModelview);
        glGetDoublev(GL_PROJECTION_MATRIX, mProjection);
        glGetIntegerv(GL_VIEWPORT, mViewport);
    }

    glEnable(GL_LIGHTING);
}

void MarkerEditorApp::drawMarkerLabels()
{
    if (!mCharacter || !mCharacter->hasMarkers() || !mShowLabels) return;

    const auto& markers = mCharacter->getMarkers();
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();  // Behind ImGui panels
    ImFont* font = ImGui::GetFont();

    const float fontSize = 18.0f;
    const float outlineOffset = 1.0f;
    const ImU32 outlineColor = IM_COL32(0, 0, 0, 255);

    for (size_t i = 0; i < markers.size(); ++i) {
        // Skip hidden markers
        if (i < mMarkerVisible.size() && !mMarkerVisible[i]) continue;

        Eigen::Vector3d pos = markers[i].getGlobalPos();
        if (!pos.array().isFinite().all()) continue;

        GLdouble screenX, screenY, screenZ;
        if (gluProject(pos.x(), pos.y(), pos.z(), mModelview, mProjection, mViewport,
                       &screenX, &screenY, &screenZ) == GL_TRUE) {
            if (screenZ > 0.0 && screenZ < 1.0) {
                float x = static_cast<float>(screenX) + 5;
                float y = mHeight - static_cast<float>(screenY) - 5;
                std::string label = std::to_string(i) + ": " + markers[i].name;

                // Determine fill color
                ImU32 fillColor;
                if (static_cast<int>(i) == mSelectedMarkerIndex) {
                    fillColor = IM_COL32(255, 200, 0, 255);  // Yellow for selected
                } else if (static_cast<int>(i) == mReferenceMarkerIndex) {
                    fillColor = IM_COL32(100, 200, 255, 255);  // Cyan for reference
                } else {
                    fillColor = IM_COL32(255, 255, 255, 255);  // White for others
                }

                // Draw black outline (8 directions)
                drawList->AddText(font, fontSize, ImVec2(x - outlineOffset, y), outlineColor, label.c_str());
                drawList->AddText(font, fontSize, ImVec2(x + outlineOffset, y), outlineColor, label.c_str());
                drawList->AddText(font, fontSize, ImVec2(x, y - outlineOffset), outlineColor, label.c_str());
                drawList->AddText(font, fontSize, ImVec2(x, y + outlineOffset), outlineColor, label.c_str());
                drawList->AddText(font, fontSize, ImVec2(x - outlineOffset, y - outlineOffset), outlineColor, label.c_str());
                drawList->AddText(font, fontSize, ImVec2(x + outlineOffset, y - outlineOffset), outlineColor, label.c_str());
                drawList->AddText(font, fontSize, ImVec2(x - outlineOffset, y + outlineOffset), outlineColor, label.c_str());
                drawList->AddText(font, fontSize, ImVec2(x + outlineOffset, y + outlineOffset), outlineColor, label.c_str());

                // Draw fill color on top
                drawList->AddText(font, fontSize, ImVec2(x, y), fillColor, label.c_str());
            }
        }
    }
}

// =============================================================================
// ImGui Editor Panel
// =============================================================================

void MarkerEditorApp::drawEditorPanel()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, mHeight - 20), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Marker Editor")) {
        ImGui::End();
        return;
    }

    // File section
    if (ImGui::CollapsingHeader("Files", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Skeleton: %s", mSkeletonPath.c_str());
        ImGui::Text("Markers:  %s", mMarkerPath.c_str());
        ImGui::Separator();

        // Export marker path input (user enters stem only)
        static char exportBuf[256];
        std::strncpy(exportBuf, mExportPath.c_str(), sizeof(exportBuf) - 1);
        ImGui::Text("data/marker/");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputText(".xml", exportBuf, sizeof(exportBuf))) {
            mExportPath = exportBuf;
        }
        ImGui::SameLine();
        if (ImGui::Button("Save##Marker")) {
            std::string fullPath = "data/marker/" + mExportPath + ".xml";
            exportMarkers(fullPath);
        }

        // // Export skeleton path input
        // ImGui::SetNextItemWidth(250);
        // if (ImGui::InputText("Skeleton", mExportSkeletonPath, sizeof(mExportSkeletonPath))) {
        //     // path updated in buffer
        // }
        // ImGui::SameLine();
        // if (ImGui::Button("Save##Skeleton")) {
        //     exportSkeleton(std::string(mExportSkeletonPath));
        // }
    }

    // Options
    if (ImGui::CollapsingHeader("Options")) {
        ImGui::Checkbox("Show Labels", &mShowLabels);

        ImGui::Separator();
        ImGui::Text("Axis Planes:");
        ImGui::Checkbox("XY", &mShowPlaneXY);
        ImGui::SameLine();
        ImGui::Checkbox("YZ", &mShowPlaneYZ);
        ImGui::SameLine();
        ImGui::Checkbox("ZX", &mShowPlaneZX);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(50);
        ImGui::DragFloat("##PlaneOpacity", &mPlaneOpacity, 0.01f, 0.0f, 1.0f, "%.2f");

        ImGui::Separator();
        ImGui::Text("Render Mode (O to cycle):");
        int mode = static_cast<int>(mRenderMode);
        if (ImGui::RadioButton("Primitive", &mode, 0)) mRenderMode = RenderMode::Primitive;
        ImGui::SameLine();
        if (ImGui::RadioButton("Mesh", &mode, 1)) mRenderMode = RenderMode::Mesh;
        ImGui::SameLine();
        if (ImGui::RadioButton("Overlay", &mode, 2)) mRenderMode = RenderMode::Overlay;
        ImGui::SameLine();
        if (ImGui::RadioButton("Wire", &mode, 3)) mRenderMode = RenderMode::Wireframe;
    }

    // Body nodes visibility
    drawBodyNodesSection();

    // Marker list section
    drawMarkerListSection();

    // Selected marker section
    drawSelectedMarkerSection();

    ImGui::End();
}

void MarkerEditorApp::drawMarkerListSection()
{
    if (!mCharacter) return;

    if (ImGui::CollapsingHeader("Markers", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto& markers = mCharacter->getMarkersForEdit();

        // Search filter
        ImGui::SetNextItemWidth(-100);
        ImGui::InputText("##Search", mSearchFilter, sizeof(mSearchFilter));
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            mSearchFilter[0] = '\0';
        }

        // Add marker button
        ImGui::SameLine();
        if (ImGui::Button("+Add")) {
            ImGui::OpenPopup("AddMarkerPopup");
        }
        drawAddMarkerPopup();

        // Marker table
        if (ImGui::BeginTable("MarkerTable", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg,
            ImVec2(0, 350)))
        {
            ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed, 35.0f);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Bone", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("V", ImGuiTableColumnFlags_WidthFixed, 25.0f);
            ImGui::TableHeadersRow();

            std::string filterLower = mSearchFilter;
            std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

            for (size_t i = 0; i < markers.size(); ++i) {
                const auto& marker = markers[i];

                // Filter by name or index
                if (filterLower.length() > 0) {
                    std::string nameLower = marker.name;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                    std::string idxStr = std::to_string(i);
                    bool nameMatch = nameLower.find(filterLower) != std::string::npos;
                    bool idxMatch = idxStr.find(filterLower) != std::string::npos;
                    if (!nameMatch && !idxMatch) continue;
                }

                ImGui::TableNextRow();

                // Highlight selected row (yellow) and reference row (cyan)
                if (static_cast<int>(i) == mSelectedMarkerIndex) {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, IM_COL32(255, 255, 100, 80));
                } else if (static_cast<int>(i) == mReferenceMarkerIndex) {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, IM_COL32(100, 200, 255, 80));
                }

                // Index column
                ImGui::TableSetColumnIndex(0);
                if (ImGui::Selectable(std::to_string(i).c_str(), mSelectedMarkerIndex == static_cast<int>(i),
                    ImGuiSelectableFlags_SpanAllColumns)) {
                    // Shift+Click sets reference marker, normal click sets selected marker
                    if (ImGui::GetIO().KeyShift) {
                        mReferenceMarkerIndex = static_cast<int>(i);
                    } else {
                        mSelectedMarkerIndex = static_cast<int>(i);
                    }
                }

                // Name column
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%s", marker.name.c_str());

                // Bone column
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", marker.bodyNode ? marker.bodyNode->getName().c_str() : "?");

                // Visibility checkbox
                ImGui::TableSetColumnIndex(3);
                ImGui::PushID(static_cast<int>(i));
                // Ensure visibility vector is large enough
                if (mMarkerVisible.size() <= i) {
                    mMarkerVisible.resize(i + 1, true);
                }
                bool visible = mMarkerVisible[i];
                if (ImGui::Checkbox("##vis", &visible)) {
                    mMarkerVisible[i] = visible;
                }
                ImGui::PopID();
            }

            ImGui::EndTable();
        }

        ImGui::Text("Total: %zu markers", markers.size());
    }
}

void MarkerEditorApp::drawSelectedMarkerSection()
{
    if (!mCharacter) return;

    auto& markers = mCharacter->getMarkersForEdit();
    if (mSelectedMarkerIndex < 0 || mSelectedMarkerIndex >= static_cast<int>(markers.size())) {
        if (ImGui::CollapsingHeader("Selected Marker")) {
            ImGui::TextDisabled("No marker selected");
        }
        return;
    }

    if (ImGui::CollapsingHeader("Selected Marker", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto& marker = markers[mSelectedMarkerIndex];

        // Name editing
        static char nameBuf[64];
        std::strncpy(nameBuf, marker.name.c_str(), sizeof(nameBuf) - 1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
            marker.name = nameBuf;
        }

        // Bone selection dropdown
        int currentBoneIdx = 0;
        std::string currentBoneName = marker.bodyNode ? marker.bodyNode->getName() : "";
        for (size_t i = 0; i < mBodyNodeNames.size(); ++i) {
            if (mBodyNodeNames[i] == currentBoneName) {
                currentBoneIdx = static_cast<int>(i);
                break;
            }
        }

        if (!mBodyNodeNames.empty()) {
            std::vector<const char*> boneItems;
            for (const auto& name : mBodyNodeNames) {
                boneItems.push_back(name.c_str());
            }

            ImGui::SetNextItemWidth(-1);
            if (ImGui::Combo("Bone", &currentBoneIdx, boneItems.data(), static_cast<int>(boneItems.size()))) {
                // Save current global position before changing bone
                Eigen::Vector3d savedGlobalPos = marker.getGlobalPos();

                // Change to new body node
                auto* newBodyNode = mCharacter->getSkeleton()->getBodyNode(mBodyNodeNames[currentBoneIdx]);
                marker.bodyNode = newBodyNode;

                // Recompute offset to maintain the same global position
                if (newBodyNode) {
                    auto* shapeNode = newBodyNode->getShapeNodeWith<dart::dynamics::VisualAspect>(0);
                    if (shapeNode) {
                        const auto* boxShape = dynamic_cast<const dart::dynamics::BoxShape*>(shapeNode->getShape().get());
                        if (boxShape) {
                            Eigen::Vector3d size = boxShape->getSize();
                            // Convert global position to local frame
                            Eigen::Vector3d localPos = newBodyNode->getTransform().inverse() * savedGlobalPos;
                            // Compute normalized offset: offset = 2 * localPos / size
                            marker.offset = Eigen::Vector3d(
                                (std::abs(size[0]) > 1e-6) ? 2.0 * localPos[0] / size[0] : 0.0,
                                (std::abs(size[1]) > 1e-6) ? 2.0 * localPos[1] / size[1] : 0.0,
                                (std::abs(size[2]) > 1e-6) ? 2.0 * localPos[2] / size[2] : 0.0
                            );
                        }
                    }
                }
            }
        }

        ImGui::Separator();

        // Offset editing with drag input
        ImGui::Text("Offset (normalized):");
        float offset[3] = {
            static_cast<float>(marker.offset[0]),
            static_cast<float>(marker.offset[1]),
            static_cast<float>(marker.offset[2])
        };

        bool changed = false;
        const float dragSpeed = 0.01f;
        ImGui::SetNextItemWidth(-1);
        changed |= ImGui::DragFloat("X##offset", &offset[0], dragSpeed, 0.0f, 0.0f, "%.3f");
        ImGui::SetNextItemWidth(-1);
        changed |= ImGui::DragFloat("Y##offset", &offset[1], dragSpeed, 0.0f, 0.0f, "%.3f");
        ImGui::SetNextItemWidth(-1);
        changed |= ImGui::DragFloat("Z##offset", &offset[2], dragSpeed, 0.0f, 0.0f, "%.3f");

        if (changed) {
            marker.offset = Eigen::Vector3d(offset[0], offset[1], offset[2]);
        }

        // Center feature: set one axis of offset to 0
        ImGui::Separator();
        ImGui::Text("Center:");
        ImGui::SameLine();
        if (ImGui::RadioButton("X##center", false)) { marker.offset[0] = 0.0; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Y##center", false)) { marker.offset[1] = 0.0; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Z##center", false)) { marker.offset[2] = 0.0; }

        // Align feature: copy global position axis from reference marker
        if (mReferenceMarkerIndex >= 0 && mReferenceMarkerIndex != mSelectedMarkerIndex) {
            auto& markers = mCharacter->getMarkersForEdit();
            if (mReferenceMarkerIndex < static_cast<int>(markers.size())) {
                ImGui::Separator();
                ImGui::Text("Align to Ref (%s):", markers[mReferenceMarkerIndex].name.c_str());
                if (ImGui::Button("X##align")) { alignMarkerAxis(0); }
                ImGui::SameLine();
                if (ImGui::Button("Y##align")) { alignMarkerAxis(1); }
                ImGui::SameLine();
                if (ImGui::Button("Z##align")) { alignMarkerAxis(2); }

                ImGui::Text("Mirror w.r.t. Ref:");
                if (ImGui::Button("-X##mirror")) { mirrorMarkerAxis(0); }  // Mirror across YZ plane (flip X)
                ImGui::SameLine();
                if (ImGui::Button("-Y##mirror")) { mirrorMarkerAxis(1); }  // Mirror across ZX plane (flip Y)
                ImGui::SameLine();
                if (ImGui::Button("-Z##mirror")) { mirrorMarkerAxis(2); }  // Mirror across XY plane (flip Z)
            }
        }

        ImGui::Separator();

        // World position (editable)
        Eigen::Vector3d worldPos = marker.getGlobalPos();
        ImGui::Text("World Position:");
        float worldX = static_cast<float>(worldPos[0]);
        float worldY = static_cast<float>(worldPos[1]);
        float worldZ = static_cast<float>(worldPos[2]);
        bool worldChanged = false;
        ImGui::SetNextItemWidth(100);
        if (ImGui::DragFloat("WX", &worldX, 0.001f, -FLT_MAX, FLT_MAX, "%.4f")) worldChanged = true;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        if (ImGui::DragFloat("WY", &worldY, 0.001f, -FLT_MAX, FLT_MAX, "%.4f")) worldChanged = true;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        if (ImGui::DragFloat("WZ", &worldZ, 0.001f, -FLT_MAX, FLT_MAX, "%.4f")) worldChanged = true;

        if (worldChanged && marker.bodyNode) {
            Eigen::Vector3d newGlobalPos(worldX, worldY, worldZ);
            auto* shapeNode = marker.bodyNode->getShapeNodeWith<dart::dynamics::VisualAspect>(0);
            if (shapeNode) {
                const auto* boxShape = dynamic_cast<const dart::dynamics::BoxShape*>(shapeNode->getShape().get());
                if (boxShape) {
                    Eigen::Vector3d size = boxShape->getSize();
                    Eigen::Vector3d localPos = marker.bodyNode->getTransform().inverse() * newGlobalPos;
                    marker.offset = Eigen::Vector3d(
                        (std::abs(size[0]) > 1e-6) ? 2.0 * localPos[0] / size[0] : 0.0,
                        (std::abs(size[1]) > 1e-6) ? 2.0 * localPos[1] / size[1] : 0.0,
                        (std::abs(size[2]) > 1e-6) ? 2.0 * localPos[2] / size[2] : 0.0
                    );
                }
            }
        }

        ImGui::Separator();

        // Duplicate and Remove buttons
        if (ImGui::Button("Duplicate")) {
            mCharacter->duplicateMarker(mSelectedMarkerIndex);
            mSelectedMarkerIndex++;  // Select the new copy
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove")) {
            mCharacter->removeMarker(mSelectedMarkerIndex);
            // Adjust selection
            if (mSelectedMarkerIndex >= static_cast<int>(mCharacter->getMarkers().size())) {
                mSelectedMarkerIndex = static_cast<int>(mCharacter->getMarkers().size()) - 1;
            }
            // Clear reference if it was removed or shifted
            if (mReferenceMarkerIndex >= static_cast<int>(mCharacter->getMarkers().size())) {
                mReferenceMarkerIndex = -1;
            }
        }
    }
}

void MarkerEditorApp::drawBodyNodesSection()
{
    if (!mCharacter || mBodyNodeNames.empty()) return;

    if (ImGui::CollapsingHeader("Body Nodes")) {
        // Search filter
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##BodyNodeSearch", "Search body nodes...", mBodyNodeFilter, sizeof(mBodyNodeFilter));

        // Buttons row
        if (ImGui::Button("Show All")) {
            std::fill(mBodyNodeVisible.begin(), mBodyNodeVisible.end(), true);
        }
        ImGui::SameLine();
        if (ImGui::Button("Hide All")) {
            std::fill(mBodyNodeVisible.begin(), mBodyNodeVisible.end(), false);
        }
        ImGui::SameLine();
        if (ImGui::Button("Fit Visible")) {
            fitVisibleBoxesToMesh();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fit all visible boxes to mesh AABBs");
        }
        ImGui::SameLine();
        bool canUndo = canUndoFit();
        if (!canUndo) ImGui::BeginDisabled();
        if (ImGui::Button("Undo Fit")) {
            undoFit();
        }
        if (!canUndo) ImGui::EndDisabled();

        ImGui::Separator();

        // Scrollable list with checkboxes + selection (filtered)
        ImGui::BeginChild("BodyNodeList", ImVec2(0, 150), true);
        std::string filterLower = mBodyNodeFilter;
        std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

        for (size_t i = 0; i < mBodyNodeNames.size(); ++i) {
            // Apply search filter (case-insensitive)
            if (filterLower.length() > 0) {
                std::string nameLower = mBodyNodeNames[i];
                std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                if (nameLower.find(filterLower) == std::string::npos) {
                    continue;
                }
            }

            // Visibility checkbox (small, no label)
            bool visible = mBodyNodeVisible[i];
            std::string checkboxId = "##vis" + std::to_string(i);
            if (ImGui::Checkbox(checkboxId.c_str(), &visible)) {
                mBodyNodeVisible[i] = visible;
            }

            ImGui::SameLine();

            // Selectable name for geometry editing
            bool selected = (mSelectedBodyNodeIndex == static_cast<int>(i));
            if (ImGui::Selectable(mBodyNodeNames[i].c_str(), selected)) {
                mSelectedBodyNodeIndex = static_cast<int>(i);
            }
        }
        ImGui::EndChild();

        // Geometry editing section (when body node selected)
        if (mSelectedBodyNodeIndex >= 0 && mSelectedBodyNodeIndex < static_cast<int>(mBodyNodeNames.size())) {
            ImGui::Separator();
            ImGui::Text("Geometry: %s", mBodyNodeNames[mSelectedBodyNodeIndex].c_str());

            auto skel = mCharacter->getSkeleton();
            if (skel && mSelectedBodyNodeIndex < static_cast<int>(skel->getNumBodyNodes())) {
                auto bn = skel->getBodyNode(mSelectedBodyNodeIndex);

                // Show current shape info
                if (bn->getNumShapeNodes() > 0) {
                    auto shape = bn->getShapeNode(0)->getShape();

                    if (auto box = std::dynamic_pointer_cast<BoxShape>(shape)) {
                        Eigen::Vector3d size = box->getSize();
                        ImGui::Text("Box: %.4f x %.4f x %.4f", size[0], size[1], size[2]);
                    } else if (auto capsule = std::dynamic_pointer_cast<CapsuleShape>(shape)) {
                        ImGui::Text("Capsule: r=%.4f h=%.4f", capsule->getRadius(), capsule->getHeight());
                    } else if (auto mesh = std::dynamic_pointer_cast<MeshShape>(shape)) {
                        ImGui::Text("Mesh shape");
                    }
                }

                // Fit to Mesh button
                if (ImGui::Button("Fit Box to Mesh")) {
                    fitBoxToMesh(static_cast<size_t>(mSelectedBodyNodeIndex));
                }
            }
        }
    }
}

void MarkerEditorApp::drawAddMarkerPopup()
{
    if (ImGui::BeginPopup("AddMarkerPopup")) {
        ImGui::Text("Add New Marker");
        ImGui::Separator();

        ImGui::InputText("Name##new", mNewMarkerName, sizeof(mNewMarkerName));

        if (!mBodyNodeNames.empty()) {
            std::vector<const char*> boneItems;
            for (const auto& name : mBodyNodeNames) {
                boneItems.push_back(name.c_str());
            }
            ImGui::Combo("Bone##new", &mNewMarkerBoneIndex, boneItems.data(), static_cast<int>(boneItems.size()));
        }

        ImGui::Separator();
        if (ImGui::Button("Add", ImVec2(100, 0))) {
            if (mCharacter && mNewMarkerBoneIndex < static_cast<int>(mBodyNodeNames.size())) {
                mCharacter->addMarker(mNewMarkerName, mBodyNodeNames[mNewMarkerBoneIndex], Eigen::Vector3d::Zero());
                mSelectedMarkerIndex = static_cast<int>(mCharacter->getMarkers().size()) - 1;
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

// =============================================================================
// Input Callbacks
// =============================================================================

void MarkerEditorApp::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<MarkerEditorApp*>(glfwGetWindowUserPointer(window));
    app->resize(width, height);
}

void MarkerEditorApp::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    auto* app = static_cast<MarkerEditorApp*>(glfwGetWindowUserPointer(window));
    app->mousePress(button, action, mods);
}

void MarkerEditorApp::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* app = static_cast<MarkerEditorApp*>(glfwGetWindowUserPointer(window));
    app->mouseMove(xpos, ypos);
}

void MarkerEditorApp::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto* app = static_cast<MarkerEditorApp*>(glfwGetWindowUserPointer(window));
    app->mouseScroll(xoffset, yoffset);
}

void MarkerEditorApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto* app = static_cast<MarkerEditorApp*>(glfwGetWindowUserPointer(window));
    app->keyPress(key, scancode, action, mods);
}

void MarkerEditorApp::resize(int width, int height)
{
    mWidth = width;
    mHeight = height;
    glViewport(0, 0, width, height);
}

void MarkerEditorApp::mousePress(int button, int action, int mods)
{
    // Don't handle camera if ImGui wants mouse
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (action == GLFW_PRESS) {
        mMouseDown = true;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = true;
            mTrackball.startBall(mMouseX, mHeight - mMouseY);
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = true;
        }
    } else if (action == GLFW_RELEASE) {
        mMouseDown = false;
        mRotate = false;
        mTranslate = false;
    }
}

void MarkerEditorApp::mouseMove(double x, double y)
{
    double dx = x - mMouseX;
    double dy = y - mMouseY;
    mMouseX = x;
    mMouseY = y;

    if (ImGui::GetIO().WantCaptureMouse) return;

    if (mRotate) {
        mTrackball.updateBall(x, mHeight - y);
    }
    if (mTranslate) {
        double scale = 0.005 / mZoom;  // Scale with zoom level
        Eigen::Matrix3d rot = mTrackball.getRotationMatrix();
        Eigen::Vector3d delta = rot.transpose() * Eigen::Vector3d(dx * scale, -dy * scale, 0.0);
        mTrans += delta;
    }
}

void MarkerEditorApp::mouseScroll(double xoff, double yoff)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    mZoom *= (1.0 + yoff * 0.1);
    mZoom = std::max(0.1, std::min(10.0, mZoom));
}

void MarkerEditorApp::keyPress(int key, int scancode, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
                break;
            case GLFW_KEY_L:
                mShowLabels = !mShowLabels;
                break;
            case GLFW_KEY_O:
                // Cycle render mode: Primitive -> Mesh -> Overlay -> Wireframe -> Primitive
                mRenderMode = static_cast<RenderMode>((static_cast<int>(mRenderMode) + 1) % 4);
                break;
            case GLFW_KEY_R:
                // Reset camera
                mZoom = 1.0;
                mTrans = Eigen::Vector3d(0.0, -0.8, 0.0);
                mTrackball = dart::gui::Trackball();
                break;
            case GLFW_KEY_1:
            case GLFW_KEY_KP_1:
                alignCameraToPlane(1);  // XY plane
                break;
            case GLFW_KEY_2:
            case GLFW_KEY_KP_2:
                alignCameraToPlane(2);  // YZ plane
                break;
            case GLFW_KEY_3:
            case GLFW_KEY_KP_3:
                alignCameraToPlane(3);  // ZX plane
                break;
        }
    }
}

// =============================================================================
// File Operations
// =============================================================================

void MarkerEditorApp::loadSkeleton(const std::string& path)
{
    try {
        // Resolve URI scheme (e.g., @data/ -> absolute path)
        std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(path);
        mCharacter = std::make_unique<RenderCharacter>(resolvedPath);
        mSkeletonPath = path;  // Keep original path for display
        LOG_INFO("[MarkerEditor] Loaded skeleton: " << resolvedPath);
    } catch (const std::exception& e) {
        LOG_ERROR("[MarkerEditor] Failed to load skeleton: " << e.what());
        mCharacter = nullptr;
    }
}

void MarkerEditorApp::loadMarkers(const std::string& path)
{
    if (!mCharacter) return;
    // Resolve URI scheme (e.g., @data/ -> absolute path)
    std::string resolvedPath = PMuscle::URIResolver::getInstance().resolve(path);
    mCharacter->loadMarkers(resolvedPath);
    mMarkerPath = path;  // Keep original path for display
    mSelectedMarkerIndex = -1;
}

void MarkerEditorApp::exportMarkers(const std::string& path)
{
    if (!mCharacter) return;
    if (mCharacter->saveMarkersToXml(path)) {
        LOG_INFO("[MarkerEditor] Exported markers to: " << path);
    }
}

void MarkerEditorApp::updateBodyNodeNames()
{
    if (!mCharacter) {
        mBodyNodeNames.clear();
        mBodyNodeVisible.clear();
        return;
    }
    mBodyNodeNames = mCharacter->getBodyNodeNames();
    // Initialize all body nodes as visible
    mBodyNodeVisible.resize(mBodyNodeNames.size(), true);
    std::fill(mBodyNodeVisible.begin(), mBodyNodeVisible.end(), true);
}

void MarkerEditorApp::loadRenderConfig()
{
    try {
        std::string resolved_path = PMuscle::URIResolver::getInstance().resolve("render.yaml");
        YAML::Node config = YAML::LoadFile(resolved_path);

        if (config["geometry"] && config["geometry"]["window"]) {
            auto window = config["geometry"]["window"];
            if (window["width"])
                mWidth = window["width"].as<int>();
            if (window["height"])
                mHeight = window["height"].as<int>();
            if (window["xpos"])
                mWindowXPos = window["xpos"].as<int>();
            if (window["ypos"])
                mWindowYPos = window["ypos"].as<int>();
        }

        LOG_INFO("[MarkerEditor] Loaded render config: " << mWidth << "x" << mHeight
                 << " at (" << mWindowXPos << ", " << mWindowYPos << ")");
    } catch (const std::exception& e) {
        LOG_WARN("[MarkerEditor] Could not load render.yaml: " << e.what()
                 << " - using default window size");
    }
}

// =============================================================================
// Geometry Editing
// =============================================================================

namespace {
// Helper to recursively process aiNode and accumulate AABB
void processNodeAABB(const aiScene* scene, const aiNode* node,
                     const Eigen::Vector3d& scale,
                     const Eigen::Isometry3d& shapeTransform,
                     Eigen::Vector3d& minPt, Eigen::Vector3d& maxPt)
{
    // Process all meshes in this node
    for (unsigned int m = 0; m < node->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[m]];
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            // Apply scale first
            Eigen::Vector3d vertex(
                mesh->mVertices[v].x * scale[0],
                mesh->mVertices[v].y * scale[1],
                mesh->mVertices[v].z * scale[2]
            );
            // Transform to body-local frame using shape's relative transform
            vertex = shapeTransform * vertex;
            minPt = minPt.cwiseMin(vertex);
            maxPt = maxPt.cwiseMax(vertex);
        }
    }

    // Recursively process children
    for (unsigned int c = 0; c < node->mNumChildren; ++c) {
        processNodeAABB(scene, node->mChildren[c], scale, shapeTransform, minPt, maxPt);
    }
}
} // anonymous namespace

Eigen::Vector3d MarkerEditorApp::computeMeshAABB(const dart::dynamics::BodyNode* bn)
{
    if (!bn) return Eigen::Vector3d::Zero();

    // Find MeshShape in body node (usually the second shape node with VisualAspect only)
    for (size_t i = 0; i < bn->getNumShapeNodes(); ++i) {
        auto shapeNode = bn->getShapeNode(i);
        auto shape = shapeNode->getShape();

        if (auto meshShape = std::dynamic_pointer_cast<const MeshShape>(shape)) {
            const aiScene* scene = meshShape->getMesh();
            if (!scene || !scene->mRootNode) continue;

            Eigen::Vector3d scale = meshShape->getScale();

            // Get the shape node's relative transform (mesh is in world coords,
            // this transform converts to body-local frame)
            Eigen::Isometry3d shapeTransform = shapeNode->getRelativeTransform();

            // Initialize min/max
            Eigen::Vector3d minPt(DBL_MAX, DBL_MAX, DBL_MAX);
            Eigen::Vector3d maxPt(-DBL_MAX, -DBL_MAX, -DBL_MAX);

            // Recursively traverse all meshes in scene
            processNodeAABB(scene, scene->mRootNode, scale, shapeTransform, minPt, maxPt);

            // Return AABB size
            if (minPt[0] < DBL_MAX) {
                LOG_INFO("[MarkerEditor] AABB for " << bn->getName()
                         << ": min=[" << minPt.transpose() << "] max=[" << maxPt.transpose()
                         << "] size=[" << (maxPt - minPt).transpose() << "]");
                return maxPt - minPt;
            }
        }
    }
    return Eigen::Vector3d::Zero();
}

void MarkerEditorApp::fitBoxToMesh(size_t bodyNodeIndex, bool clearUndoStack)
{
    if (!mCharacter) return;

    auto skel = mCharacter->getSkeleton();
    if (!skel || bodyNodeIndex >= skel->getNumBodyNodes()) return;

    auto bn = skel->getBodyNode(bodyNodeIndex);
    Eigen::Vector3d aabbSize = computeMeshAABB(bn);

    if (aabbSize.isZero() || aabbSize.norm() < 1e-6) {
        LOG_WARN("[MarkerEditor] No mesh found or empty AABB for body node: " << bn->getName());
        return;
    }

    // Clear undo stack if this is a new standalone operation
    if (clearUndoStack) {
        mFitUndoStack.clear();
    }

    // Find existing BoxShape and replace it
    for (size_t i = 0; i < bn->getNumShapeNodes(); ++i) {
        auto shapeNode = bn->getShapeNode(i);
        if (auto box = std::dynamic_pointer_cast<BoxShape>(shapeNode->getShape())) {
            // Save old size for undo
            mFitUndoStack.push_back({bodyNodeIndex, box->getSize()});

            // Create new BoxShape with AABB size (DART BoxShape is immutable)
            auto newBox = std::make_shared<BoxShape>(aabbSize);
            shapeNode->setShape(newBox);
            LOG_INFO("[MarkerEditor] Fitted box to mesh AABB: "
                     << aabbSize[0] << " x " << aabbSize[1] << " x " << aabbSize[2]
                     << " for " << bn->getName());
            return;
        }
    }

    LOG_WARN("[MarkerEditor] No BoxShape found to replace for: " << bn->getName());
}

void MarkerEditorApp::fitVisibleBoxesToMesh()
{
    if (!mCharacter) return;

    auto skel = mCharacter->getSkeleton();
    if (!skel) return;

    // Clear undo stack for this batch operation
    mFitUndoStack.clear();

    int fittedCount = 0;
    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        // Skip if not visible
        if (i >= mBodyNodeVisible.size() || !mBodyNodeVisible[i]) {
            continue;
        }

        auto bn = skel->getBodyNode(i);

        // Check if this body node has both a MeshShape and a BoxShape
        bool hasMesh = false;
        bool hasBox = false;
        for (size_t j = 0; j < bn->getNumShapeNodes(); ++j) {
            auto shape = bn->getShapeNode(j)->getShape();
            if (std::dynamic_pointer_cast<MeshShape>(shape)) hasMesh = true;
            if (std::dynamic_pointer_cast<BoxShape>(shape)) hasBox = true;
        }

        if (hasMesh && hasBox) {
            // Use false to not clear undo stack (we already cleared it)
            fitBoxToMesh(i, false);
            fittedCount++;
        }
    }

    LOG_INFO("[MarkerEditor] Fitted " << fittedCount << " visible boxes to mesh AABBs");
}

void MarkerEditorApp::undoFit()
{
    if (mFitUndoStack.empty()) return;
    if (!mCharacter) return;

    auto skel = mCharacter->getSkeleton();
    if (!skel) return;

    for (const auto& state : mFitUndoStack) {
        if (state.bodyNodeIndex >= skel->getNumBodyNodes()) continue;

        auto bn = skel->getBodyNode(state.bodyNodeIndex);

        // Find BoxShape and restore old size
        for (size_t i = 0; i < bn->getNumShapeNodes(); ++i) {
            auto shapeNode = bn->getShapeNode(i);
            if (std::dynamic_pointer_cast<BoxShape>(shapeNode->getShape())) {
                auto restoredBox = std::make_shared<BoxShape>(state.oldSize);
                shapeNode->setShape(restoredBox);
                LOG_INFO("[MarkerEditor] Restored box size for " << bn->getName());
                break;
            }
        }
    }

    mFitUndoStack.clear();
}

bool MarkerEditorApp::canUndoFit() const
{
    return !mFitUndoStack.empty();
}

void MarkerEditorApp::exportSkeleton(const std::string& path)
{
    if (!mCharacter) {
        LOG_ERROR("[MarkerEditor] No character loaded for export");
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) {
        LOG_ERROR("[MarkerEditor] No skeleton found");
        return;
    }

    // Resolve path
    std::string resolved_path = path;
    if (path.find("@data/") == 0) {
        resolved_path = PMuscle::URIResolver::getInstance().resolve(path);
    }

    std::ofstream ofs(resolved_path);
    if (!ofs.is_open()) {
        LOG_ERROR("[MarkerEditor] Failed to open file: " << resolved_path);
        return;
    }

    // Save current skeleton state
    Eigen::VectorXd saved_positions = skel->getPositions();

    // Move to zero pose
    skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Write YAML header
    ofs << "metadata:" << std::endl;
    ofs << "  generator: \"MarkerEditorApp\"" << std::endl;
    ofs << "  version: v1" << std::endl;
    ofs << std::endl;

    ofs << "skeleton:" << std::endl;
    ofs << "  name: \"" << skel->getName() << "\"" << std::endl;
    ofs << "  nodes:" << std::endl;

    // Write each body node
    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        auto bn = skel->getBodyNode(i);
        auto parent = bn->getParentBodyNode();
        std::string parentName = parent ? parent->getName() : "None";

        ofs << "    - {name: " << bn->getName()
            << ", parent: " << parentName;

        // Body properties
        if (bn->getNumShapeNodes() > 0) {
            auto shapeNode = bn->getShapeNode(0);
            auto shape = shapeNode->getShape();

            std::string shapeType = "Box";
            Eigen::Vector3d size(0.1, 0.1, 0.1);

            if (auto box = std::dynamic_pointer_cast<BoxShape>(shape)) {
                shapeType = "Box";
                size = box->getSize();
            } else if (auto capsule = std::dynamic_pointer_cast<CapsuleShape>(shape)) {
                shapeType = "Capsule";
                size = Eigen::Vector3d(capsule->getRadius(), capsule->getHeight(), capsule->getRadius());
            } else if (auto sphere = std::dynamic_pointer_cast<SphereShape>(shape)) {
                shapeType = "Sphere";
                double r = sphere->getRadius();
                size = Eigen::Vector3d(r, r, r);
            }

            double mass = bn->getMass();

            ofs << ", body: {type: " << shapeType
                << ", mass: " << std::fixed << std::setprecision(1) << mass
                << ", size: [" << std::fixed << std::setprecision(4)
                << size[0] << ", " << size[1] << ", " << size[2] << "]";

            ofs << "}";
        }

        ofs << "}" << std::endl;
    }

    ofs.close();

    // Restore skeleton state
    skel->setPositions(saved_positions);

    LOG_INFO("[MarkerEditor] Exported skeleton to: " << resolved_path);
}

void MarkerEditorApp::alignMarkerAxis(int axis)
{
    if (!mCharacter) return;

    auto& markers = mCharacter->getMarkersForEdit();
    if (mSelectedMarkerIndex < 0 || mReferenceMarkerIndex < 0) return;
    if (mSelectedMarkerIndex >= static_cast<int>(markers.size()) ||
        mReferenceMarkerIndex >= static_cast<int>(markers.size())) return;

    auto& marker = markers[mSelectedMarkerIndex];
    const auto& refMarker = markers[mReferenceMarkerIndex];

    // Get reference marker's global position
    Eigen::Vector3d refGlobalPos = refMarker.getGlobalPos();
    Eigen::Vector3d currentGlobalPos = marker.getGlobalPos();

    // Copy the specified axis from reference
    currentGlobalPos[axis] = refGlobalPos[axis];

    // Recompute offset for the new global position
    if (marker.bodyNode) {
        auto* shapeNode = marker.bodyNode->getShapeNodeWith<dart::dynamics::VisualAspect>(0);
        if (shapeNode) {
            const auto* boxShape = dynamic_cast<const dart::dynamics::BoxShape*>(shapeNode->getShape().get());
            if (boxShape) {
                Eigen::Vector3d size = boxShape->getSize();
                Eigen::Vector3d localPos = marker.bodyNode->getTransform().inverse() * currentGlobalPos;
                marker.offset = Eigen::Vector3d(
                    (std::abs(size[0]) > 1e-6) ? 2.0 * localPos[0] / size[0] : 0.0,
                    (std::abs(size[1]) > 1e-6) ? 2.0 * localPos[1] / size[1] : 0.0,
                    (std::abs(size[2]) > 1e-6) ? 2.0 * localPos[2] / size[2] : 0.0
                );
            }
        }
    }
}

void MarkerEditorApp::mirrorMarkerAxis(int axis)
{
    if (!mCharacter) return;

    auto& markers = mCharacter->getMarkersForEdit();
    if (mSelectedMarkerIndex < 0 || mReferenceMarkerIndex < 0) return;
    if (mSelectedMarkerIndex >= static_cast<int>(markers.size()) ||
        mReferenceMarkerIndex >= static_cast<int>(markers.size())) return;

    auto& marker = markers[mSelectedMarkerIndex];
    const auto& refMarker = markers[mReferenceMarkerIndex];

    // Get positions
    Eigen::Vector3d refGlobalPos = refMarker.getGlobalPos();
    Eigen::Vector3d currentGlobalPos = marker.getGlobalPos();

    // Mirror: only negate the specified axis coordinate
    // axis=0: YZ plane → sel_x = -ref_x (keep sel_y, sel_z)
    // axis=1: ZX plane → sel_y = -ref_y (keep sel_x, sel_z)
    // axis=2: XY plane → sel_z = -ref_z (keep sel_x, sel_y)
    currentGlobalPos[axis] = -refGlobalPos[axis];

    // Recompute offset for the new global position
    if (marker.bodyNode) {
        auto* shapeNode = marker.bodyNode->getShapeNodeWith<dart::dynamics::VisualAspect>(0);
        if (shapeNode) {
            const auto* boxShape = dynamic_cast<const dart::dynamics::BoxShape*>(shapeNode->getShape().get());
            if (boxShape) {
                Eigen::Vector3d size = boxShape->getSize();
                Eigen::Vector3d localPos = marker.bodyNode->getTransform().inverse() * currentGlobalPos;
                marker.offset = Eigen::Vector3d(
                    (std::abs(size[0]) > 1e-6) ? 2.0 * localPos[0] / size[0] : 0.0,
                    (std::abs(size[1]) > 1e-6) ? 2.0 * localPos[1] / size[1] : 0.0,
                    (std::abs(size[2]) > 1e-6) ? 2.0 * localPos[2] / size[2] : 0.0
                );
            }
        }
    }
}

void MarkerEditorApp::alignCameraToPlane(int plane)
{
    // Align camera to view a specific plane
    // 1 = XY plane (view from +Z, looking at -Z, up is +Y)
    // 2 = YZ plane (view from +X, looking at -X, up is +Y)
    // 3 = ZX plane (view from +Y, looking at -Y, up is +Z)

    Eigen::Quaterniond quat;
    const char* planeName = "";

    switch (plane) {
        case 1: // XY plane - view from +Z axis
            // Looking along -Z, up is +Y → identity rotation (default OpenGL view)
            quat = Eigen::Quaterniond::Identity();
            planeName = "XY";
            break;
        case 2: // YZ plane - view from +X axis
            // Rotate -90° around Y axis to look along -X
            quat = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitY());
            planeName = "YZ";
            break;
        case 3: // ZX plane - view from +Y axis (top-down)
            // Rotate +90° around X axis to look along -Y
            quat = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX());
            planeName = "ZX";
            break;
        default:
            LOG_WARN("[MarkerEditor] Invalid plane index: " << plane);
            return;
    }

    mTrackball.setQuaternion(quat);
    LOG_INFO("[MarkerEditor] Camera aligned to " << planeName << " plane");
}
