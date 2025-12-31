#ifndef IMGUI_COMMON_H
#define IMGUI_COMMON_H

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <cfloat>  // For FLT_MAX
#include "imgui.h"

class Muscle;

namespace ImGuiCommon {

// Muscle selection widget with filter, select/deselect all, and scrollable list
// Returns true if any selection changed
// Note: selectionStates must be pre-initialized by caller
bool MuscleSelector(
    const char* label,
    const std::vector<Muscle*>& muscles,
    std::vector<bool>& selectionStates,
    char* filterBuffer,
    size_t filterBufferSize,
    float listHeight = 250.0f
);

// Simplified version that manages its own filter buffer
bool MuscleSelector(
    const char* label,
    const std::vector<Muscle*>& muscles,
    std::vector<bool>& selectionStates,
    float listHeight = 250.0f
);

// Generic filterable checklist widget (multi-selection)
// Returns true if any selection changed
bool FilterableChecklist(
    const char* label,
    const std::vector<std::string>& items,
    std::vector<bool>& selectionStates,
    char* filterBuffer,
    size_t filterBufferSize,
    float listHeight = 200.0f
);

// Generic filterable radio list widget (single-selection)
// Returns true if selection changed
// Optional colorFunc: returns color for each item (nullptr for default)
bool FilterableRadioList(
    const char* label,
    const std::vector<std::string>& items,
    int* selectedIdx,
    char* filterBuffer,
    size_t filterBufferSize,
    float listHeight = 200.0f,
    std::function<ImVec4(int idx)> colorFunc = nullptr
);

// Collapsing header with consistent styling
bool StyledCollapsingHeader(const char* label, bool defaultOpen = true);

// Labeled separator
void LabeledSeparator(const char* label);

// Color legend item (colored square + text)
void ColorLegendItem(const ImVec4& color, const char* text);

// Horizontal color bar legend
void ColorBarLegend(
    const char* label,
    float minVal, float maxVal,
    const char* minLabel, const char* maxLabel
);

// ============================================================
// Slider Helpers - Full width with consistent styling
// ============================================================

// Full-width float slider
bool SliderFloatFullWidth(const char* label, float* v, float min, float max,
                          const char* format = "%.2f");

// Full-width int slider
bool SliderIntFullWidth(const char* label, int* v, int min, int max);

// Full-width drag float
bool DragFloatFullWidth(const char* label, float* v, float speed = 0.1f,
                        float min = 0.0f, float max = 0.0f,
                        const char* format = "%.2f");

// ============================================================
// Radio Button Group - Horizontal or vertical layout
// ============================================================

// Returns true if selection changed
bool RadioButtonGroup(const char* id, const std::vector<std::string>& labels,
                      int* selected, bool horizontal = true);

// ============================================================
// Button Row - Multiple buttons in a row
// ============================================================

struct ButtonDef {
    std::string label;
    bool enabled = true;
};

// Returns index of clicked button, or -1 if none clicked
int ButtonRow(const char* id, const std::vector<ButtonDef>& buttons);

// ============================================================
// Status Text - Color-coded status messages
// ============================================================

enum class StatusType { Info, Success, Warning, Error, Disabled };

void StatusText(const char* text, StatusType type = StatusType::Info);

// ============================================================
// Input Text with Button - Common pattern for file paths
// ============================================================

// Returns true if button clicked
bool InputTextWithButton(const char* id, char* buf, size_t size,
                         const char* buttonLabel, float buttonWidth = 60.0f);

// ============================================================
// Drag Float 3 - XYZ triplet editor
// ============================================================

// Returns true if any value changed
bool DragFloat3Labeled(const char* label, float v[3], float speed = 0.01f,
                       float min = -FLT_MAX, float max = FLT_MAX,
                       const char* format = "%.3f");

// ============================================================
// Scrollable List Box - With optional filtering
// ============================================================

// Returns true if selection changed
bool ScrollableListBox(const char* label, const std::vector<std::string>& items,
                       int* selectedIdx, float height = 200.0f,
                       const char* filterText = nullptr);

// ============================================================
// ImPlot Helpers
// ============================================================

// Setup X-axis limits for time-series plots
// If xMin is significant (abs > 1e-6), uses [xMin, 0], otherwise uses [defaultMin, 0]
void SetupPlotXAxis(double xMin, double defaultMin = -1.5);

} // namespace ImGuiCommon

#endif // IMGUI_COMMON_H
