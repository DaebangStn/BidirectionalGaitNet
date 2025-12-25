#ifndef IMGUI_COMMON_H
#define IMGUI_COMMON_H

#include <vector>
#include <string>
#include <memory>
#include "imgui.h"

class Muscle;

namespace ImGuiCommon {

// Muscle selection widget with filter, select/deselect all, and scrollable list
// Returns true if any selection changed
bool MuscleSelector(
    const char* label,
    const std::vector<std::shared_ptr<Muscle>>& muscles,
    std::vector<bool>& selectionStates,
    char* filterBuffer,
    size_t filterBufferSize,
    float listHeight = 250.0f
);

// Simplified version that manages its own filter buffer
bool MuscleSelector(
    const char* label,
    const std::vector<std::shared_ptr<Muscle>>& muscles,
    std::vector<bool>& selectionStates,
    float listHeight = 250.0f
);

// Generic filterable checklist widget
// Returns true if any selection changed
bool FilterableChecklist(
    const char* label,
    const std::vector<std::string>& items,
    std::vector<bool>& selectionStates,
    char* filterBuffer,
    size_t filterBufferSize,
    float listHeight = 200.0f
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

} // namespace ImGuiCommon

#endif // IMGUI_COMMON_H
