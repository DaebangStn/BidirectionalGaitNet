#include "imgui_common.h"
#include "Muscle.h"
#include <algorithm>
#include <cctype>

namespace ImGuiCommon {

// Static filter buffer for simplified MuscleSelector
static char s_muscleFilterBuffer[256] = "";

bool MuscleSelector(
    const char* label,
    const std::vector<Muscle*>& muscles,
    std::vector<bool>& selectionStates,
    char* filterBuffer,
    size_t filterBufferSize,
    float listHeight)
{
    bool changed = false;

    // Initialize selection states if needed
    if (selectionStates.size() != muscles.size()) {
        selectionStates.resize(muscles.size(), true);
        changed = true;
    }

    // Count selected
    int selectedCount = 0;
    for (bool sel : selectionStates) {
        if (sel) selectedCount++;
    }

    ImGui::Text("Selected: %d / %zu", selectedCount, muscles.size());

    // Filter input
    ImGui::PushItemWidth(-1);
    ImGui::InputTextWithHint("##filter", "Filter muscles...", filterBuffer, filterBufferSize);
    ImGui::PopItemWidth();

    // Build filter string (lowercase)
    std::string filterStr(filterBuffer);
    std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Find matching indices
    std::vector<int> filteredIndices;
    for (size_t i = 0; i < muscles.size(); i++) {
        std::string muscleName = muscles[i]->name;
        std::transform(muscleName.begin(), muscleName.end(), muscleName.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        if (filterStr.empty() || muscleName.find(filterStr) != std::string::npos) {
            filteredIndices.push_back(static_cast<int>(i));
        }
    }

    // Select/Deselect buttons
    if (ImGui::Button("Select All")) {
        for (int idx : filteredIndices) {
            if (!selectionStates[idx]) {
                selectionStates[idx] = true;
                changed = true;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Deselect All")) {
        for (int idx : filteredIndices) {
            if (selectionStates[idx]) {
                selectionStates[idx] = false;
                changed = true;
            }
        }
    }
    ImGui::SameLine();
    ImGui::Text("(%zu shown)", filteredIndices.size());

    // Scrollable muscle list
    ImGui::BeginChild(label, ImVec2(0, listHeight), true);
    for (int idx : filteredIndices) {
        bool selected = selectionStates[idx];
        if (ImGui::Checkbox(muscles[idx]->name.c_str(), &selected)) {
            selectionStates[idx] = selected;
            changed = true;
        }
    }
    ImGui::EndChild();

    return changed;
}

bool MuscleSelector(
    const char* label,
    const std::vector<Muscle*>& muscles,
    std::vector<bool>& selectionStates,
    float listHeight)
{
    return MuscleSelector(label, muscles, selectionStates,
                          s_muscleFilterBuffer, sizeof(s_muscleFilterBuffer), listHeight);
}

bool FilterableChecklist(
    const char* label,
    const std::vector<std::string>& items,
    std::vector<bool>& selectionStates,
    char* filterBuffer,
    size_t filterBufferSize,
    float listHeight)
{
    bool changed = false;

    // Initialize selection states if needed
    if (selectionStates.size() != items.size()) {
        selectionStates.resize(items.size(), true);
        changed = true;
    }

    // Count selected
    int selectedCount = 0;
    for (bool sel : selectionStates) {
        if (sel) selectedCount++;
    }

    ImGui::Text("Selected: %d / %zu", selectedCount, items.size());

    // Filter input
    ImGui::PushItemWidth(-1);
    ImGui::InputTextWithHint("##filter", "Filter...", filterBuffer, filterBufferSize);
    ImGui::PopItemWidth();

    // Build filter string (lowercase)
    std::string filterStr(filterBuffer);
    std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Find matching indices
    std::vector<int> filteredIndices;
    for (size_t i = 0; i < items.size(); i++) {
        std::string itemName = items[i];
        std::transform(itemName.begin(), itemName.end(), itemName.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        if (filterStr.empty() || itemName.find(filterStr) != std::string::npos) {
            filteredIndices.push_back(static_cast<int>(i));
        }
    }

    // Select/Deselect buttons
    if (ImGui::Button("Select All")) {
        for (int idx : filteredIndices) {
            if (!selectionStates[idx]) {
                selectionStates[idx] = true;
                changed = true;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Deselect All")) {
        for (int idx : filteredIndices) {
            if (selectionStates[idx]) {
                selectionStates[idx] = false;
                changed = true;
            }
        }
    }

    // Scrollable list
    ImGui::BeginChild(label, ImVec2(0, listHeight), true);
    for (int idx : filteredIndices) {
        bool selected = selectionStates[idx];
        if (ImGui::Checkbox(items[idx].c_str(), &selected)) {
            selectionStates[idx] = selected;
            changed = true;
        }
    }
    ImGui::EndChild();

    return changed;
}

bool StyledCollapsingHeader(const char* label, bool defaultOpen)
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_None;
    if (defaultOpen) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    return ImGui::CollapsingHeader(label, flags);
}

void LabeledSeparator(const char* label)
{
    ImGui::Spacing();
    ImGui::TextDisabled("%s", label);
    ImGui::Separator();
}

void ColorLegendItem(const ImVec4& color, const char* text)
{
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    float size = ImGui::GetTextLineHeight();
    drawList->AddRectFilled(p, ImVec2(p.x + size, p.y + size),
                            ImGui::ColorConvertFloat4ToU32(color));
    drawList->AddRect(p, ImVec2(p.x + size, p.y + size),
                      ImGui::ColorConvertFloat4ToU32(ImVec4(0.5f, 0.5f, 0.5f, 1.0f)));

    ImGui::Dummy(ImVec2(size + 4, size));
    ImGui::SameLine();
    ImGui::Text("%s", text);
}

void ColorBarLegend(
    const char* label,
    float minVal, float maxVal,
    const char* minLabel, const char* maxLabel)
{
    ImGui::Text("%s", label);

    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    float width = ImGui::GetContentRegionAvail().x;
    float height = 15.0f;

    // Draw gradient bar
    int segments = 50;
    float segWidth = width / segments;
    for (int i = 0; i < segments; i++) {
        float t = static_cast<float>(i) / (segments - 1);
        ImVec4 color;

        // Blue -> Cyan -> Green -> Yellow -> Red
        if (t < 0.25f) {
            float s = t / 0.25f;
            color = ImVec4(0.0f, s, 1.0f, 1.0f);
        } else if (t < 0.5f) {
            float s = (t - 0.25f) / 0.25f;
            color = ImVec4(0.0f, 1.0f, 1.0f - s, 1.0f);
        } else if (t < 0.75f) {
            float s = (t - 0.5f) / 0.25f;
            color = ImVec4(s, 1.0f, 0.0f, 1.0f);
        } else {
            float s = (t - 0.75f) / 0.25f;
            color = ImVec4(1.0f, 1.0f - s, 0.0f, 1.0f);
        }

        drawList->AddRectFilled(
            ImVec2(p.x + i * segWidth, p.y),
            ImVec2(p.x + (i + 1) * segWidth, p.y + height),
            ImGui::ColorConvertFloat4ToU32(color)
        );
    }

    ImGui::Dummy(ImVec2(width, height));

    // Labels
    ImGui::Text("%.2f %s", minVal, minLabel);
    ImGui::SameLine(width - ImGui::CalcTextSize(maxLabel).x - 50);
    ImGui::Text("%.2f %s", maxVal, maxLabel);
}

// ============================================================
// Slider Helpers
// ============================================================

bool SliderFloatFullWidth(const char* label, float* v, float min, float max,
                          const char* format)
{
    ImGui::PushItemWidth(-1);
    bool changed = ImGui::SliderFloat(label, v, min, max, format);
    ImGui::PopItemWidth();
    return changed;
}

bool SliderIntFullWidth(const char* label, int* v, int min, int max)
{
    ImGui::PushItemWidth(-1);
    bool changed = ImGui::SliderInt(label, v, min, max);
    ImGui::PopItemWidth();
    return changed;
}

bool DragFloatFullWidth(const char* label, float* v, float speed,
                        float min, float max, const char* format)
{
    ImGui::PushItemWidth(-1);
    bool changed = ImGui::DragFloat(label, v, speed, min, max, format);
    ImGui::PopItemWidth();
    return changed;
}

// ============================================================
// Radio Button Group
// ============================================================

bool RadioButtonGroup(const char* id, const std::vector<std::string>& labels,
                      int* selected, bool horizontal)
{
    bool changed = false;
    ImGui::PushID(id);

    for (size_t i = 0; i < labels.size(); i++) {
        if (i > 0 && horizontal) {
            ImGui::SameLine();
        }
        if (ImGui::RadioButton(labels[i].c_str(), *selected == static_cast<int>(i))) {
            *selected = static_cast<int>(i);
            changed = true;
        }
    }

    ImGui::PopID();
    return changed;
}

// ============================================================
// Button Row
// ============================================================

int ButtonRow(const char* id, const std::vector<ButtonDef>& buttons)
{
    int clicked = -1;
    ImGui::PushID(id);

    for (size_t i = 0; i < buttons.size(); i++) {
        if (i > 0) {
            ImGui::SameLine();
        }

        if (!buttons[i].enabled) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button(buttons[i].label.c_str())) {
            clicked = static_cast<int>(i);
        }

        if (!buttons[i].enabled) {
            ImGui::EndDisabled();
        }
    }

    ImGui::PopID();
    return clicked;
}

// ============================================================
// Status Text
// ============================================================

void StatusText(const char* text, StatusType type)
{
    ImVec4 color;
    switch (type) {
        case StatusType::Info:
            color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);  // Light gray
            break;
        case StatusType::Success:
            color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green
            break;
        case StatusType::Warning:
            color = ImVec4(1.0f, 0.7f, 0.2f, 1.0f);  // Orange
            break;
        case StatusType::Error:
            color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red
            break;
        case StatusType::Disabled:
            color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Dark gray
            break;
    }
    ImGui::TextColored(color, "%s", text);
}

// ============================================================
// Input Text with Button
// ============================================================

bool InputTextWithButton(const char* id, char* buf, size_t size,
                         const char* buttonLabel, float buttonWidth)
{
    ImGui::PushID(id);

    // Calculate input width to leave space for button
    float inputWidth = ImGui::GetContentRegionAvail().x - buttonWidth - ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth(inputWidth);
    ImGui::InputText("##input", buf, size);

    ImGui::SameLine();
    bool clicked = ImGui::Button(buttonLabel, ImVec2(buttonWidth, 0));

    ImGui::PopID();
    return clicked;
}

// ============================================================
// Drag Float 3 Labeled
// ============================================================

bool DragFloat3Labeled(const char* label, float v[3], float speed,
                       float min, float max, const char* format)
{
    bool changed = false;
    ImGui::PushID(label);

    ImGui::Text("%s", label);

    float width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3.0f;

    ImGui::SetNextItemWidth(width);
    if (ImGui::DragFloat("##X", &v[0], speed, min, max, format)) changed = true;
    ImGui::SameLine();

    ImGui::SetNextItemWidth(width);
    if (ImGui::DragFloat("##Y", &v[1], speed, min, max, format)) changed = true;
    ImGui::SameLine();

    ImGui::SetNextItemWidth(width);
    if (ImGui::DragFloat("##Z", &v[2], speed, min, max, format)) changed = true;

    ImGui::PopID();
    return changed;
}

// ============================================================
// Scrollable List Box
// ============================================================

bool ScrollableListBox(const char* label, const std::vector<std::string>& items,
                       int* selectedIdx, float height, const char* filterText)
{
    bool changed = false;

    // Build filter string if provided
    std::string filterStr;
    if (filterText != nullptr && filterText[0] != '\0') {
        filterStr = filterText;
        std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(),
                       [](unsigned char c) { return std::tolower(c); });
    }

    ImGui::BeginChild(label, ImVec2(0, height), true);

    for (size_t i = 0; i < items.size(); i++) {
        // Apply filter if provided
        if (!filterStr.empty()) {
            std::string itemLower = items[i];
            std::transform(itemLower.begin(), itemLower.end(), itemLower.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (itemLower.find(filterStr) == std::string::npos) {
                continue;  // Skip non-matching items
            }
        }

        bool isSelected = (*selectedIdx == static_cast<int>(i));
        if (ImGui::Selectable(items[i].c_str(), isSelected)) {
            *selectedIdx = static_cast<int>(i);
            changed = true;
        }

        // Set initial focus when opening
        if (isSelected) {
            ImGui::SetItemDefaultFocus();
        }
    }

    ImGui::EndChild();
    return changed;
}

} // namespace ImGuiCommon
