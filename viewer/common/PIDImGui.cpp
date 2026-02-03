#include "PIDImGui.h"
#include "Log.h"
#include <rm/rm.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
#include <imgui.h>

namespace PIDNav {

// ============================================================================
// PIDSelectionState Implementation
// ============================================================================

std::string PIDSelectionState::getSelectedPID() const {
    if (selectedPID >= 0 && selectedPID < static_cast<int>(pidList.size())) {
        return pidList[selectedPID];
    }
    return "";
}

std::string PIDSelectionState::getVisitDir() const {
    if (selectedPID >= 0 && selectedPID < static_cast<int>(pidVisits.size())) {
        const auto& visits = pidVisits[selectedPID];
        if (selectedVisit >= 0 && selectedVisit < static_cast<int>(visits.size())) {
            return visits[selectedVisit];
        }
    }
    return "pre";  // Default fallback
}

std::string PIDSelectionState::getPrePostDir() const {
    // Legacy compatibility: map visit to pre/post
    std::string visit = getVisitDir();
    return (visit == "pre") ? "pre" : "post";
}

// ============================================================================
// FileTypeSection - Internal structure for multi-file-type support
// ============================================================================

struct FileTypeSection {
    std::string label;
    std::unique_ptr<FileFilter> filter;
    FileSelectionCallback onSelect;
    FileDeleteCallback onDelete;
    std::vector<std::string> files;
    int selectedFile = -1;
    char filterBuf[64] = "";

    FileTypeSection() = default;
    FileTypeSection(FileTypeSection&&) = default;
    FileTypeSection& operator=(FileTypeSection&&) = default;
};

// ============================================================================
// PIDNavigator::Impl - PIMPL Implementation
// ============================================================================

struct PIDNavigator::Impl {
    // Dependencies (non-owning)
    rm::ResourceManager* resourceManager;

    // State
    PIDSelectionState state;

    // Multi-file-type sections
    std::vector<FileTypeSection> fileTypeSections;

    // Filter buffers for ImGui
    char pidFilterBuf[64] = "";

    // Callbacks
    PIDChangeCallback pidChangeCallback;
    VisitChangeCallback visitChangeCallback;

    // Empty vector for getFiles() when no sections exist
    static const std::vector<std::string> emptyFileList;

    explicit Impl(rm::ResourceManager* rm)
        : resourceManager(rm) {
    }
};

const std::vector<std::string> PIDNavigator::Impl::emptyFileList;

// ============================================================================
// PIDNavigator Public Interface
// ============================================================================

PIDNavigator::PIDNavigator(rm::ResourceManager* rm)
    : pImpl(std::make_unique<Impl>(rm)) {
}

PIDNavigator::PIDNavigator(rm::ResourceManager* rm,
                           std::unique_ptr<FileFilter> filter)
    : pImpl(std::make_unique<Impl>(rm)) {
    // Legacy constructor: register as the primary file type with default label
    if (filter) {
        FileTypeConfig config;
        config.label = "Files";
        config.filter = std::move(filter);
        registerFileType(std::move(config));
    }
}

PIDNavigator::~PIDNavigator() = default;

PIDNavigator::PIDNavigator(PIDNavigator&&) noexcept = default;
PIDNavigator& PIDNavigator::operator=(PIDNavigator&&) noexcept = default;

void PIDNavigator::registerFileType(FileTypeConfig config) {
    FileTypeSection section;
    section.label = std::move(config.label);
    section.filter = std::move(config.filter);
    section.onSelect = std::move(config.onSelect);
    section.onDelete = std::move(config.onDelete);

    // Initialize default file filter from strategy
    if (section.filter) {
        auto defaultFilter = section.filter->getDefaultFilter();
        if (!defaultFilter.empty() && defaultFilter.size() < sizeof(section.filterBuf)) {
            std::strcpy(section.filterBuf, defaultFilter.c_str());
        }
    }

    pImpl->fileTypeSections.push_back(std::move(section));
}

void PIDNavigator::scanPIDs() {
    pImpl->state.pidList.clear();
    pImpl->state.pidNames.clear();
    pImpl->state.pidGMFCS.clear();
    pImpl->state.pidVisits.clear();
    pImpl->state.selectedPID = -1;
    pImpl->state.selectedVisit = 0;

    // Clear all file sections
    for (auto& section : pImpl->fileTypeSections) {
        section.files.clear();
        section.selectedFile = -1;
    }

    if (!pImpl->resourceManager) return;

    try {
        auto entries = pImpl->resourceManager->list("@pid:");
        for (const auto& entry : entries) {
            // PIDs must be numeric only - skip non-numeric entries like scripts
            bool isNumeric = !entry.empty() && std::all_of(entry.begin(), entry.end(), ::isdigit);
            if (isNumeric) {
                pImpl->state.pidList.push_back(entry);
            }
        }
        std::sort(pImpl->state.pidList.begin(), pImpl->state.pidList.end());

        pImpl->state.pidNames.resize(pImpl->state.pidList.size());
        pImpl->state.pidGMFCS.resize(pImpl->state.pidList.size());
        pImpl->state.pidVisits.resize(pImpl->state.pidList.size());

        for (size_t i = 0; i < pImpl->state.pidList.size(); ++i) {
            const auto& pid = pImpl->state.pidList[i];

            // Fetch name
            try {
                auto h = pImpl->resourceManager->fetch("@pid:" + pid + "/name");
                pImpl->state.pidNames[i] = h.as_string();
            } catch (...) {
                pImpl->state.pidNames[i] = "";
            }

            // Fetch GMFCS
            try {
                auto h = pImpl->resourceManager->fetch("@pid:" + pid + "/gmfcs");
                pImpl->state.pidGMFCS[i] = h.as_string();
            } catch (...) {
                pImpl->state.pidGMFCS[i] = "";
            }

            // Initialize with empty visits - will be scanned when PID is selected
        }
    } catch (const rm::RMError&) {
        // Silently handle errors - list will remain empty
    }
}

void PIDNavigator::scanFiles(const std::string& pid, const std::string& visit) {
    if (!pImpl->resourceManager || pid.empty()) return;

    // Scan files for all registered file type sections
    for (auto& section : pImpl->fileTypeSections) {
        section.files.clear();
        section.selectedFile = -1;

        if (!section.filter) continue;

        std::string subdirectory = section.filter->getSubdirectory();

        // Build pattern using visit-based path: @pid:{pid}/{visit}/{subdirectory}
        std::string pattern = "@pid:" + pid + "/" + visit;
        if (!subdirectory.empty()) {
            pattern += "/" + subdirectory;
        }

        try {
            auto files = pImpl->resourceManager->list(pattern);
            for (const auto& file : files) {
                if (section.filter->matches(file)) {
                    // Extract just the filename (list() returns filenames, not paths)
                    std::filesystem::path p(file);
                    section.files.push_back(p.filename().string());
                }
            }
            std::sort(section.files.begin(), section.files.end());
        } catch (const rm::RMError&) {
            // Silently handle errors - list will remain empty
        }
    }
}


bool PIDNavigator::navigateTo(const std::string& pid, const std::string& visit) {
    if (!pImpl->resourceManager || pid.empty()) return false;

    // Find the PID in the list
    auto it = std::find(pImpl->state.pidList.begin(), pImpl->state.pidList.end(), pid);
    if (it == pImpl->state.pidList.end()) {
        return false;  // PID not found
    }

    int pidIndex = static_cast<int>(std::distance(pImpl->state.pidList.begin(), it));

    // Select the PID
    pImpl->state.selectedPID = pidIndex;

    // Scan available visits for this PID (lazy loading)
    auto& visits = pImpl->state.pidVisits[pidIndex];
    if (visits.empty()) {
        static const std::vector<std::string> VISIT_ORDER = {"pre", "op1", "op2"};
        static const std::set<std::string> VALID_VISITS = {"pre", "op1", "op2"};
        try {
            auto entries = pImpl->resourceManager->list("@pid:" + pid);
            std::set<std::string> foundVisits;
            for (const auto& entry : entries) {
                if (VALID_VISITS.count(entry)) {
                    foundVisits.insert(entry);
                }
            }
            for (const auto& v : VISIT_ORDER) {
                if (foundVisits.count(v)) {
                    visits.push_back(v);
                }
            }
        } catch (...) {}
        if (visits.empty()) visits.push_back("pre");  // Fallback
    }

    // Find and select the visit
    auto visitIt = std::find(visits.begin(), visits.end(), visit);
    if (visitIt != visits.end()) {
        pImpl->state.selectedVisit = static_cast<int>(std::distance(visits.begin(), visitIt));
    } else {
        pImpl->state.selectedVisit = 0;  // Default to first visit
    }
    pImpl->state.preOp = (pImpl->state.getVisitDir() == "pre");

    // Scan files for the selected PID/visit
    scanFiles(pid, pImpl->state.getVisitDir());

    // Invoke PID change callback if set
    if (pImpl->pidChangeCallback) {
        pImpl->pidChangeCallback(pid);
    }

    return true;
}

const PIDSelectionState& PIDNavigator::getState() const {
    return pImpl->state;
}

const std::vector<std::string>& PIDNavigator::getFiles() const {
    // Return files from the first registered file type (legacy compatibility)
    if (!pImpl->fileTypeSections.empty()) {
        return pImpl->fileTypeSections[0].files;
    }
    return Impl::emptyFileList;
}

const std::vector<std::string>& PIDNavigator::getFiles(const std::string& label) const {
    for (const auto& section : pImpl->fileTypeSections) {
        if (section.label == label) {
            return section.files;
        }
    }
    return Impl::emptyFileList;
}

void PIDNavigator::setFileSelectionCallback(FileSelectionCallback callback) {
    // Set callback on the first registered file type (legacy compatibility)
    if (!pImpl->fileTypeSections.empty()) {
        pImpl->fileTypeSections[0].onSelect = std::move(callback);
    }
}

void PIDNavigator::setPIDChangeCallback(PIDChangeCallback callback) {
    pImpl->pidChangeCallback = std::move(callback);
}

void PIDNavigator::setVisitChangeCallback(VisitChangeCallback callback) {
    pImpl->visitChangeCallback = std::move(callback);
}

void PIDNavigator::renderUI(const char* title,
                            float pidListHeight,
                            float fileSectionHeight,
                            bool defaultOpen) {
    if (!pImpl->resourceManager) return;

    // If title provided, wrap in collapsing header
    if (title) {
        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0;
        if (!ImGui::CollapsingHeader(title, flags)) {
            return;
        }
    }

    // ========== PID Filter + Refresh ==========
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##PIDFilter", pImpl->pidFilterBuf, sizeof(pImpl->pidFilterBuf));
    ImGui::SameLine();
    if (ImGui::Button("Refresh##PID")) {
        scanPIDs();
    }
    ImGui::SameLine();
    ImGui::Text("%zu PIDs", pImpl->state.pidList.size());

    // ========== PID ListBox ==========
    if (ImGui::BeginListBox("##PIDList", ImVec2(-1, pidListHeight))) {
        for (int i = 0; i < static_cast<int>(pImpl->state.pidList.size()); ++i) {
            const auto& pid = pImpl->state.pidList[i];
            const std::string& name = pImpl->state.pidNames[i];
            const std::string& gmfcs = pImpl->state.pidGMFCS[i];

            std::string displayStr = pid;
            if (!name.empty() && !gmfcs.empty())
                displayStr = pid + " (" + name + ", " + gmfcs + ")";
            else if (!name.empty())
                displayStr = pid + " (" + name + ")";
            else if (!gmfcs.empty())
                displayStr = pid + " (" + gmfcs + ")";

            // Filter
            if (pImpl->pidFilterBuf[0] &&
                pid.find(pImpl->pidFilterBuf) == std::string::npos &&
                name.find(pImpl->pidFilterBuf) == std::string::npos &&
                gmfcs.find(pImpl->pidFilterBuf) == std::string::npos) {
                continue;
            }

            if (ImGui::Selectable(displayStr.c_str(), i == pImpl->state.selectedPID)) {
                if (i != pImpl->state.selectedPID) {
                    pImpl->state.selectedPID = i;

                    // Scan available visits for this PID (lazy loading - single list call)
                    auto& visits = pImpl->state.pidVisits[i];
                    if (visits.empty()) {
                        // Order matters: pre first, then op1, op2
                        static const std::vector<std::string> VISIT_ORDER = {"pre", "op1", "op2"};
                        static const std::set<std::string> VALID_VISITS = {"pre", "op1", "op2"};
                        try {
                            // List PID directory once and filter for valid visits
                            auto entries = pImpl->resourceManager->list("@pid:" + pid);
                            std::set<std::string> foundVisits;
                            for (const auto& entry : entries) {
                                if (VALID_VISITS.count(entry)) {
                                    foundVisits.insert(entry);
                                }
                            }
                            // Add in correct order: pre, op1, op2
                            for (const auto& v : VISIT_ORDER) {
                                if (foundVisits.count(v)) {
                                    visits.push_back(v);
                                }
                            }
                        } catch (...) {}
                        if (visits.empty()) visits.push_back("pre");  // Fallback
                    }

                    pImpl->state.selectedVisit = 0;  // Reset to first visit
                    pImpl->state.preOp = (pImpl->state.getVisitDir() == "pre");
                    try {
                        scanFiles(pid, pImpl->state.getVisitDir());
                    } catch (const std::exception& e) {
                        LOG_WARN("[PIDNavigator] Failed to scan PID files: " << e.what());
                        for (auto& section : pImpl->fileTypeSections) {
                            section.files.clear();
                        }
                    }

                    // Invoke PID change callback if set
                    if (pImpl->pidChangeCallback) {
                        try {
                            pImpl->pidChangeCallback(pid);
                        } catch (const std::exception& e) {
                            LOG_WARN("[PIDNavigator] PID change callback error: " << e.what());
                        }
                    }
                }
            }
        }
        ImGui::EndListBox();
    }

    // ========== Visit selection (pre/op1/op2) ==========
    if (pImpl->state.selectedPID >= 0) {
        const auto& visits = pImpl->state.pidVisits[pImpl->state.selectedPID];
        for (int v = 0; v < static_cast<int>(visits.size()); ++v) {
            if (v > 0) ImGui::SameLine();
            if (ImGui::RadioButton(visits[v].c_str(), pImpl->state.selectedVisit == v)) {
                if (pImpl->state.selectedVisit != v) {
                    pImpl->state.selectedVisit = v;
                    pImpl->state.preOp = (visits[v] == "pre");  // Legacy compatibility
                    std::string pid = pImpl->state.pidList[pImpl->state.selectedPID];
                    try {
                        scanFiles(pid, visits[v]);
                    } catch (const std::exception& e) {
                        LOG_WARN("[PIDNavigator] Failed to scan visit: " << e.what());
                        for (auto& section : pImpl->fileTypeSections) {
                            section.files.clear();
                        }
                    }

                    // Invoke visit change callback if set
                    if (pImpl->visitChangeCallback) {
                        try {
                            pImpl->visitChangeCallback(pid, visits[v]);
                        } catch (const std::exception& e) {
                            LOG_WARN("[PIDNavigator] Visit change callback error: " << e.what());
                        }
                    }
                }
            }
        }
    }

    // ========== File sections (only if fileSectionHeight > 0 and PID selected) ==========
    if (fileSectionHeight > 0 && pImpl->state.selectedPID >= 0) {
        for (size_t sectionIdx = 0; sectionIdx < pImpl->fileTypeSections.size(); ++sectionIdx) {
            auto& section = pImpl->fileTypeSections[sectionIdx];

            ImGui::Separator();

            // Section header with file count
            std::string headerLabel = section.label + " (" + std::to_string(section.files.size()) + ")";
            if (ImGui::TreeNodeEx(headerLabel.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                // Filter input
                std::string filterId = "##Filter_" + std::to_string(sectionIdx);
                ImGui::SetNextItemWidth(100);
                ImGui::InputText(filterId.c_str(), section.filterBuf, sizeof(section.filterBuf));
                ImGui::SameLine();
                std::string clearId = "X##Clear_" + std::to_string(sectionIdx);
                if (ImGui::Button(clearId.c_str())) {
                    section.filterBuf[0] = '\0';
                }

                // File list
                std::string listId = "##FileList_" + std::to_string(sectionIdx);
                if (ImGui::BeginListBox(listId.c_str(), ImVec2(-1, fileSectionHeight))) {
                    for (int i = 0; i < static_cast<int>(section.files.size()); ++i) {
                        const auto& f = section.files[i];

                        // Apply filter
                        if (section.filterBuf[0]) {
                            std::string fLower = f;
                            std::string filterLower = section.filterBuf;
                            std::transform(fLower.begin(), fLower.end(), fLower.begin(), ::tolower);
                            std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);
                            if (fLower.find(filterLower) == std::string::npos) {
                                continue;
                            }
                        }

                        bool isSelected = (i == section.selectedFile);

                        // Create a unique selectable ID
                        std::string selectableId = f + "##" + std::to_string(sectionIdx) + "_" + std::to_string(i);

                        // Display file with Del button for selected item
                        if (isSelected && section.onDelete) {
                            // Calculate widths for layout
                            float contentWidth = ImGui::GetContentRegionAvail().x;
                            float buttonWidth = ImGui::CalcTextSize("Del").x + ImGui::GetStyle().FramePadding.x * 2;
                            float textWidth = contentWidth - buttonWidth - ImGui::GetStyle().ItemSpacing.x;

                            // File name (truncated if needed)
                            ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
                            if (ImGui::Selectable(selectableId.c_str(), true, 0, ImVec2(textWidth, 0))) {
                                // Already selected, clicking again
                            }
                            ImGui::PopStyleColor();

                            // Del button on same line
                            ImGui::SameLine();
                            std::string delId = "Del##" + std::to_string(sectionIdx) + "_" + std::to_string(i);
                            if (ImGui::SmallButton(delId.c_str())) {
                                std::string deletedFile = f;

                                // Remove from list
                                section.files.erase(section.files.begin() + i);
                                section.selectedFile = -1;

                                // Call delete callback
                                if (section.onDelete) {
                                    try {
                                        section.onDelete(deletedFile);
                                    } catch (const std::exception& e) {
                                        LOG_WARN("[PIDNavigator] Delete callback error: " << e.what());
                                    }
                                }

                                // Break out of loop since we modified the vector
                                ImGui::EndListBox();
                                ImGui::TreePop();
                                goto next_section;  // Continue to next section
                            }
                        } else {
                            if (ImGui::Selectable(selectableId.c_str(), isSelected)) {
                                section.selectedFile = i;

                                // Invoke file selection callback if set
                                if (section.onSelect) {
                                    const std::string& pid = pImpl->state.pidList[pImpl->state.selectedPID];
                                    std::string visit = pImpl->state.getVisitDir();
                                    std::string subdirectory = section.filter ? section.filter->getSubdirectory() : "";

                                    // Build URI using visit-based path
                                    std::string uri = "@pid:" + pid + "/" + visit;
                                    if (!subdirectory.empty()) {
                                        uri += "/" + subdirectory;
                                    }
                                    uri += "/" + f;

                                    try {
                                        auto handle = pImpl->resourceManager->fetch(uri);
                                        std::filesystem::path localPath = handle.local_path();
                                        section.onSelect(localPath.string(), f);
                                    } catch (const std::exception& e) {
                                        LOG_WARN("[PIDNavigator] Failed to fetch file: " << e.what());
                                    }
                                }
                            }
                        }
                    }
                    ImGui::EndListBox();
                }

                ImGui::TreePop();
            }
            next_section:;
        }
    }
}

} // namespace PIDNav
