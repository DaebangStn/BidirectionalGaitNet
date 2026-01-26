#include "PIDImGui.h"
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
// PIDNavigator::Impl - PIMPL Implementation
// ============================================================================

struct PIDNavigator::Impl {
    // Dependencies (non-owning)
    rm::ResourceManager* resourceManager;

    // Owned strategy
    std::unique_ptr<FileFilter> fileFilter;

    // State
    PIDSelectionState state;
    std::vector<std::string> files;
    int selectedFile = -1;

    // Filter buffers for ImGui
    char pidFilterBuf[64] = "";
    char fileFilterBuf[64] = "";

    // Callbacks
    FileSelectionCallback fileSelectionCallback;
    PIDChangeCallback pidChangeCallback;

    Impl(rm::ResourceManager* rm, std::unique_ptr<FileFilter> filter)
        : resourceManager(rm), fileFilter(std::move(filter)) {
        // Initialize default file filter from strategy
        if (fileFilter) {
            auto defaultFilter = fileFilter->getDefaultFilter();
            if (!defaultFilter.empty() && defaultFilter.size() < sizeof(fileFilterBuf)) {
                std::strcpy(fileFilterBuf, defaultFilter.c_str());
            }
        }
    }
};

// ============================================================================
// PIDNavigator Public Interface
// ============================================================================

PIDNavigator::PIDNavigator(rm::ResourceManager* rm,
                           std::unique_ptr<FileFilter> filter)
    : pImpl(std::make_unique<Impl>(rm, std::move(filter))) {
}

PIDNavigator::~PIDNavigator() = default;

PIDNavigator::PIDNavigator(PIDNavigator&&) noexcept = default;
PIDNavigator& PIDNavigator::operator=(PIDNavigator&&) noexcept = default;

void PIDNavigator::scanPIDs() {
    pImpl->state.pidList.clear();
    pImpl->state.pidNames.clear();
    pImpl->state.pidGMFCS.clear();
    pImpl->state.pidVisits.clear();
    pImpl->state.selectedPID = -1;
    pImpl->state.selectedVisit = 0;
    pImpl->files.clear();
    pImpl->selectedFile = -1;

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
            // This avoids scanning all PIDs at startup
        }
    } catch (const rm::RMError&) {
        // Silently handle errors - list will remain empty
    }
}

void PIDNavigator::scanFiles(const std::string& pid, const std::string& visit) {
    pImpl->files.clear();
    pImpl->selectedFile = -1;

    if (!pImpl->resourceManager || !pImpl->fileFilter || pid.empty()) return;

    std::string subdirectory = pImpl->fileFilter->getSubdirectory();

    // Build pattern using visit-based path: @pid:{pid}/{visit}/{subdirectory}
    std::string pattern = "@pid:" + pid + "/" + visit;
    if (!subdirectory.empty()) {
        pattern += "/" + subdirectory;
    }

    try {
        auto files = pImpl->resourceManager->list(pattern);
        for (const auto& file : files) {
            if (pImpl->fileFilter->matches(file)) {
                // Extract just the filename (list() returns filenames, not paths)
                std::filesystem::path p(file);
                pImpl->files.push_back(p.filename().string());
            }
        }
        std::sort(pImpl->files.begin(), pImpl->files.end());
    } catch (const rm::RMError&) {
        // Silently handle errors - list will remain empty
    }
}


const PIDSelectionState& PIDNavigator::getState() const {
    return pImpl->state;
}

const std::vector<std::string>& PIDNavigator::getFiles() const {
    return pImpl->files;
}

void PIDNavigator::setFileSelectionCallback(FileSelectionCallback callback) {
    pImpl->fileSelectionCallback = std::move(callback);
}

void PIDNavigator::setPIDChangeCallback(PIDChangeCallback callback) {
    pImpl->pidChangeCallback = std::move(callback);
}

void PIDNavigator::renderUI(const char* title,
                            float pidListHeight,
                            float fileListHeight,
                            bool defaultOpen) {
    if (!pImpl->resourceManager) return;

    ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0;
    if (ImGui::CollapsingHeader(title, flags)) {
        renderInlineSelector(pidListHeight, fileListHeight);
    }
}

void PIDNavigator::renderInlineSelector(float pidListHeight,
                                        float fileListHeight) {
    if (!pImpl->resourceManager) return;

    // PID Filter + Refresh
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##PIDFilter", pImpl->pidFilterBuf, sizeof(pImpl->pidFilterBuf));
    ImGui::SameLine();
    if (ImGui::Button("Refresh##PID")) {
        scanPIDs();
    }
    ImGui::SameLine();
    ImGui::Text("%zu PIDs", pImpl->state.pidList.size());

    // PID ListBox
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
                    scanFiles(pid, pImpl->state.getVisitDir());

                    // Invoke PID change callback if set
                    if (pImpl->pidChangeCallback) {
                        pImpl->pidChangeCallback(pid);
                    }
                }
            }
        }
        ImGui::EndListBox();
    }

    // Visit selection (pre/op1/op2)
    if (pImpl->state.selectedPID >= 0) {
        const auto& visits = pImpl->state.pidVisits[pImpl->state.selectedPID];
        for (int v = 0; v < static_cast<int>(visits.size()); ++v) {
            if (v > 0) ImGui::SameLine();
            if (ImGui::RadioButton(visits[v].c_str(), pImpl->state.selectedVisit == v)) {
                if (pImpl->state.selectedVisit != v) {
                    pImpl->state.selectedVisit = v;
                    pImpl->state.preOp = (visits[v] == "pre");  // Legacy compatibility
                    scanFiles(pImpl->state.pidList[pImpl->state.selectedPID], visits[v]);
                }
            }
        }
    }

    // Files section (only if fileListHeight > 0)
    if (fileListHeight > 0 && pImpl->state.selectedPID >= 0) {
        ImGui::Separator();
        ImGui::Text("Files: %zu", pImpl->files.size());

        ImGui::SetNextItemWidth(100);
        ImGui::InputText("##FileFilter", pImpl->fileFilterBuf, sizeof(pImpl->fileFilterBuf));
        ImGui::SameLine();
        if (ImGui::Button("X##FileClear")) {
            pImpl->fileFilterBuf[0] = '\0';
        }

        if (ImGui::BeginListBox("##FileList", ImVec2(-1, fileListHeight))) {
            for (int i = 0; i < static_cast<int>(pImpl->files.size()); ++i) {
                const auto& f = pImpl->files[i];
                if (pImpl->fileFilterBuf[0]) {
                    // Case-insensitive filtering
                    std::string fLower = f;
                    std::string filterLower = pImpl->fileFilterBuf;
                    std::transform(fLower.begin(), fLower.end(), fLower.begin(), ::tolower);
                    std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);
                    if (fLower.find(filterLower) == std::string::npos) {
                        continue;
                    }
                }

                if (ImGui::Selectable(f.c_str(), i == pImpl->selectedFile)) {
                    pImpl->selectedFile = i;

                    // Invoke file selection callback if set
                    if (pImpl->fileSelectionCallback) {
                        const std::string& pid = pImpl->state.pidList[pImpl->state.selectedPID];
                        std::string visit = pImpl->state.getVisitDir();
                        std::string subdirectory = pImpl->fileFilter->getSubdirectory();

                        // Build URI using visit-based path: @pid:{pid}/{visit}/{subdirectory}/{file}
                        std::string uri = "@pid:" + pid + "/" + visit;
                        if (!subdirectory.empty()) {
                            uri += "/" + subdirectory;
                        }
                        uri += "/" + f;

                        try {
                            auto handle = pImpl->resourceManager->fetch(uri);
                            std::filesystem::path localPath = handle.local_path();
                            pImpl->fileSelectionCallback(localPath.string(), f);
                        } catch (const rm::RMError&) {
                            // Silently handle fetch errors
                        }
                    }
                }
            }
            ImGui::EndListBox();
        }
    }
}

} // namespace PIDNav
