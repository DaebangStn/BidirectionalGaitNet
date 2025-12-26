#include "PIDNavigator.h"
#include <rm/rm.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>
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

std::string PIDSelectionState::getPrePostDir() const {
    return preOp ? "pre" : "post";
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
    pImpl->state.selectedPID = -1;
    pImpl->files.clear();
    pImpl->selectedFile = -1;

    if (!pImpl->resourceManager) return;

    try {
        auto entries = pImpl->resourceManager->list("@pid:");
        for (const auto& entry : entries) {
            pImpl->state.pidList.push_back(entry);
        }
        std::sort(pImpl->state.pidList.begin(), pImpl->state.pidList.end());

        pImpl->state.pidNames.resize(pImpl->state.pidList.size());
        pImpl->state.pidGMFCS.resize(pImpl->state.pidList.size());
        for (size_t i = 0; i < pImpl->state.pidList.size(); ++i) {
            try {
                auto h = pImpl->resourceManager->fetch("@pid:" + pImpl->state.pidList[i] + "/name");
                pImpl->state.pidNames[i] = h.as_string();
            } catch (...) {
                pImpl->state.pidNames[i] = "";
            }
            try {
                auto h = pImpl->resourceManager->fetch("@pid:" + pImpl->state.pidList[i] + "/gmfcs");
                pImpl->state.pidGMFCS[i] = h.as_string();
            } catch (...) {
                pImpl->state.pidGMFCS[i] = "";
            }
        }
    } catch (const rm::RMError&) {
        // Silently handle errors - list will remain empty
    }
}

void PIDNavigator::scanFiles(const std::string& pid, bool preOp) {
    pImpl->files.clear();
    pImpl->selectedFile = -1;

    if (!pImpl->resourceManager || !pImpl->fileFilter || pid.empty()) return;

    std::string prePost = preOp ? "pre" : "post";
    std::string subdirectory = pImpl->fileFilter->getSubdirectory();

    // Build pattern - handle empty subdirectory (e.g., C3D files)
    std::string pattern = "@pid:" + pid + "/gait/" + prePost;
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
                    scanFiles(pid, pImpl->state.preOp);

                    // Invoke PID change callback if set
                    if (pImpl->pidChangeCallback) {
                        pImpl->pidChangeCallback(pid);
                    }
                }
            }
        }
        ImGui::EndListBox();
    }

    // Pre/Post toggle
    if (ImGui::RadioButton("Pre-op", pImpl->state.preOp)) {
        if (!pImpl->state.preOp) {
            pImpl->state.preOp = true;
            if (pImpl->state.selectedPID >= 0) {
                scanFiles(pImpl->state.pidList[pImpl->state.selectedPID], true);
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Post-op", !pImpl->state.preOp)) {
        if (pImpl->state.preOp) {
            pImpl->state.preOp = false;
            if (pImpl->state.selectedPID >= 0) {
                scanFiles(pImpl->state.pidList[pImpl->state.selectedPID], false);
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
                if (pImpl->fileFilterBuf[0] && f.find(pImpl->fileFilterBuf) == std::string::npos) {
                    continue;
                }

                if (ImGui::Selectable(f.c_str(), i == pImpl->selectedFile)) {
                    pImpl->selectedFile = i;

                    // Invoke file selection callback if set
                    if (pImpl->fileSelectionCallback) {
                        const std::string& pid = pImpl->state.pidList[pImpl->state.selectedPID];
                        std::string prePost = pImpl->state.preOp ? "pre" : "post";
                        std::string subdirectory = pImpl->fileFilter->getSubdirectory();

                        // Build URI - handle empty subdirectory (e.g., C3D files)
                        std::string uri = "@pid:" + pid + "/gait/" + prePost;
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
