#pragma once

#include "PIDFileFilter.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rm {
class ResourceManager;
}

namespace PIDNav {

/**
 * POD structure holding PID selection state.
 * Maintains the list of PIDs with their metadata and current selection.
 */
struct PIDSelectionState {
    std::vector<std::string> pidList;
    std::vector<std::string> pidNames;
    std::vector<std::string> pidGMFCS;
    std::vector<std::vector<std::string>> pidVisits;  // Available visits per PID
    int selectedPID = -1;
    int selectedVisit = 0;  // Index into pidVisits[selectedPID]

    // Legacy compatibility
    bool preOp = true;  // Derived from selectedVisit == 0

    /**
     * Returns the currently selected PID string, or empty if none selected.
     */
    std::string getSelectedPID() const;

    /**
     * Returns the selected visit directory name (e.g., "pre", "op1", "op2").
     */
    std::string getVisitDir() const;

    /**
     * Returns "pre" or "post" based on current visit (legacy compatibility).
     * @deprecated Use getVisitDir() instead.
     */
    std::string getPrePostDir() const;
};

/**
 * Callback invoked when user selects a file.
 * Parameters: (resolvedPath, filename)
 */
using FileSelectionCallback = std::function<void(
    const std::string& resolvedPath,
    const std::string& filename
)>;

/**
 * Callback invoked when PID selection changes.
 * Parameter: pid (the selected PID string)
 */
using PIDChangeCallback = std::function<void(const std::string& pid)>;

/**
 * PID Navigator component for browsing clinical data.
 *
 * Manages PID list, file browsing with pluggable file type filters,
 * and renders ImGui UI for selection. Uses dependency injection for
 * ResourceManager and strategy pattern for file filtering.
 *
 * Usage:
 *   auto nav = std::make_unique<PIDNavigator>(
 *       resourceManager,
 *       std::make_unique<HDFFileFilter>()
 *   );
 *   nav->setFileSelectionCallback([](auto path, auto name) {
 *       loadFile(path);
 *   });
 *   nav->scanPIDs();
 *
 *   // In UI rendering:
 *   nav->renderUI("Clinical Data");
 */
class PIDNavigator {
public:
    /**
     * Constructs PID navigator with dependency injection.
     *
     * @param rm Non-owning pointer to ResourceManager singleton
     * @param filter File filter strategy (ownership transferred)
     */
    PIDNavigator(rm::ResourceManager* rm,
                 std::unique_ptr<FileFilter> filter);

    ~PIDNavigator();

    // Non-copyable, movable
    PIDNavigator(const PIDNavigator&) = delete;
    PIDNavigator& operator=(const PIDNavigator&) = delete;
    PIDNavigator(PIDNavigator&&) noexcept;
    PIDNavigator& operator=(PIDNavigator&&) noexcept;

    /**
     * Scans the resource manager for available PIDs and their metadata.
     * Populates the PID list, names, and GMFCS levels.
     */
    void scanPIDs();

    /**
     * Scans for files of the configured type in the specified PID/visit directory.
     *
     * @param pid The PID to scan (e.g., "12964246")
     * @param visit The visit directory (e.g., "pre", "op1", "op2")
     */
    void scanFiles(const std::string& pid, const std::string& visit);

    /**
     * Returns the current PID selection state (read-only).
     */
    const PIDSelectionState& getState() const;

    /**
     * Returns the current file list (read-only).
     */
    const std::vector<std::string>& getFiles() const;

    /**
     * Sets callback to invoke when user selects a file.
     */
    void setFileSelectionCallback(FileSelectionCallback callback);

    /**
     * Sets callback to invoke when PID selection changes.
     */
    void setPIDChangeCallback(PIDChangeCallback callback);

    /**
     * Renders full UI section with collapsing header.
     *
     * @param title Section title for collapsing header
     * @param pidListHeight Height of PID list box in pixels
     * @param fileListHeight Height of file list box in pixels
     * @param defaultOpen Whether the section is open by default
     */
    void renderUI(const char* title = "Clinical Data",
                  float pidListHeight = 150.0f,
                  float fileListHeight = 150.0f,
                  bool defaultOpen = false);

    /**
     * Renders inline selector content without collapsing header.
     * Suitable for tab-based UIs.
     *
     * @param pidListHeight Height of PID list box in pixels
     * @param fileListHeight Height of file list box in pixels
     */
    void renderInlineSelector(float pidListHeight = 120.0f,
                              float fileListHeight = 120.0f);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace PIDNav
